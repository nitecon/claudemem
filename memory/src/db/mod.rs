pub mod models;
pub mod queries;

use rusqlite::Connection;
use std::path::Path;

use crate::error::MemoryError;

pub fn open_database(db_path: &Path) -> Result<Connection, MemoryError> {
    let conn = Connection::open(db_path)?;

    // WAL mode for better concurrent read performance
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;

    run_migrations(&conn)?;

    Ok(conn)
}

pub(crate) fn run_migrations(conn: &Connection) -> Result<(), MemoryError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );",
    )?;

    let version: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    if version < 1 {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT,
                project TEXT,
                agent TEXT,
                source_file TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                embedding BLOB,
                memory_type TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent);
            CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at);

            INSERT OR IGNORE INTO schema_version (version) VALUES (1);",
        )?;
    }

    if version < 2 {
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content='memories',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;

            -- Populate FTS index from existing data
            INSERT INTO memories_fts(memories_fts) VALUES('rebuild');

            INSERT OR IGNORE INTO schema_version (version) VALUES (2);",
        )?;
    }

    if version < 3 {
        // Schema v3 — columns added to support `memory-dream` (Release 2).
        //
        // All four columns are nullable so pre-dream rows remain valid:
        //   * content_raw        — original verbatim text the user stored.
        //     When dream condenses a memory, the short form replaces `content`
        //     and the original moves here so nothing is lost.
        //   * superseded_by      — UUID of a newer memory that subsumes this
        //     one (dedup). Default reads filter `superseded_by IS NULL` so
        //     obsoleted rows stay in the DB for audit but don't surface in
        //     search, context, or list.
        //   * condenser_version  — stamp identifying the prompt/model combo
        //     that produced the current `content`. Lets a future dream pass
        //     detect stale condensations and re-run.
        //   * embedding_model    — name of the embedder used to compute
        //     `embedding`. Dream uses this so it only dedups rows that share
        //     a vector space.
        //
        // The FTS triggers are dropped and recreated to index
        // `content || ' ' || COALESCE(content_raw, '')` — otherwise terms the
        // condenser elided stop surfacing via lexical recall. The FTS table
        // itself stays single-column; the triggers concatenate before insert.
        conn.execute_batch(
            "ALTER TABLE memories ADD COLUMN content_raw TEXT;
             ALTER TABLE memories ADD COLUMN superseded_by TEXT;
             ALTER TABLE memories ADD COLUMN condenser_version TEXT;
             ALTER TABLE memories ADD COLUMN embedding_model TEXT;

             DROP TRIGGER IF EXISTS memories_fts_ai;
             DROP TRIGGER IF EXISTS memories_fts_ad;
             DROP TRIGGER IF EXISTS memories_fts_au;

             CREATE TRIGGER memories_fts_ai AFTER INSERT ON memories BEGIN
                 INSERT INTO memories_fts(rowid, content)
                     VALUES (new.rowid, new.content || ' ' || COALESCE(new.content_raw, ''));
             END;

             CREATE TRIGGER memories_fts_ad AFTER DELETE ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                     VALUES('delete', old.rowid, old.content || ' ' || COALESCE(old.content_raw, ''));
             END;

             CREATE TRIGGER memories_fts_au AFTER UPDATE OF content, content_raw ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                     VALUES('delete', old.rowid, old.content || ' ' || COALESCE(old.content_raw, ''));
                 INSERT INTO memories_fts(rowid, content)
                     VALUES (new.rowid, new.content || ' ' || COALESCE(new.content_raw, ''));
             END;

             -- Rebuild so rows inserted pre-v3 reindex under the new trigger body.
             INSERT INTO memories_fts(memories_fts) VALUES('rebuild');

             CREATE INDEX IF NOT EXISTS idx_memories_superseded_by
                 ON memories(superseded_by);

             INSERT OR IGNORE INTO schema_version (version) VALUES (3);",
        )?;
    }

    if version < 4 {
        // Schema v4 — per-project dream state (Release 2.3).
        //
        // The agentic dream pass runs incrementally. Each project's
        // `last_dream_at` records the wall-clock instant we last curated
        // memories for that project, so the next pass can pull only rows
        // with `updated_at > last_dream_at` OR a stale `condenser_version`.
        //
        // Absent row = "never dreamed"; dream treats the timestamp as epoch
        // on the first pass. Stored as RFC3339 so the column survives
        // timezone migrations. No backfill needed — pre-v4 DBs simply have
        // no rows, so every project re-walks once on the first upgraded
        // dream pass (expected behavior).
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS project_state (
                 project TEXT PRIMARY KEY,
                 last_dream_at TEXT NOT NULL
             );
             INSERT OR IGNORE INTO schema_version (version) VALUES (4);",
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod migration_tests {
    use super::*;

    /// Simulates a DB created at schema v2 (pre-Release-2), then runs
    /// `run_migrations` against it to confirm the v3 upgrade applies cleanly,
    /// new columns read as NULL on pre-existing rows, and the FTS index is
    /// rebuilt with the concatenated-content trigger body.
    #[test]
    fn v2_database_upgrades_to_v3_preserving_existing_rows() {
        let conn = Connection::open_in_memory().expect("open in-memory db");

        // Hand-construct a v2 schema (what the DB looked like before Release 2).
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             CREATE TABLE memories (
                 id TEXT PRIMARY KEY,
                 content TEXT NOT NULL,
                 tags TEXT,
                 project TEXT,
                 agent TEXT,
                 source_file TEXT,
                 created_at TEXT NOT NULL,
                 updated_at TEXT NOT NULL,
                 access_count INTEGER DEFAULT 0,
                 embedding BLOB,
                 memory_type TEXT
             );
             CREATE VIRTUAL TABLE memories_fts USING fts5(
                 content, content='memories', content_rowid='rowid',
                 tokenize='porter unicode61'
             );
             CREATE TRIGGER memories_fts_ai AFTER INSERT ON memories BEGIN
                 INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
             END;
             CREATE TRIGGER memories_fts_ad AFTER DELETE ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                     VALUES('delete', old.rowid, old.content);
             END;
             CREATE TRIGGER memories_fts_au AFTER UPDATE OF content ON memories BEGIN
                 INSERT INTO memories_fts(memories_fts, rowid, content)
                     VALUES('delete', old.rowid, old.content);
                 INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
             END;
             INSERT INTO schema_version (version) VALUES (1);
             INSERT INTO schema_version (version) VALUES (2);
             INSERT INTO memories (id, content, created_at, updated_at)
                 VALUES ('legacy-row', 'existing v2 content', '2026-01-01', '2026-01-01');",
        )
        .expect("seed v2 db");

        // Apply migrations — should run v3 and v4 steps in sequence.
        run_migrations(&conn).expect("migrate to v4");

        // Schema version advanced to latest (v4).
        let max_v: i64 = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .expect("query schema_version");
        assert_eq!(max_v, 4);

        // New columns are present and NULL on the pre-existing row.
        let (raw, sup, cond, emb): (
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        ) = conn
            .query_row(
                "SELECT content_raw, superseded_by, condenser_version, embedding_model
                 FROM memories WHERE id = 'legacy-row'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("select new columns");
        assert!(raw.is_none());
        assert!(sup.is_none());
        assert!(cond.is_none());
        assert!(emb.is_none());

        // FTS rebuild ran — the pre-existing content should still be searchable.
        let hit_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'existing'",
                [],
                |row| row.get(0),
            )
            .expect("fts query");
        assert_eq!(hit_count, 1);
    }

    /// Fresh DB path: no prior schema_version rows, run_migrations should
    /// apply every step (1, 2, 3, 4) and leave an empty but well-formed DB.
    #[test]
    fn fresh_database_applies_all_migrations() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        run_migrations(&conn).expect("migrate fresh");
        let max_v: i64 = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .expect("query schema_version");
        assert_eq!(max_v, 4);
    }

    /// v4 migration from a v3 fixture DB must create the `project_state`
    /// table without disturbing existing memory rows. Mirrors the v2→v3 test
    /// so we exercise the upgrade path for each release independently.
    #[test]
    fn v3_database_upgrades_to_v4_creating_project_state() {
        let conn = Connection::open_in_memory().expect("open in-memory db");

        // Hand-construct a v3 schema + a memory row so we can verify the v4
        // migration adds `project_state` without touching existing data.
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             CREATE TABLE memories (
                 id TEXT PRIMARY KEY, content TEXT NOT NULL,
                 tags TEXT, project TEXT, agent TEXT, source_file TEXT,
                 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                 access_count INTEGER DEFAULT 0, embedding BLOB, memory_type TEXT,
                 content_raw TEXT, superseded_by TEXT,
                 condenser_version TEXT, embedding_model TEXT
             );
             CREATE VIRTUAL TABLE memories_fts USING fts5(
                 content, content='memories', content_rowid='rowid',
                 tokenize='porter unicode61'
             );
             INSERT INTO schema_version (version) VALUES (1);
             INSERT INTO schema_version (version) VALUES (2);
             INSERT INTO schema_version (version) VALUES (3);
             INSERT INTO memories (id, content, project, created_at, updated_at)
                 VALUES ('existing-row', 'body', 'agent-memory', '2026-01-01', '2026-01-01');",
        )
        .expect("seed v3 db");

        run_migrations(&conn).expect("migrate to v4");

        // project_state now exists and is empty (no backfill).
        let row_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM project_state", [], |row| row.get(0))
            .expect("query project_state");
        assert_eq!(row_count, 0);

        // Schema version advanced exactly one step.
        let max_v: i64 = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .expect("query schema_version");
        assert_eq!(max_v, 4);

        // Existing memory row survived untouched.
        let existing: String = conn
            .query_row(
                "SELECT content FROM memories WHERE id = 'existing-row'",
                [],
                |row| row.get(0),
            )
            .expect("select existing row");
        assert_eq!(existing, "body");
    }

    /// Content + content_raw are concatenated in the FTS index so terms that
    /// only appear in the raw field remain lexically searchable after dream
    /// condenses a memory.
    #[test]
    fn fts_triggers_index_content_raw_after_v3() {
        let conn = Connection::open_in_memory().expect("open db");
        run_migrations(&conn).expect("migrate");

        conn.execute(
            "INSERT INTO memories (id, content, content_raw, created_at, updated_at)
             VALUES ('x', 'short summary', 'full verbatim needle in raw', '2026-01-01', '2026-01-01')",
            [],
        )
        .expect("insert row with content_raw");

        // 'needle' is only in content_raw — FTS must still find it.
        let needle_hits: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'needle'",
                [],
                |row| row.get(0),
            )
            .expect("fts query");
        assert_eq!(needle_hits, 1);

        // 'summary' is only in content — FTS finds it too.
        let summary_hits: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'summary'",
                [],
                |row| row.get(0),
            )
            .expect("fts query");
        assert_eq!(summary_hits, 1);
    }
}
