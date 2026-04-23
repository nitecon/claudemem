use rusqlite::{params, Connection};

use crate::db::models::{blob_to_embedding, embedding_to_blob, Memory};
use crate::error::MemoryError;

/// Minimum length accepted by [`resolve_id_prefix`].
///
/// 4 hex characters gives 65,536 distinct prefixes — short enough for humans
/// to type but long enough that ambiguity with a typical memory DB is still
/// rare. Shorter prefixes are rejected with `MemoryError::Config` so callers
/// don't accidentally trigger massive candidate-list returns on 2-char input.
pub const MIN_ID_PREFIX_LEN: usize = 4;

/// Column list used by every SELECT that materializes a full [`Memory`].
///
/// Keeping this as a single source of truth means schema additions (like the
/// v3 dream columns) require editing exactly one string constant plus
/// [`map_memory_row`]; the dozens of SELECTs in this file don't drift apart.
///
/// Positional layout (0-indexed):
///   0  id                  5  source_file         10 memory_type
///   1  content             6  created_at          11 content_raw
///   2  tags                7  updated_at          12 superseded_by
///   3  project             8  access_count        13 condenser_version
///   4  agent               9  embedding           14 embedding_model
pub const MEMORY_COLS: &str = "id, content, tags, project, agent, source_file, \
created_at, updated_at, access_count, embedding, memory_type, \
content_raw, superseded_by, condenser_version, embedding_model";

/// Filter clause appended to every *default* read path.
///
/// Dream marks obsoleted memories with `superseded_by = <newer-id>` instead of
/// deleting them (audit trail). The default CLI surface must not surface them,
/// so every non-admin query adds this filter. Helpers meant for `memory-dream`
/// itself (e.g. the raw walk over every row) bypass this filter deliberately.
pub const DEFAULT_VISIBILITY_FILTER: &str = "superseded_by IS NULL";

/// Row mapper shared by every SELECT in this module. Expects the columns in
/// the order defined by [`MEMORY_COLS`].
fn map_memory_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    let tags_str: Option<String> = row.get(2)?;
    let embedding_blob: Option<Vec<u8>> = row.get(9)?;
    Ok(Memory {
        id: row.get(0)?,
        content: row.get(1)?,
        tags: tags_str.and_then(|s| serde_json::from_str(&s).ok()),
        project: row.get(3)?,
        agent: row.get(4)?,
        source_file: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        access_count: row.get(8)?,
        embedding: embedding_blob.map(|b| blob_to_embedding(&b)),
        memory_type: row.get(10)?,
        content_raw: row.get(11)?,
        superseded_by: row.get(12)?,
        condenser_version: row.get(13)?,
        embedding_model: row.get(14)?,
    })
}

/// Result of a short-ID prefix lookup.
///
/// Emitted by [`resolve_id_prefix`] so every CLI subcommand that takes an `id`
/// argument can use a uniform short-prefix flow. The `Ambiguous` variant
/// carries the candidate list so the caller can render a disambiguation
/// prompt without re-querying.
#[derive(Debug)]
pub enum ResolvedId {
    /// Prefix or full UUID uniquely matched exactly one memory. The full UUID
    /// is carried so callers can pass it straight to the other query helpers.
    Exact(String),
    /// Multiple memories share the prefix. The vec holds enough metadata to
    /// render a human-pickable list (no embeddings — we don't need them to
    /// disambiguate).
    Ambiguous(Vec<Memory>),
    /// No memory in the DB starts with the given prefix.
    NotFound,
}

/// Expand a short ID prefix to a full UUID.
///
/// Accepts any prefix of length ≥ [`MIN_ID_PREFIX_LEN`]; a full UUID still
/// returns `Exact` (fast path below). If the prefix is shorter than the
/// minimum the function returns `MemoryError::Config` rather than trying to
/// resolve it — very short prefixes can match arbitrarily many rows.
///
/// Superseded rows (`superseded_by IS NOT NULL`) are still resolvable by
/// prefix so `memory get <id>` can inspect an audit row by exact lookup, but
/// they're excluded from the ambiguous-candidate list so the common case
/// doesn't surface dead rows to the user.
pub fn resolve_id_prefix(conn: &Connection, prefix: &str) -> Result<ResolvedId, MemoryError> {
    if prefix.len() < MIN_ID_PREFIX_LEN {
        return Err(MemoryError::Config(format!(
            "ID prefix must be at least {MIN_ID_PREFIX_LEN} characters (got {})",
            prefix.len()
        )));
    }

    // Fast path: the caller might already have the full UUID. This path
    // intentionally does NOT filter superseded rows — `memory get <full-uuid>`
    // on an audit row should still return it.
    if let Ok(m) = get_memory_by_id(conn, prefix) {
        return Ok(ResolvedId::Exact(m.id));
    }

    // Prefix match. Use a bounded LIMIT so a very short accidental prefix
    // cannot blow the heap; `MAX_PREFIX_CANDIDATES` + 1 so we can detect
    // "more candidates exist than we're willing to list" and keep the
    // disambiguation output short. Only live rows participate in
    // disambiguation so dream-obsoleted rows don't confuse pickers.
    const MAX_PREFIX_CANDIDATES: usize = 16;
    let like_pattern = format!("{prefix}%");
    let sql = format!(
        "SELECT {MEMORY_COLS} FROM memories \
         WHERE id LIKE ?1 AND {DEFAULT_VISIBILITY_FILTER} LIMIT ?2"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(
        params![like_pattern, (MAX_PREFIX_CANDIDATES + 1) as i64],
        map_memory_row,
    )?;
    let mut candidates: Vec<Memory> = Vec::new();
    for row in rows {
        candidates.push(row?);
    }

    match candidates.len() {
        0 => Ok(ResolvedId::NotFound),
        1 => Ok(ResolvedId::Exact(candidates.remove(0).id)),
        _ => Ok(ResolvedId::Ambiguous(candidates)),
    }
}

pub fn insert_memory(conn: &Connection, memory: &Memory) -> Result<(), MemoryError> {
    let tags_json = memory
        .tags
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;

    let embedding_blob = memory.embedding.as_ref().map(|e| embedding_to_blob(e));

    conn.execute(
        "INSERT INTO memories (id, content, tags, project, agent, source_file,
         created_at, updated_at, access_count, embedding, memory_type,
         content_raw, superseded_by, condenser_version, embedding_model)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            memory.id,
            memory.content,
            tags_json,
            memory.project,
            memory.agent,
            memory.source_file,
            memory.created_at,
            memory.updated_at,
            memory.access_count,
            embedding_blob,
            memory.memory_type,
            memory.content_raw,
            memory.superseded_by,
            memory.condenser_version,
            memory.embedding_model,
        ],
    )?;
    Ok(())
}

pub fn get_memory_by_id(conn: &Connection, id: &str) -> Result<Memory, MemoryError> {
    // NB: `get_memory_by_id` intentionally does NOT apply the default
    // visibility filter. Callers that want that behavior should use
    // `resolve_id_prefix` (which filters for prefix lookups but still
    // returns exact UUID matches on audit rows for diagnostic access).
    let sql = format!("SELECT {MEMORY_COLS} FROM memories WHERE id = ?1");
    let mut stmt = conn.prepare(&sql)?;
    stmt.query_row(params![id], map_memory_row)
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                MemoryError::NotFound(format!("Memory {id} not found"))
            }
            other => MemoryError::Database(other),
        })
}

pub fn list_memories(
    conn: &Connection,
    project: Option<&str>,
    agent: Option<&str>,
    tags: Option<&[String]>,
    memory_type: Option<&str>,
    limit: usize,
) -> Result<Vec<Memory>, MemoryError> {
    let mut sql = format!("SELECT {MEMORY_COLS} FROM memories WHERE {DEFAULT_VISIBILITY_FILTER}");
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(p) = project {
        sql.push_str(&format!(" AND project = ?{}", param_values.len() + 1));
        param_values.push(Box::new(p.to_string()));
    }
    if let Some(a) = agent {
        sql.push_str(&format!(" AND agent = ?{}", param_values.len() + 1));
        param_values.push(Box::new(a.to_string()));
    }
    if let Some(mt) = memory_type {
        sql.push_str(&format!(" AND memory_type = ?{}", param_values.len() + 1));
        param_values.push(Box::new(mt.to_string()));
    }
    if let Some(tag_list) = tags {
        for tag in tag_list {
            sql.push_str(&format!(" AND tags LIKE ?{}", param_values.len() + 1));
            param_values.push(Box::new(format!("%\"{tag}\"%")));
        }
    }

    sql.push_str(&format!(
        " ORDER BY updated_at DESC LIMIT ?{}",
        param_values.len() + 1
    ));
    param_values.push(Box::new(limit as i64));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(param_refs.as_slice(), map_memory_row)?;

    let mut memories = Vec::new();
    for row in rows {
        memories.push(row?);
    }
    Ok(memories)
}

pub fn get_all_embeddings(conn: &Connection) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
    // Hybrid-search only considers live rows — superseded memories would
    // pollute the vector ranking with stale content otherwise.
    let sql = format!(
        "SELECT id, embedding FROM memories \
         WHERE embedding IS NOT NULL AND {DEFAULT_VISIBILITY_FILTER}"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        Ok((id, blob_to_embedding(&blob)))
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

pub fn delete_memory(conn: &Connection, id: &str) -> Result<bool, MemoryError> {
    let changed = conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
    Ok(changed > 0)
}

pub fn increment_access(conn: &Connection, ids: &[String]) -> Result<(), MemoryError> {
    let mut stmt = conn.prepare(
        "UPDATE memories SET access_count = access_count + 1, updated_at = ?1 WHERE id = ?2",
    )?;
    let now = chrono::Utc::now().to_rfc3339();
    for id in ids {
        stmt.execute(params![now, id])?;
    }
    Ok(())
}

/// Reassign the `project` column on a single memory. Returns true if the row existed.
pub fn move_memory_by_id(
    conn: &Connection,
    id: &str,
    new_project: Option<&str>,
) -> Result<bool, MemoryError> {
    let now = chrono::Utc::now().to_rfc3339();
    let changed = conn.execute(
        "UPDATE memories SET project = ?1, updated_at = ?2 WHERE id = ?3",
        params![new_project, now, id],
    )?;
    Ok(changed > 0)
}

/// Reassign the `project` column for every memory currently tagged with `from`.
/// `from` may be `None` to match memories with a NULL project. Returns the
/// number of rows updated.
pub fn move_memories_by_project(
    conn: &Connection,
    from: Option<&str>,
    new_project: Option<&str>,
) -> Result<usize, MemoryError> {
    let now = chrono::Utc::now().to_rfc3339();
    let changed = match from {
        Some(f) => conn.execute(
            "UPDATE memories SET project = ?1, updated_at = ?2 WHERE project = ?3",
            params![new_project, now, f],
        )?,
        None => conn.execute(
            "UPDATE memories SET project = ?1, updated_at = ?2 WHERE project IS NULL",
            params![new_project, now],
        )?,
    };
    Ok(changed)
}

/// Preview rows that `move_memories_by_project` would update.
pub fn list_memories_by_project(
    conn: &Connection,
    project: Option<&str>,
) -> Result<Vec<Memory>, MemoryError> {
    let (sql, bind_project) = match project {
        Some(_) => (
            format!(
                "SELECT {MEMORY_COLS} FROM memories \
                 WHERE project = ?1 AND {DEFAULT_VISIBILITY_FILTER}"
            ),
            true,
        ),
        None => (
            format!(
                "SELECT {MEMORY_COLS} FROM memories \
                 WHERE project IS NULL AND {DEFAULT_VISIBILITY_FILTER}"
            ),
            false,
        ),
    };
    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<Memory> = if bind_project {
        stmt.query_map(params![project.unwrap()], map_memory_row)?
            .collect::<rusqlite::Result<_>>()?
    } else {
        stmt.query_map([], map_memory_row)?
            .collect::<rusqlite::Result<_>>()?
    };
    Ok(rows)
}

/// Duplicate a memory under a new project ident. Preserves content, tags,
/// agent, source_file, memory_type, and the embedding. A new UUID is minted
/// and created_at/updated_at are set to now; access_count resets to 0.
/// Returns the new memory ID.
pub fn copy_memory_by_id(
    conn: &Connection,
    src_id: &str,
    new_project: Option<&str>,
) -> Result<String, MemoryError> {
    let src = get_memory_by_id(conn, src_id)?;
    let mut copy = Memory::new(
        src.content,
        src.tags,
        new_project.map(|s| s.to_string()),
        src.agent,
        src.source_file,
        src.memory_type,
    );
    copy.embedding = src.embedding;
    // Preserve dream-produced metadata so copies don't look un-condensed.
    copy.content_raw = src.content_raw;
    copy.condenser_version = src.condenser_version;
    copy.embedding_model = src.embedding_model;
    insert_memory(conn, &copy)?;
    Ok(copy.id)
}

/// Duplicate every memory tagged with `from` under `new_project`. Returns the
/// newly-created memory IDs.
pub fn copy_memories_by_project(
    conn: &Connection,
    from: Option<&str>,
    new_project: Option<&str>,
) -> Result<Vec<String>, MemoryError> {
    let sources = list_memories_by_project(conn, from)?;
    let mut new_ids = Vec::with_capacity(sources.len());
    for src in sources {
        let mut copy = Memory::new(
            src.content,
            src.tags,
            new_project.map(|s| s.to_string()),
            src.agent,
            src.source_file,
            src.memory_type,
        );
        copy.embedding = src.embedding;
        copy.content_raw = src.content_raw;
        copy.condenser_version = src.condenser_version;
        copy.embedding_model = src.embedding_model;
        insert_memory(conn, &copy)?;
        new_ids.push(copy.id);
    }
    Ok(new_ids)
}

/// List distinct project idents with the number of memories tagged under each.
/// A NULL project is reported as `None`. Ordered by count DESC. Only counts
/// live (non-superseded) rows so dream-obsoleted memories don't inflate the
/// numbers a user sees.
pub fn list_projects(conn: &Connection) -> Result<Vec<(Option<String>, i64)>, MemoryError> {
    let sql = format!(
        "SELECT project, COUNT(*) as cnt FROM memories \
         WHERE {DEFAULT_VISIBILITY_FILTER} \
         GROUP BY project ORDER BY cnt DESC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let project: Option<String> = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok((project, count))
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

pub fn prune_memories(
    conn: &Connection,
    max_age_days: u64,
    min_access_count: i64,
    dry_run: bool,
) -> Result<Vec<Memory>, MemoryError> {
    let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days as i64);
    let cutoff_str = cutoff.to_rfc3339();

    // Prune operates on live rows only — superseded rows get garbage-collected
    // by a separate dream-aware GC pass (future work) rather than via the
    // age-based prune flow.
    let sql = format!(
        "SELECT {MEMORY_COLS} FROM memories \
         WHERE updated_at < ?1 AND access_count <= ?2 AND {DEFAULT_VISIBILITY_FILTER}"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![cutoff_str, min_access_count], |row| {
        // Prune's consumer doesn't need the embedding — drop it to save heap.
        let mut m = map_memory_row(row)?;
        m.embedding = None;
        Ok(m)
    })?;

    let mut to_prune = Vec::new();
    for row in rows {
        to_prune.push(row?);
    }

    if !dry_run {
        conn.execute(
            "DELETE FROM memories WHERE updated_at < ?1 AND access_count <= ?2
             AND superseded_by IS NULL",
            params![cutoff_str, min_access_count],
        )?;
    }

    Ok(to_prune)
}

// -- Dream-support helpers ---------------------------------------------------
//
// These functions are public because `memory-dream` needs them from an
// external crate, but they deliberately bypass `DEFAULT_VISIBILITY_FILTER`:
// dream walks the entire table including already-processed rows. Regular CLI
// paths should continue to use the filtered helpers above.
//
// `#[allow(dead_code)]` lives on each — the `memory` binary target never
// calls these directly (they're a library surface for `memory-dream`). The
// lib crate exposes them; the bin's dead-code scan flags them when built
// as a binary, so the annotation silences the bin's false-positive warning
// without hiding any real dead code.

/// Iterate every memory by `updated_at ASC` (oldest first). Used by the dream
/// orchestrator so early memories are condensed / deduped before younger ones
/// can supersede them. Includes superseded rows — the caller is expected to
/// skip them explicitly if desired.
///
/// `limit` caps the walk for incremental runs; `0` means "no limit".
#[allow(dead_code)]
pub fn list_all_for_dream(conn: &Connection, limit: usize) -> Result<Vec<Memory>, MemoryError> {
    let sql = if limit == 0 {
        format!("SELECT {MEMORY_COLS} FROM memories ORDER BY updated_at ASC")
    } else {
        format!("SELECT {MEMORY_COLS} FROM memories ORDER BY updated_at ASC LIMIT {limit}")
    };
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], map_memory_row)?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

/// Fetch dedup candidates for a given memory. Candidates share the same
/// `project`, `memory_type`, and `embedding_model` (so cosine similarity
/// compares apples-to-apples), are not already superseded, and are not the
/// source row itself.
///
/// NULL semantics: a `None` project matches other NULL-project rows; same for
/// `memory_type` and `embedding_model`. This lets dream handle mixed-scope
/// DBs cleanly even when one axis is missing.
#[allow(dead_code)]
pub fn list_dedup_candidates(
    conn: &Connection,
    source_id: &str,
    project: Option<&str>,
    memory_type: Option<&str>,
    embedding_model: Option<&str>,
) -> Result<Vec<Memory>, MemoryError> {
    // Build a predicate that matches IS NULL when the axis is None, and = ?
    // when it's Some. We use param indexing to dodge SQL injection.
    let mut sql = format!(
        "SELECT {MEMORY_COLS} FROM memories \
         WHERE id != ?1 AND {DEFAULT_VISIBILITY_FILTER}"
    );
    let mut bind: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(source_id.to_string())];

    match project {
        Some(p) => {
            sql.push_str(&format!(" AND project = ?{}", bind.len() + 1));
            bind.push(Box::new(p.to_string()));
        }
        None => sql.push_str(" AND project IS NULL"),
    }
    match memory_type {
        Some(t) => {
            sql.push_str(&format!(" AND memory_type = ?{}", bind.len() + 1));
            bind.push(Box::new(t.to_string()));
        }
        None => sql.push_str(" AND memory_type IS NULL"),
    }
    match embedding_model {
        Some(m) => {
            sql.push_str(&format!(" AND embedding_model = ?{}", bind.len() + 1));
            bind.push(Box::new(m.to_string()));
        }
        None => sql.push_str(" AND embedding_model IS NULL"),
    }

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = bind.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(param_refs.as_slice(), map_memory_row)?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

/// Persist a dream-condensed memory: moves the original text into
/// `content_raw`, replaces `content` with the short form, stamps the
/// condenser version and embedding metadata, and refreshes the embedding blob.
///
/// Callers must wrap this in a `BEGIN IMMEDIATE` transaction so concurrent
/// CLI writes can't race the dream pass.
#[allow(dead_code)]
pub fn update_condensation(
    conn: &Connection,
    id: &str,
    new_content: &str,
    new_content_raw: &str,
    condenser_version: &str,
    embedding: &[f32],
    embedding_model: &str,
) -> Result<(), MemoryError> {
    let now = chrono::Utc::now().to_rfc3339();
    let blob = embedding_to_blob(embedding);
    conn.execute(
        "UPDATE memories SET
             content = ?1,
             content_raw = ?2,
             condenser_version = ?3,
             embedding = ?4,
             embedding_model = ?5,
             updated_at = ?6
         WHERE id = ?7",
        params![
            new_content,
            new_content_raw,
            condenser_version,
            blob,
            embedding_model,
            now,
            id
        ],
    )?;
    Ok(())
}

/// Mark `older_id` as superseded by `newer_id`. Call inside a transaction.
/// Idempotent: re-marking an already-superseded row is a no-op conceptually
/// but updates the pointer to the latest newer_id, which is fine.
#[allow(dead_code)]
pub fn mark_superseded(
    conn: &Connection,
    older_id: &str,
    newer_id: &str,
) -> Result<(), MemoryError> {
    conn.execute(
        "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
        params![newer_id, older_id],
    )?;
    Ok(())
}

// -- Tests ------------------------------------------------------------------
// Lives at the end of the file so clippy's `items_after_test_module` lint is
// happy. Each test opens an in-memory SQLite connection with the full
// migration set, mirroring the behavior callers get from `open_database`.

#[cfg(test)]
mod resolve_id_tests {
    use super::*;
    use crate::db::models::Memory;
    use crate::db::run_migrations;

    fn fresh_db() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory");
        run_migrations(&conn).expect("migrate");
        conn
    }

    fn insert(conn: &Connection, id: &str) {
        let mut m = Memory::new(
            format!("content {id}"),
            None,
            Some("test".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        m.id = id.to_string();
        insert_memory(conn, &m).expect("insert");
    }

    #[test]
    fn exact_match_on_full_uuid() {
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-333333333333");
        let r = resolve_id_prefix(&conn, "aaaaaaaa-0000-1111-2222-333333333333").unwrap();
        match r {
            ResolvedId::Exact(id) => assert_eq!(id, "aaaaaaaa-0000-1111-2222-333333333333"),
            other => panic!("expected Exact, got {other:?}"),
        }
    }

    #[test]
    fn exact_match_on_short_prefix() {
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-333333333333");
        insert(&conn, "bbbbbbbb-0000-1111-2222-333333333333");
        let r = resolve_id_prefix(&conn, "aaaaaaaa").unwrap();
        match r {
            ResolvedId::Exact(id) => assert_eq!(id, "aaaaaaaa-0000-1111-2222-333333333333"),
            other => panic!("expected Exact, got {other:?}"),
        }
    }

    #[test]
    fn ambiguous_returns_all_candidates() {
        let conn = fresh_db();
        insert(&conn, "4c82c482-c081-4937-aaaa-111111111111");
        insert(&conn, "4c82c482-d7f2-4a18-bbbb-222222222222");
        let r = resolve_id_prefix(&conn, "4c82c482").unwrap();
        match r {
            ResolvedId::Ambiguous(cands) => {
                assert_eq!(cands.len(), 2);
                let ids: Vec<_> = cands.iter().map(|m| m.id.clone()).collect();
                assert!(ids.contains(&"4c82c482-c081-4937-aaaa-111111111111".to_string()));
                assert!(ids.contains(&"4c82c482-d7f2-4a18-bbbb-222222222222".to_string()));
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn not_found_when_no_match() {
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-333333333333");
        let r = resolve_id_prefix(&conn, "deadbeef").unwrap();
        assert!(matches!(r, ResolvedId::NotFound));
    }

    #[test]
    fn prefix_too_short_returns_config_error() {
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-333333333333");
        let err = resolve_id_prefix(&conn, "aaa").unwrap_err();
        match err {
            MemoryError::Config(msg) => {
                assert!(msg.contains("at least 4"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn superseded_row_hidden_from_prefix_ambiguity() {
        // When two rows share a prefix but one is superseded, prefix
        // resolution should treat the live row as the unambiguous match
        // rather than surfacing an `Ambiguous` result.
        let conn = fresh_db();
        insert(&conn, "4c82c482-aaaa-1111-2222-333333333333");
        insert(&conn, "4c82c482-bbbb-1111-2222-444444444444");
        mark_superseded(
            &conn,
            "4c82c482-aaaa-1111-2222-333333333333",
            "4c82c482-bbbb-1111-2222-444444444444",
        )
        .unwrap();

        let r = resolve_id_prefix(&conn, "4c82c482").unwrap();
        match r {
            ResolvedId::Exact(id) => {
                assert_eq!(id, "4c82c482-bbbb-1111-2222-444444444444");
            }
            other => panic!("expected Exact (only live row), got {other:?}"),
        }
    }

    #[test]
    fn default_reads_filter_superseded_rows() {
        // list_memories / list_memories_by_project / list_projects all apply
        // the default visibility filter. A superseded row should vanish from
        // each of them.
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-000000000001");
        insert(&conn, "bbbbbbbb-0000-1111-2222-000000000002");
        mark_superseded(
            &conn,
            "aaaaaaaa-0000-1111-2222-000000000001",
            "bbbbbbbb-0000-1111-2222-000000000002",
        )
        .unwrap();

        let all = list_memories(&conn, None, None, None, None, 100).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "bbbbbbbb-0000-1111-2222-000000000002");

        let by_project = list_memories_by_project(&conn, Some("test")).unwrap();
        assert_eq!(by_project.len(), 1);

        let projects = list_projects(&conn).unwrap();
        let (_p, n) = &projects[0];
        assert_eq!(*n, 1, "superseded row should not inflate project count");
    }

    #[test]
    fn list_all_for_dream_includes_superseded() {
        // Dream's walk must see every row, superseded or not — that's how
        // it can re-run its pass after a prompt change without missing
        // previously-processed memories.
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-000000000001");
        insert(&conn, "bbbbbbbb-0000-1111-2222-000000000002");
        mark_superseded(
            &conn,
            "aaaaaaaa-0000-1111-2222-000000000001",
            "bbbbbbbb-0000-1111-2222-000000000002",
        )
        .unwrap();

        let all = list_all_for_dream(&conn, 0).unwrap();
        assert_eq!(all.len(), 2, "dream walk must see superseded rows");
    }

    #[test]
    fn list_dedup_candidates_shares_axis_and_excludes_source() {
        let conn = fresh_db();

        // Three rows, all under the same project/type/embedding_model axis.
        let id_source = "11111111-0000-1111-2222-000000000001";
        let id_same_axis = "22222222-0000-1111-2222-000000000002";
        let id_other_project = "33333333-0000-1111-2222-000000000003";

        let mut a = Memory::new(
            "source".to_string(),
            None,
            Some("proj".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        a.id = id_source.to_string();
        a.embedding_model = Some("mini".to_string());
        insert_memory(&conn, &a).unwrap();

        let mut b = Memory::new(
            "peer".to_string(),
            None,
            Some("proj".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        b.id = id_same_axis.to_string();
        b.embedding_model = Some("mini".to_string());
        insert_memory(&conn, &b).unwrap();

        let mut c = Memory::new(
            "other".to_string(),
            None,
            Some("other-proj".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        c.id = id_other_project.to_string();
        c.embedding_model = Some("mini".to_string());
        insert_memory(&conn, &c).unwrap();

        let cands =
            list_dedup_candidates(&conn, id_source, Some("proj"), Some("user"), Some("mini"))
                .unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].id, id_same_axis);
    }

    #[test]
    fn update_condensation_moves_content_and_stamps_metadata() {
        let conn = fresh_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id);

        update_condensation(
            &conn,
            id,
            "condensed form",
            "original verbose raw content",
            "gemma3:abcd",
            &[0.1_f32, 0.2, 0.3],
            "all-MiniLM-L6-v2",
        )
        .unwrap();

        let got = get_memory_by_id(&conn, id).unwrap();
        assert_eq!(got.content, "condensed form");
        assert_eq!(
            got.content_raw.as_deref(),
            Some("original verbose raw content")
        );
        assert_eq!(got.condenser_version.as_deref(), Some("gemma3:abcd"));
        assert_eq!(got.embedding_model.as_deref(), Some("all-MiniLM-L6-v2"));
        assert_eq!(got.embedding.as_ref().map(|v| v.len()), Some(3));
    }
}
