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

/// Bump the `access_count` counter for every id in `ids`.
///
/// Intentionally does NOT touch `updated_at` — reads are not edits, and
/// stamping them as such would make every `memory get` / `memory context`
/// resurface a memory at the top of `ORDER BY updated_at DESC` lists (and
/// poison the dream incremental cutoff, which uses `updated_at > last_dream_at`
/// to find new work). Prior to v1.5.0 this function did bump `updated_at`
/// and caused exactly those regressions; keep the SQL minimal and the
/// semantics read-only so neither problem returns.
pub fn increment_access(conn: &Connection, ids: &[String]) -> Result<(), MemoryError> {
    let mut stmt =
        conn.prepare("UPDATE memories SET access_count = access_count + 1 WHERE id = ?1")?;
    for id in ids {
        stmt.execute(params![id])?;
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
/// Issues a single `UPDATE` statement, which SQLite auto-commits as an
/// atomic unit — no explicit transaction wrapper is needed. Callers
/// should NOT hold an outer write-locking tx across slow work (LLM
/// inference, embedding) when using this helper, or concurrent
/// `memory store`/`update`/`forget` invocations will block.
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

/// Atomic content replacement for `memory update <id>` and the agentic dream
/// re-author flow.
///
/// Semantics:
///   - Old `content` moves into `content_raw` (provenance / audit trail).
///   - New `content` replaces the old.
///   - `updated_at` bumps to now.
///   - `superseded_by` clears (if the row had been dedup'd, a manual re-author
///     resurrects it as an active record).
///   - Optional `tags` / `memory_type` updates are applied when provided.
///
/// Returns true when a row was updated. Embedding is NOT touched here — the
/// CLI handler re-embeds the new content separately so query helpers stay
/// free of the fastembed dependency.
#[allow(dead_code)]
pub fn update_content(
    conn: &Connection,
    id: &str,
    new_content: &str,
    new_tags: Option<&[String]>,
    new_memory_type: Option<&str>,
) -> Result<bool, MemoryError> {
    let now = chrono::Utc::now().to_rfc3339();
    let tags_json = new_tags.map(serde_json::to_string).transpose()?;

    // SQL is dynamic on the presence of tags/memory_type so absent flags don't
    // clobber the row's existing values with NULLs. The "move content to
    // content_raw" step uses `COALESCE(content_raw, content)` so re-authoring
    // a previously-condensed memory preserves the first-ever raw form — we
    // never chain condensations through (raw → condensed → newer condensed →
    // lose original) style drift.
    let mut sql = String::from(
        "UPDATE memories SET
             content_raw = COALESCE(content_raw, content),
             content = ?1,
             updated_at = ?2,
             superseded_by = NULL",
    );
    let mut bind: Vec<Box<dyn rusqlite::types::ToSql>> =
        vec![Box::new(new_content.to_string()), Box::new(now.clone())];

    if new_tags.is_some() {
        sql.push_str(&format!(", tags = ?{}", bind.len() + 1));
        bind.push(Box::new(tags_json));
    }
    if let Some(mt) = new_memory_type {
        sql.push_str(&format!(", memory_type = ?{}", bind.len() + 1));
        bind.push(Box::new(mt.to_string()));
    }

    sql.push_str(&format!(" WHERE id = ?{}", bind.len() + 1));
    bind.push(Box::new(id.to_string()));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = bind.iter().map(|b| b.as_ref()).collect();
    let changed = conn.execute(&sql, param_refs.as_slice())?;
    Ok(changed > 0)
}

// -- project_state helpers (Release 2.3) -------------------------------------
//
// Track per-project "last dream pass" timestamps so the incremental filter
// in `list_dream_candidates` can skip rows that haven't changed since the
// previous pass. RFC3339 strings throughout — same format every other
// timestamp column uses in this schema.

/// Return the RFC3339 timestamp of the last successful dream pass for a
/// project, or `None` when the project has never been dreamed.
///
/// NULL semantics: `project` may be `None` for memories with no project tag.
/// Internally stored as the literal string `"__null__"` so the primary key
/// constraint on `project_state.project` stays strict TEXT NOT NULL — we
/// don't need the three-valued logic headaches a nullable PK would bring.
#[allow(dead_code)]
pub fn get_last_dream_at(
    conn: &Connection,
    project: Option<&str>,
) -> Result<Option<String>, MemoryError> {
    let key = project.unwrap_or(NULL_PROJECT_SENTINEL);
    let res = conn.query_row(
        "SELECT last_dream_at FROM project_state WHERE project = ?1",
        params![key],
        |row| row.get::<_, String>(0),
    );
    match res {
        Ok(ts) => Ok(Some(ts)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(MemoryError::Database(e)),
    }
}

/// Upsert a project's last-dream timestamp. Call after successfully processing
/// a project's batch so the next incremental pass can skip untouched rows.
#[allow(dead_code)]
pub fn set_last_dream_at(
    conn: &Connection,
    project: Option<&str>,
    timestamp: &str,
) -> Result<(), MemoryError> {
    let key = project.unwrap_or(NULL_PROJECT_SENTINEL);
    conn.execute(
        "INSERT INTO project_state (project, last_dream_at) VALUES (?1, ?2)
         ON CONFLICT(project) DO UPDATE SET last_dream_at = excluded.last_dream_at",
        params![key, timestamp],
    )?;
    Ok(())
}

/// Sentinel used as the `project_state.project` key for memories whose
/// `project` column is NULL. Visible in the DB for diagnostic purposes but
/// never exposed to the CLI surface — the helpers above translate it back
/// into `None` transparently.
pub const NULL_PROJECT_SENTINEL: &str = "__null__";

/// List the distinct project idents that have at least one live memory. The
/// dream orchestrator walks this list to produce per-project batches. `NULL`
/// projects surface as `None` so the caller can bind the
/// [`NULL_PROJECT_SENTINEL`] translation via `set_last_dream_at`.
#[allow(dead_code)]
pub fn list_distinct_projects_for_dream(
    conn: &Connection,
) -> Result<Vec<Option<String>>, MemoryError> {
    // Dream must see every project, even ones where every memory was
    // superseded on a previous pass — clearing a superseded-by pointer via
    // `memory update` should still cause the next dream run to re-examine
    // the project.
    let mut stmt = conn.prepare("SELECT DISTINCT project FROM memories")?;
    let rows = stmt.query_map([], |row| row.get::<_, Option<String>>(0))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

/// Light-weight snapshot of the dream-relevant columns for a set of
/// memory ids. Returned as a `Vec` of `(id, project, updated_at)` tuples
/// ordered by `id` for deterministic diffing.
///
/// Used by the dream orchestrator to compute per-batch deltas: it snapshots
/// this triple for every candidate before a batch, invokes the agentic
/// model (which may `memory forget` / `memory update` / `memory move`
/// arbitrary rows), then re-queries to classify each candidate as
/// `forgot` (missing from post-snapshot), `moved` (project changed), or
/// `updated` (updated_at advanced within the same project).
///
/// Superseded rows ARE included in the result because the dream pass
/// walks audit rows too; callers doing delta math treat them the same
/// as any other row in the candidate list.
///
/// Empty input returns an empty vec without touching the DB.
#[allow(dead_code)]
pub fn snapshot_dream_rows(
    conn: &Connection,
    ids: &[String],
) -> Result<Vec<(String, Option<String>, String)>, MemoryError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    // Build the `IN (?, ?, ?)` placeholder list dynamically. SQLite has a
    // default 999-parameter ceiling which is comfortably above any real
    // dream batch (capped at 100 upstream), so no chunking is required.
    let placeholders: String = (1..=ids.len())
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT id, project, updated_at FROM memories \
         WHERE id IN ({placeholders}) ORDER BY id ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let bind: Vec<&dyn rusqlite::types::ToSql> = ids
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(bind.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let mut out = Vec::with_capacity(ids.len());
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

/// Incremental candidate selection for an agentic or non-agentic dream pass.
///
/// Returns memories within `project` (NULL when `project.is_none()`) that
/// either:
///   - Have `updated_at > last_dream_at` (new or recently edited rows), OR
///   - Have `condenser_version IS NULL` or a stale value (prompt revision),
///   - AND are not superseded.
///
/// When `last_dream_at` is `None` (project has never been dreamed) the
/// update-time filter is effectively epoch, i.e. every live row matches.
///
/// `current_condenser_version` scopes the "stale" check — rows whose
/// `condenser_version` equals the current stamp are skipped even if they
/// are newer than `last_dream_at`, because re-walking them produces no
/// work. Passing `None` disables the condenser-version guard so the caller
/// gets every live row (`--full` flag semantics).
#[allow(dead_code)]
pub fn list_dream_candidates(
    conn: &Connection,
    project: Option<&str>,
    last_dream_at: Option<&str>,
    current_condenser_version: Option<&str>,
    limit: usize,
) -> Result<Vec<Memory>, MemoryError> {
    let mut sql = format!(
        "SELECT {MEMORY_COLS} FROM memories \
         WHERE superseded_by IS NULL"
    );
    let mut bind: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    match project {
        Some(p) => {
            sql.push_str(&format!(" AND project = ?{}", bind.len() + 1));
            bind.push(Box::new(p.to_string()));
        }
        None => sql.push_str(" AND project IS NULL"),
    }

    // Incremental gate: either the row is newer than the last dream cutoff,
    // or its condenser_version is stale (or unstamped). Both branches fall
    // through when `last_dream_at` is None (never dreamed) or when no
    // current condenser version is provided (--full path).
    match (last_dream_at, current_condenser_version) {
        (Some(last), Some(cur)) => {
            sql.push_str(&format!(
                " AND (updated_at > ?{} OR condenser_version IS NULL OR condenser_version != ?{})",
                bind.len() + 1,
                bind.len() + 2
            ));
            bind.push(Box::new(last.to_string()));
            bind.push(Box::new(cur.to_string()));
        }
        (Some(last), None) => {
            sql.push_str(&format!(" AND updated_at > ?{}", bind.len() + 1));
            bind.push(Box::new(last.to_string()));
        }
        (None, _) => {
            // No cutoff — every live row in this project is a candidate.
        }
    }

    sql.push_str(" ORDER BY updated_at ASC");
    if limit > 0 {
        sql.push_str(&format!(" LIMIT ?{}", bind.len() + 1));
        bind.push(Box::new(limit as i64));
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
    fn update_content_swaps_content_and_archives_raw() {
        let conn = fresh_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id);

        // Before: content = "content aaaaaaaa-...", content_raw = None,
        // superseded_by = None (fresh insert).
        let before = get_memory_by_id(&conn, id).unwrap();
        assert!(before.content_raw.is_none());

        let changed = update_content(&conn, id, "new bullets\n- fact 1\n- fact 2", None, None)
            .expect("update ok");
        assert!(changed);

        let after = get_memory_by_id(&conn, id).unwrap();
        assert_eq!(after.content, "new bullets\n- fact 1\n- fact 2");
        assert_eq!(after.content_raw.as_deref(), Some(before.content.as_str()));
        assert!(after.superseded_by.is_none());
        assert_ne!(after.updated_at, before.updated_at, "updated_at must bump");
    }

    #[test]
    fn update_content_clears_superseded_by() {
        // Re-authoring a previously-superseded row resurrects it; the dedup
        // pointer clears so it rejoins default reads.
        let conn = fresh_db();
        let older = "aaaaaaaa-0000-1111-2222-000000000001";
        let newer = "bbbbbbbb-0000-1111-2222-000000000002";
        insert(&conn, older);
        insert(&conn, newer);
        mark_superseded(&conn, older, newer).unwrap();

        assert!(
            get_memory_by_id(&conn, older)
                .unwrap()
                .superseded_by
                .is_some(),
            "pre-condition: older row should be marked superseded"
        );

        let changed = update_content(&conn, older, "resurrected", None, None).expect("update");
        assert!(changed);
        let after = get_memory_by_id(&conn, older).unwrap();
        assert!(
            after.superseded_by.is_none(),
            "update_content must clear superseded_by"
        );
        assert_eq!(after.content, "resurrected");
    }

    #[test]
    fn update_content_preserves_first_raw_across_multiple_rewrites() {
        // First update archives the original; a second update must NOT
        // overwrite content_raw with the first condensed form (that would
        // lose the verbatim user content).
        let conn = fresh_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id);
        let original = get_memory_by_id(&conn, id).unwrap().content;

        update_content(&conn, id, "first rewrite", None, None).unwrap();
        update_content(&conn, id, "second rewrite", None, None).unwrap();
        let after = get_memory_by_id(&conn, id).unwrap();
        assert_eq!(after.content, "second rewrite");
        assert_eq!(
            after.content_raw.as_deref(),
            Some(original.as_str()),
            "content_raw must retain the very first pre-update body"
        );
    }

    #[test]
    fn update_content_accepts_optional_tags_and_type() {
        let conn = fresh_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id);

        let new_tags = vec!["a".to_string(), "b".to_string()];
        let changed =
            update_content(&conn, id, "x", Some(&new_tags), Some("project")).expect("update");
        assert!(changed);
        let after = get_memory_by_id(&conn, id).unwrap();
        assert_eq!(after.tags.as_deref(), Some(&new_tags[..]));
        assert_eq!(after.memory_type.as_deref(), Some("project"));
    }

    #[test]
    fn update_content_missing_id_returns_false() {
        let conn = fresh_db();
        let changed =
            update_content(&conn, "no-such-id", "body", None, None).expect("no error on miss");
        assert!(!changed);
    }

    #[test]
    fn project_state_round_trip() {
        let conn = fresh_db();
        // Absent row reads as None.
        assert!(get_last_dream_at(&conn, Some("agent-memory"))
            .unwrap()
            .is_none());

        set_last_dream_at(&conn, Some("agent-memory"), "2026-04-23T00:00:00Z").unwrap();
        let ts = get_last_dream_at(&conn, Some("agent-memory"))
            .unwrap()
            .expect("should be present");
        assert_eq!(ts, "2026-04-23T00:00:00Z");

        // Upsert to a newer time.
        set_last_dream_at(&conn, Some("agent-memory"), "2026-04-24T00:00:00Z").unwrap();
        let updated = get_last_dream_at(&conn, Some("agent-memory"))
            .unwrap()
            .expect("should still be present");
        assert_eq!(updated, "2026-04-24T00:00:00Z");
    }

    #[test]
    fn project_state_null_project_is_supported() {
        let conn = fresh_db();
        assert!(get_last_dream_at(&conn, None).unwrap().is_none());
        set_last_dream_at(&conn, None, "2026-04-23T00:00:00Z").unwrap();
        assert_eq!(
            get_last_dream_at(&conn, None).unwrap().as_deref(),
            Some("2026-04-23T00:00:00Z")
        );
        // Null-project state does not leak to a named project key.
        assert!(get_last_dream_at(&conn, Some("other")).unwrap().is_none());
    }

    #[test]
    fn list_dream_candidates_returns_all_live_rows_without_cutoff() {
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-000000000001");
        insert(&conn, "bbbbbbbb-0000-1111-2222-000000000002");
        let rows = list_dream_candidates(&conn, Some("test"), None, None, 0).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn list_dream_candidates_excludes_superseded() {
        let conn = fresh_db();
        let older = "aaaaaaaa-0000-1111-2222-000000000001";
        let newer = "bbbbbbbb-0000-1111-2222-000000000002";
        insert(&conn, older);
        insert(&conn, newer);
        mark_superseded(&conn, older, newer).unwrap();
        let rows = list_dream_candidates(&conn, Some("test"), None, None, 0).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, newer);
    }

    #[test]
    fn list_dream_candidates_filters_by_updated_at() {
        let conn = fresh_db();

        // Seed two rows with different updated_at values. We set them manually
        // so the cutoff comparison is deterministic (insert uses "now").
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                "aaaaaaaa-0000-1111-2222-000000000001",
                "old body",
                "test",
                "user",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                "bbbbbbbb-0000-1111-2222-000000000002",
                "new body",
                "test",
                "user",
                "2026-04-23T00:00:00Z",
                "2026-04-23T00:00:00Z",
            ],
        )
        .unwrap();

        // Cutoff between the two — only the newer row surfaces.
        let rows =
            list_dream_candidates(&conn, Some("test"), Some("2026-02-01T00:00:00Z"), None, 0)
                .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "bbbbbbbb-0000-1111-2222-000000000002");
    }

    #[test]
    fn list_dream_candidates_stale_condenser_version_overrides_cutoff() {
        // Row is older than the cutoff BUT has a stale condenser_version;
        // it must still be returned so a prompt change re-condenses it.
        let conn = fresh_db();
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type, created_at, updated_at,
                                   condenser_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                "aaaaaaaa-0000-1111-2222-000000000001",
                "old body",
                "test",
                "user",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "gemma3:deadbeef",
            ],
        )
        .unwrap();

        // Cutoff is well after the row's updated_at, so the time filter alone
        // would reject it. Current condenser version mismatches → keep it.
        let rows = list_dream_candidates(
            &conn,
            Some("test"),
            Some("2026-04-23T00:00:00Z"),
            Some("gemma3:cafefeed"),
            0,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn list_dream_candidates_current_condenser_skips_fresh_rows() {
        // Row is new-ish AND stamped with the current condenser; treat as
        // already processed (nothing for agentic pass to do).
        let conn = fresh_db();
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type, created_at, updated_at,
                                   condenser_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                "aaaaaaaa-0000-1111-2222-000000000001",
                "old body",
                "test",
                "user",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "gemma3:cafefeed",
            ],
        )
        .unwrap();

        let rows = list_dream_candidates(
            &conn,
            Some("test"),
            Some("2026-04-23T00:00:00Z"),
            Some("gemma3:cafefeed"),
            0,
        )
        .unwrap();
        assert!(
            rows.is_empty(),
            "fresh-enough row with current stamp must be skipped"
        );
    }

    #[test]
    fn list_distinct_projects_for_dream_surfaces_null() {
        let conn = fresh_db();

        let mut a = Memory::new(
            "with project".to_string(),
            None,
            Some("p1".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        a.id = "aaaaaaaa-0000-1111-2222-000000000001".to_string();
        insert_memory(&conn, &a).unwrap();

        let mut b = Memory::new(
            "null project".to_string(),
            None,
            None,
            None,
            None,
            Some("user".to_string()),
        );
        b.id = "bbbbbbbb-0000-1111-2222-000000000002".to_string();
        insert_memory(&conn, &b).unwrap();

        let rows = list_distinct_projects_for_dream(&conn).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.contains(&Some("p1".to_string())));
        assert!(rows.contains(&None));
    }

    #[test]
    fn snapshot_dream_rows_returns_id_project_and_updated_at() {
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-000000000001");
        insert(&conn, "bbbbbbbb-0000-1111-2222-000000000002");
        let ids = vec![
            "aaaaaaaa-0000-1111-2222-000000000001".to_string(),
            "bbbbbbbb-0000-1111-2222-000000000002".to_string(),
        ];
        let snap = snapshot_dream_rows(&conn, &ids).unwrap();
        assert_eq!(snap.len(), 2);
        // `insert` helper plants project="test"; result must echo that.
        for (id, project, updated_at) in &snap {
            assert!(ids.contains(id));
            assert_eq!(project.as_deref(), Some("test"));
            assert!(!updated_at.is_empty());
        }
    }

    #[test]
    fn snapshot_dream_rows_is_empty_for_empty_input() {
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-000000000001");
        let out = snapshot_dream_rows(&conn, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn snapshot_dream_rows_omits_missing_ids_without_error() {
        // A batch may ask about ids that were already deleted by a
        // concurrent writer. The helper must not error — callers treat
        // the missing id as "forgot" in their delta math.
        let conn = fresh_db();
        insert(&conn, "aaaaaaaa-0000-1111-2222-000000000001");
        let ids = vec![
            "aaaaaaaa-0000-1111-2222-000000000001".to_string(),
            "deadbeef-0000-1111-2222-000000000002".to_string(),
        ];
        let snap = snapshot_dream_rows(&conn, &ids).unwrap();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].0, "aaaaaaaa-0000-1111-2222-000000000001");
    }

    #[test]
    fn increment_access_does_not_touch_updated_at() {
        // Reads must not bump `updated_at`. Prior to v1.5.0 this function
        // did, and it polluted `ORDER BY updated_at DESC` lists + dream's
        // incremental cutoff. This test pins the read-only semantics.
        let conn = fresh_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id);
        let before = get_memory_by_id(&conn, id).unwrap();

        // Sleep briefly so any accidental `updated_at = now()` would land
        // on a distinguishable timestamp, making the negative assertion
        // meaningful even on fast machines.
        std::thread::sleep(std::time::Duration::from_millis(10));
        increment_access(&conn, std::slice::from_ref(&id.to_string())).unwrap();

        let after = get_memory_by_id(&conn, id).unwrap();
        assert_eq!(
            after.updated_at, before.updated_at,
            "read path must not bump updated_at"
        );
        assert_eq!(
            after.access_count,
            before.access_count + 1,
            "access_count must still increment"
        );
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
