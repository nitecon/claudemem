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
/// Implementation: first try an exact-match lookup (hot path for callers that
/// already have a full UUID, and cheap enough to be worth the extra roundtrip).
/// On miss, fall back to `WHERE id LIKE prefix || '%'` bounded to a small
/// candidate window so accidental 4-char prefixes don't spill thousands of
/// rows into memory.
pub fn resolve_id_prefix(conn: &Connection, prefix: &str) -> Result<ResolvedId, MemoryError> {
    if prefix.len() < MIN_ID_PREFIX_LEN {
        return Err(MemoryError::Config(format!(
            "ID prefix must be at least {MIN_ID_PREFIX_LEN} characters (got {})",
            prefix.len()
        )));
    }

    // Fast path: the caller might already have the full UUID.
    if let Ok(m) = get_memory_by_id(conn, prefix) {
        return Ok(ResolvedId::Exact(m.id));
    }

    // Prefix match. Use a bounded LIMIT so a very short accidental prefix
    // cannot blow the heap; `MAX_PREFIX_CANDIDATES` + 1 so we can detect
    // "more candidates exist than we're willing to list" and keep the
    // disambiguation output short.
    const MAX_PREFIX_CANDIDATES: usize = 16;
    let like_pattern = format!("{prefix}%");
    let mut stmt = conn.prepare(
        "SELECT id, content, tags, project, agent, source_file,
         created_at, updated_at, access_count, embedding, memory_type
         FROM memories WHERE id LIKE ?1 LIMIT ?2",
    )?;
    let rows = stmt.query_map(
        params![like_pattern, (MAX_PREFIX_CANDIDATES + 1) as i64],
        |row| {
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
            })
        },
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
         created_at, updated_at, access_count, embedding, memory_type)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
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
        ],
    )?;
    Ok(())
}

pub fn get_memory_by_id(conn: &Connection, id: &str) -> Result<Memory, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, content, tags, project, agent, source_file,
         created_at, updated_at, access_count, embedding, memory_type
         FROM memories WHERE id = ?1",
    )?;

    stmt.query_row(params![id], |row| {
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
        })
    })
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
    let mut sql = String::from(
        "SELECT id, content, tags, project, agent, source_file,
         created_at, updated_at, access_count, embedding, memory_type
         FROM memories WHERE 1=1",
    );
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
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
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
        })
    })?;

    let mut memories = Vec::new();
    for row in rows {
        memories.push(row?);
    }
    Ok(memories)
}

pub fn get_all_embeddings(conn: &Connection) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
    let mut stmt =
        conn.prepare("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL")?;
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
    let mut stmt = match project {
        Some(_) => conn.prepare(
            "SELECT id, content, tags, project, agent, source_file,
             created_at, updated_at, access_count, embedding, memory_type
             FROM memories WHERE project = ?1",
        )?,
        None => conn.prepare(
            "SELECT id, content, tags, project, agent, source_file,
             created_at, updated_at, access_count, embedding, memory_type
             FROM memories WHERE project IS NULL",
        )?,
    };
    let map_row = |row: &rusqlite::Row| -> rusqlite::Result<Memory> {
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
        })
    };
    let rows: Vec<Memory> = match project {
        Some(p) => stmt
            .query_map(params![p], map_row)?
            .collect::<rusqlite::Result<_>>()?,
        None => stmt
            .query_map([], map_row)?
            .collect::<rusqlite::Result<_>>()?,
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
        insert_memory(conn, &copy)?;
        new_ids.push(copy.id);
    }
    Ok(new_ids)
}

/// List distinct project idents with the number of memories tagged under each.
/// A NULL project is reported as `None`. Ordered by count DESC.
pub fn list_projects(conn: &Connection) -> Result<Vec<(Option<String>, i64)>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT project, COUNT(*) as cnt FROM memories GROUP BY project ORDER BY cnt DESC",
    )?;
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

    let mut stmt = conn.prepare(
        "SELECT id, content, tags, project, agent, source_file,
         created_at, updated_at, access_count, embedding, memory_type
         FROM memories WHERE updated_at < ?1 AND access_count <= ?2",
    )?;

    let rows = stmt.query_map(params![cutoff_str, min_access_count], |row| {
        let tags_str: Option<String> = row.get(2)?;
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
            embedding: None,
            memory_type: row.get(10)?,
        })
    })?;

    let mut to_prune = Vec::new();
    for row in rows {
        to_prune.push(row?);
    }

    if !dry_run {
        conn.execute(
            "DELETE FROM memories WHERE updated_at < ?1 AND access_count <= ?2",
            params![cutoff_str, min_access_count],
        )?;
    }

    Ok(to_prune)
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
}
