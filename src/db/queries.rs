use rusqlite::{params, Connection};

use crate::db::models::{blob_to_embedding, embedding_to_blob, Memory};
use crate::error::MemoryError;

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
        sql.push_str(&format!(
            " AND project = ?{}",
            param_values.len() + 1
        ));
        param_values.push(Box::new(p.to_string()));
    }
    if let Some(a) = agent {
        sql.push_str(&format!(" AND agent = ?{}", param_values.len() + 1));
        param_values.push(Box::new(a.to_string()));
    }
    if let Some(mt) = memory_type {
        sql.push_str(&format!(
            " AND memory_type = ?{}",
            param_values.len() + 1
        ));
        param_values.push(Box::new(mt.to_string()));
    }
    if let Some(tag_list) = tags {
        for tag in tag_list {
            sql.push_str(&format!(
                " AND tags LIKE ?{}",
                param_values.len() + 1
            ));
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

pub fn prune_memories(
    conn: &Connection,
    max_age_days: u64,
    min_access_count: i64,
    dry_run: bool,
) -> Result<Vec<Memory>, MemoryError> {
    let cutoff = chrono::Utc::now()
        - chrono::Duration::days(max_age_days as i64);
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
