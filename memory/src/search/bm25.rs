use rusqlite::{params, Connection};

use crate::error::MemoryError;

/// Search memories using SQLite FTS5 with BM25 ranking.
/// The FTS5 virtual table is kept in sync via triggers on the memories table.
pub fn search_bm25(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<(String, f32)>, MemoryError> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    // The JOIN filters `m.superseded_by IS NULL` so lexical hits against a
    // dream-obsoleted memory don't pollute the ranked results. FTS5 indexes
    // every row (superseded or not) via the content-triggered virtual table;
    // the filter lives on the `memories` side where the `superseded_by`
    // column actually lives.
    let mut stmt = conn.prepare(
        "SELECT m.id, bm25(memories_fts) as score
         FROM memories_fts
         JOIN memories m ON m.rowid = memories_fts.rowid
         WHERE memories_fts MATCH ?1 AND m.superseded_by IS NULL
         ORDER BY score
         LIMIT ?2",
    )?;

    let rows = match stmt.query_map(params![trimmed, limit as i64], |row| {
        let id: String = row.get(0)?;
        let score: f64 = row.get(1)?;
        Ok((id, (-score) as f32)) // Negate: FTS5 bm25() returns negative values (closer to 0 = better)
    }) {
        Ok(r) => r,
        Err(_) => return Ok(Vec::new()), // Malformed FTS5 query
    };

    let mut results = Vec::new();
    for row in rows {
        match row {
            Ok(r) => results.push(r),
            Err(_) => return Ok(results), // Stop on error (likely FTS5 parse issue)
        }
    }
    Ok(results)
}
