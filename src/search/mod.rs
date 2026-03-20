pub mod bm25;
pub mod fusion;
pub mod vector;

use rusqlite::Connection;
use std::path::Path;

use crate::db::models::Memory;
use crate::db::queries;
use crate::embedding;
use crate::error::MemoryError;

use self::bm25::search_bm25;
use self::fusion::{reciprocal_rank_fusion, RankedResult};
use self::vector::vector_search;

pub struct SearchResult {
    pub memory: Memory,
    pub rank_info: RankedResult,
}

pub fn hybrid_search(
    conn: &Connection,
    query: &str,
    limit: usize,
    model_cache_dir: &Path,
) -> Result<Vec<SearchResult>, MemoryError> {
    let expanded_limit = limit * 3;

    // BM25 search via FTS5
    let bm25_results = search_bm25(conn, query, expanded_limit).unwrap_or_default();

    // Vector search
    let query_embedding = embedding::embed_text(query, model_cache_dir)?;
    let all_embeddings = queries::get_all_embeddings(conn)?;
    let vector_results = vector_search(&query_embedding, &all_embeddings, expanded_limit);

    // Fuse results
    let fused = reciprocal_rank_fusion(&bm25_results, &vector_results, limit);

    // Fetch full memories and increment access counts
    let mut results = Vec::new();
    let mut accessed_ids = Vec::new();

    for ranked in fused {
        match queries::get_memory_by_id(conn, &ranked.id) {
            Ok(memory) => {
                accessed_ids.push(ranked.id.clone());
                results.push(SearchResult {
                    memory,
                    rank_info: ranked,
                });
            }
            Err(MemoryError::NotFound(_)) => continue,
            Err(e) => return Err(e),
        }
    }

    queries::increment_access(conn, &accessed_ids)?;

    Ok(results)
}
