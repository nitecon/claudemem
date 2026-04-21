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
    pub match_quality: MatchQuality,
    pub is_current_project: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchQuality {
    High,
    Medium,
    Low,
}

impl MatchQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchQuality::High => "high",
            MatchQuality::Medium => "medium",
            MatchQuality::Low => "low",
        }
    }
}

/// High: top-5 in both BM25 and vector lists.
/// Medium: top-5 in one list, or top-10 in both.
/// Low: anything else (deep tail of one list, absent from the other).
fn classify_quality(bm25: Option<usize>, vector: Option<usize>) -> MatchQuality {
    match (bm25, vector) {
        (Some(b), Some(v)) if b < 5 && v < 5 => MatchQuality::High,
        (Some(b), Some(v)) if b < 10 && v < 10 => MatchQuality::Medium,
        (Some(b), _) if b < 5 => MatchQuality::Medium,
        (_, Some(v)) if v < 5 => MatchQuality::Medium,
        _ => MatchQuality::Low,
    }
}

#[derive(Clone)]
pub struct SearchOptions<'a> {
    pub limit: usize,
    /// Project whose memories receive a score boost. Typically the cwd-derived
    /// project ident; `None` disables the boost entirely (flat ranking).
    pub current_project: Option<&'a str>,
    /// Multiplier applied to current-project scores before re-sorting.
    pub boost_factor: f32,
    /// Hard filter: only return memories whose `project` equals this string.
    pub only_project: Option<&'a str>,
}

impl<'a> SearchOptions<'a> {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            current_project: None,
            boost_factor: 1.0,
            only_project: None,
        }
    }
}

pub fn hybrid_search(
    conn: &Connection,
    query: &str,
    opts: SearchOptions<'_>,
    model_cache_dir: &Path,
) -> Result<Vec<SearchResult>, MemoryError> {
    // Over-fetch from both rankers so the post-fusion project boost has
    // headroom to reshuffle instead of working within a tiny top-K.
    let candidate_limit = opts.limit.saturating_mul(3).max(opts.limit);

    let bm25_results = search_bm25(conn, query, candidate_limit).unwrap_or_default();

    let query_embedding = embedding::embed_text(query, model_cache_dir)?;
    let all_embeddings = queries::get_all_embeddings(conn)?;
    let vector_results = vector_search(&query_embedding, &all_embeddings, candidate_limit);

    let fused = reciprocal_rank_fusion(&bm25_results, &vector_results, candidate_limit);

    let mut results = Vec::new();
    let mut accessed_ids = Vec::new();

    for ranked in fused {
        match queries::get_memory_by_id(conn, &ranked.id) {
            Ok(memory) => {
                let quality = classify_quality(ranked.bm25_rank, ranked.vector_rank);
                let is_current = match opts.current_project {
                    Some(cp) => memory.project.as_deref() == Some(cp),
                    None => false,
                };
                results.push(SearchResult {
                    memory,
                    rank_info: ranked,
                    match_quality: quality,
                    is_current_project: is_current,
                });
            }
            Err(MemoryError::NotFound(_)) => continue,
            Err(e) => return Err(e),
        }
    }

    if opts.boost_factor != 1.0 && opts.current_project.is_some() {
        for r in &mut results {
            if r.is_current_project {
                r.rank_info.score *= opts.boost_factor;
            }
        }
        results.sort_by(|a, b| {
            b.rank_info
                .score
                .partial_cmp(&a.rank_info.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    if let Some(op) = opts.only_project {
        results.retain(|r| r.memory.project.as_deref() == Some(op));
    }

    results.truncate(opts.limit);

    for r in &results {
        accessed_ids.push(r.memory.id.clone());
    }
    queries::increment_access(conn, &accessed_ids)?;

    Ok(results)
}
