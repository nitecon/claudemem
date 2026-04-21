use std::collections::HashMap;

use serde::Serialize;

const RRF_K: f32 = 60.0;

#[derive(Debug, Serialize)]
pub struct RankedResult {
    pub id: String,
    pub score: f32,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
}

pub fn reciprocal_rank_fusion(
    bm25_results: &[(String, f32)],
    vector_results: &[(String, f32)],
    limit: usize,
) -> Vec<RankedResult> {
    let mut scores: HashMap<String, RankedResult> = HashMap::new();

    for (rank, (id, _)) in bm25_results.iter().enumerate() {
        let entry = scores.entry(id.clone()).or_insert_with(|| RankedResult {
            id: id.clone(),
            score: 0.0,
            bm25_rank: None,
            vector_rank: None,
        });
        entry.score += 1.0 / (RRF_K + (rank as f32) + 1.0);
        entry.bm25_rank = Some(rank);
    }

    for (rank, (id, _)) in vector_results.iter().enumerate() {
        let entry = scores.entry(id.clone()).or_insert_with(|| RankedResult {
            id: id.clone(),
            score: 0.0,
            bm25_rank: None,
            vector_rank: None,
        });
        entry.score += 1.0 / (RRF_K + (rank as f32) + 1.0);
        entry.vector_rank = Some(rank);
    }

    let mut results: Vec<_> = scores.into_values().collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    results
}
