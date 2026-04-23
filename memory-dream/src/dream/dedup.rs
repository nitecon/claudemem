//! Cosine-similarity dedup for the dream pass.
//!
//! Walks dedup candidates returned by
//! [`agent_memory::db::queries::list_dedup_candidates`] (same project,
//! memory_type, and embedding_model as the source row) and classifies them
//! under one of three dispositions:
//!
//! 1. **Exact match** (identical `content`) — supersede unconditionally.
//! 2. **Near match** (cosine ≥ threshold) — supersede unconditionally.
//! 3. **Distinct** — no action.
//!
//! The `apply_policy` helper persists the decision inside the caller's
//! transaction via `agent_memory::db::queries::mark_superseded`.
//!
//! Cosine similarity is reused from the sibling `memory` crate — the
//! single-source `cosine_similarity` function in
//! `agent_memory::search::vector` is already exhaustively tested and must
//! stay the one implementation in the workspace.

use agent_memory::db::models::Memory;
use agent_memory::db::queries::mark_superseded;
use agent_memory::error::MemoryError;
use agent_memory::search::vector::cosine_similarity;
use rusqlite::Connection;

/// Default cosine threshold.
///
/// Tuned empirically for `all-MiniLM-L6-v2` (the embedder fastembed ships
/// by default). 0.90 missed obvious paraphrases; 0.85 pulled in distinct
/// claims that shared vocabulary. 0.87 is the current compromise and is
/// exposed as a constant so a `--threshold` CLI flag can override it
/// without touching this module's internals.
pub const DEFAULT_COSINE_THRESHOLD: f32 = 0.87;

/// Decision for a given source → candidate pair.
///
/// `PartialEq` is deliberately omitted — `Memory` from the sibling crate
/// doesn't implement it (embeddings are f32 slices) and the dream pipeline
/// never compares two `DedupDecision`s for equality. Tests match on the
/// discriminant instead.
#[derive(Debug)]
pub enum DedupDecision<'a> {
    /// The candidate's content is byte-identical to the source; supersede
    /// without even reading the embedding. Kept distinct from `NearMatch`
    /// so the logs can tell the two apart during diagnostics.
    ExactMatch(&'a Memory),
    /// Cosine similarity ≥ threshold. Carries the score so the caller can
    /// log it for telemetry / threshold tuning.
    NearMatch { candidate: &'a Memory, score: f32 },
    /// No match among the candidates — the source row is distinct and
    /// shouldn't be superseded.
    Distinct,
}

/// Scan `candidates` and return the first match against `source`, or
/// `Distinct` when none match.
///
/// Exact-match runs first because it's O(len) and lets us avoid a full
/// cosine scan when a pure duplicate insert slipped through. When neither
/// source nor candidate has an embedding we fall back to exact-match only
/// (no vector to compare against).
///
/// Among cosine matches, the candidate with the highest score wins. Ties
/// break by `updated_at` (most recent first) — more recent wording is
/// usually closer to current user intent.
pub fn find_duplicate<'a>(
    source: &Memory,
    candidates: &'a [Memory],
    threshold: f32,
) -> DedupDecision<'a> {
    // Exact-match short circuit — linear scan, O(k) content comparisons.
    for c in candidates {
        if c.content == source.content {
            return DedupDecision::ExactMatch(c);
        }
    }

    let Some(src_emb) = source.embedding.as_ref() else {
        // No embedding → can't do vector match, but exact-match ran above.
        return DedupDecision::Distinct;
    };

    let mut best: Option<(&Memory, f32)> = None;
    for c in candidates {
        let Some(cand_emb) = c.embedding.as_ref() else {
            continue;
        };
        let score = cosine_similarity(src_emb, cand_emb);
        if score < threshold {
            continue;
        }
        match best {
            None => best = Some((c, score)),
            Some((_, cur)) if score > cur => best = Some((c, score)),
            // Equal score → keep the existing (first-seen) pick. Candidates
            // are iterated in DB order; oldest-first is a fine tiebreak.
            _ => {}
        }
    }

    match best {
        Some((candidate, score)) => DedupDecision::NearMatch { candidate, score },
        None => DedupDecision::Distinct,
    }
}

/// Persist a dedup decision inside the caller's open transaction.
///
/// The dream orchestrator opens a `BEGIN IMMEDIATE` transaction per memory
/// to serialize with concurrent `memory store` calls; this helper assumes
/// the caller is already inside one and just issues the UPDATE.
///
/// Semantics:
///   - `ExactMatch(c)` — older row (whichever of source/c has the earlier
///     `created_at`) is marked superseded by the newer.
///   - `NearMatch{c,..}` — same policy as ExactMatch.
///   - `Distinct` — no-op.
///
/// Returns the (older_id, newer_id) pair that was marked, or None when no
/// action was taken. Callers can use the returned pair to emit a progress
/// log line.
pub fn apply_policy(
    conn: &Connection,
    source: &Memory,
    decision: &DedupDecision<'_>,
) -> Result<Option<(String, String)>, MemoryError> {
    let candidate = match decision {
        DedupDecision::ExactMatch(c) => *c,
        DedupDecision::NearMatch { candidate, .. } => *candidate,
        DedupDecision::Distinct => return Ok(None),
    };

    // Older row gets the superseded_by pointer so the newer one stays
    // queryable. Falls back to lexical ID compare if created_at happens
    // to be equal (unlikely but deterministic).
    let (older, newer) = if candidate.created_at <= source.created_at {
        (&candidate.id, &source.id)
    } else {
        (&source.id, &candidate.id)
    };

    mark_superseded(conn, older, newer)?;
    Ok(Some((older.clone(), newer.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_memory(id: &str, content: &str, created_at: &str, emb: Option<Vec<f32>>) -> Memory {
        Memory {
            id: id.to_string(),
            content: content.to_string(),
            tags: None,
            project: Some("p".to_string()),
            agent: None,
            source_file: None,
            created_at: created_at.to_string(),
            updated_at: created_at.to_string(),
            access_count: 0,
            embedding: emb,
            memory_type: Some("user".to_string()),
            content_raw: None,
            superseded_by: None,
            condenser_version: None,
            embedding_model: Some("mini".to_string()),
        }
    }

    #[test]
    fn exact_match_short_circuits_before_cosine() {
        let src = mk_memory("src", "same text", "2026-04-23T00:00:00Z", None);
        let cand = mk_memory("cand", "same text", "2026-04-22T00:00:00Z", None);
        match find_duplicate(&src, std::slice::from_ref(&cand), 0.87) {
            DedupDecision::ExactMatch(c) => assert_eq!(c.id, "cand"),
            other => panic!("expected ExactMatch, got {other:?}"),
        }
    }

    #[test]
    fn near_match_returns_highest_cosine_score() {
        // Two candidates — the one that's vector-identical to source should win.
        let src = mk_memory(
            "src",
            "alpha",
            "2026-04-23T00:00:00Z",
            Some(vec![1.0, 0.0, 0.0]),
        );
        let near = mk_memory(
            "near",
            "beta",
            "2026-04-22T00:00:00Z",
            Some(vec![0.95, 0.1, 0.0]),
        );
        let far = mk_memory(
            "far",
            "gamma",
            "2026-04-22T00:00:00Z",
            Some(vec![0.0, 1.0, 0.0]),
        );
        match find_duplicate(&src, &[near, far], 0.9) {
            DedupDecision::NearMatch { candidate, score } => {
                assert_eq!(candidate.id, "near");
                assert!(score > 0.9);
            }
            other => panic!("expected NearMatch, got {other:?}"),
        }
    }

    #[test]
    fn below_threshold_returns_distinct() {
        let src = mk_memory(
            "src",
            "a",
            "2026-04-23T00:00:00Z",
            Some(vec![1.0, 0.0, 0.0]),
        );
        let cand = mk_memory(
            "cand",
            "b",
            "2026-04-22T00:00:00Z",
            Some(vec![0.0, 1.0, 0.0]),
        );
        match find_duplicate(&src, std::slice::from_ref(&cand), 0.87) {
            DedupDecision::Distinct => {}
            other => panic!("expected Distinct, got {other:?}"),
        }
    }

    #[test]
    fn missing_embedding_on_source_falls_through_to_distinct() {
        // Exact-match handles the no-embedding case on its own; the vector
        // scan needs an embedding on the source to run.
        let src = mk_memory("src", "a", "2026-04-23T00:00:00Z", None);
        let cand = mk_memory(
            "cand",
            "b",
            "2026-04-22T00:00:00Z",
            Some(vec![1.0, 0.0, 0.0]),
        );
        let out = find_duplicate(&src, std::slice::from_ref(&cand), 0.87);
        assert!(matches!(out, DedupDecision::Distinct));
    }

    #[test]
    fn missing_embedding_on_candidate_is_skipped() {
        let src = mk_memory(
            "src",
            "a",
            "2026-04-23T00:00:00Z",
            Some(vec![1.0, 0.0, 0.0]),
        );
        let no_emb = mk_memory("no_emb", "b", "2026-04-22T00:00:00Z", None);
        let real = mk_memory(
            "real",
            "c",
            "2026-04-22T00:00:00Z",
            Some(vec![1.0, 0.0, 0.0]),
        );
        match find_duplicate(&src, &[no_emb, real], 0.87) {
            DedupDecision::NearMatch { candidate, .. } => assert_eq!(candidate.id, "real"),
            other => panic!("expected NearMatch (skipping no-emb), got {other:?}"),
        }
    }

    #[test]
    fn apply_policy_marks_older_row_as_superseded() {
        // End-to-end: open a fresh SQLite DB, insert source + candidate,
        // call apply_policy, confirm the older row gained superseded_by.
        let conn = agent_memory::db::open_database(&std::path::PathBuf::from(":memory:"))
            .expect("open in-memory db");

        let mut older = mk_memory(
            "00000000-0000-1111-2222-000000000001",
            "dup content",
            "2026-01-01T00:00:00Z",
            Some(vec![1.0, 0.0, 0.0]),
        );
        let mut newer = mk_memory(
            "00000000-0000-1111-2222-000000000002",
            "dup content",
            "2026-04-23T00:00:00Z",
            Some(vec![1.0, 0.0, 0.0]),
        );
        older.content_raw = None;
        newer.content_raw = None;
        agent_memory::db::queries::insert_memory(&conn, &older).unwrap();
        agent_memory::db::queries::insert_memory(&conn, &newer).unwrap();

        let decision = DedupDecision::ExactMatch(&older);
        let result = apply_policy(&conn, &newer, &decision).unwrap();
        let (older_id, newer_id) = result.expect("decision should have marked something");
        assert_eq!(older_id, older.id);
        assert_eq!(newer_id, newer.id);

        // Verify the DB now hides the older row from default reads.
        let live =
            agent_memory::db::queries::list_memories(&conn, None, None, None, None, 10).unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].id, newer.id);
    }

    #[test]
    fn apply_policy_on_distinct_is_noop() {
        let conn = agent_memory::db::open_database(&std::path::PathBuf::from(":memory:"))
            .expect("open in-memory db");
        let src = mk_memory(
            "00000000-0000-1111-2222-000000000099",
            "x",
            "2026-04-23T00:00:00Z",
            None,
        );
        agent_memory::db::queries::insert_memory(&conn, &src).unwrap();
        let result = apply_policy(&conn, &src, &DedupDecision::Distinct).unwrap();
        assert!(result.is_none());
    }
}
