//! Project-level cross-memory review pass (v1.6.0).
//!
//! This is the **primary** cross-memory consolidation path. It runs before
//! Stage A (cosine dedup) and Stage B (per-memory condense) on every dream
//! invocation. Unlike [`crate::dream::dedup`] — which compares memories
//! pairwise by vector similarity and misses paraphrased duplicates — this
//! pass sends every memory in a project to the model as a single message
//! and receives structured `keep` / `drop` / `merge` / `supersede` /
//! `extract` decisions back. It can catch semantically-identical memories
//! that share no vocabulary (e.g. three different wordings of the same
//! release milestone).
//!
//! # Ordering inside a dream pass
//!
//! ```text
//! Stage 0: project_review  (this file)   — LLM sees the whole project
//! Stage A: dedup                         — cosine fallback on survivors
//! Stage B: condense                      — per-memory polish on survivors
//! ```
//!
//! `dedup.rs` stays in place as a secondary signal. The user has confirmed
//! cross-memory consolidation is now the primary path, but the cosine
//! pass is cheap and catches byte-identical inserts that slipped through
//! without needing a model round-trip.
//!
//! # Decisions
//!
//! The model returns, for each memory id, one of:
//!
//! * `keep` — leave untouched; it will flow into Stage A/B.
//! * `drop` — delete the memory (e.g. reconstructable from `git log`).
//! * `merge_into:<id>` — fold this memory's facts into another memory that
//!   the model also saw. The "into" id must also appear in the response
//!   with a `supersede_by` clause or a `keep`; dangling `merge_into`
//!   pointers are rejected.
//! * `supersede_by:{content, tags?}` — replace this memory's content with
//!   a new body. Used when a cluster of memories shares a common milestone
//!   and the model wants to produce one canonical entry in its place.
//! * `extract:{content, tags?}` — drop this memory but retain an
//!   architectural note that was buried inside it. Input to the same
//!   sub-memory in which the garbage framing was; the extracted body
//!   becomes a NEW memory row, the original is deleted.
//!
//! # Fallback clustering
//!
//! When a project has too many memories (`MAX_PROJECT_MEMORIES`) or the
//! serialized input is too large (`MAX_PROJECT_CHARS`), the pass splits
//! the memory set into clusters by `(±14-day updated_at window, tag
//! overlap ≥ 2)` and invokes the model once per cluster. No project in
//! the wild currently crosses this cutoff; the fallback is here so a
//! future growth spurt doesn't silently break the pass.

use std::collections::{HashMap, HashSet};

use agent_memory::db::models::Memory;
use agent_memory::db::queries as q;
use agent_memory::embedding::embed_text;
use agent_memory::error::MemoryError;
use agent_memory::render;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::dream::prompt::condenser_version_stamp;
use crate::inference::{Inference, InferenceError};

pub mod prompt;

/// Hard cap on memory count per single-message invocation. Above this the
/// pass splits the project into clusters. Measured per-project; 200 is
/// comfortably above every project in the current store and well below
/// any reasonable model context window when combined with
/// [`MAX_PROJECT_CHARS`].
pub const MAX_PROJECT_MEMORIES: usize = 200;

/// Hard cap on serialized memory-content characters per single-message
/// invocation. ~40k tokens × 4 chars/token = 160k chars — conservative for
/// modern long-context models. Tuned together with
/// [`MAX_PROJECT_MEMORIES`]: either threshold being crossed triggers the
/// clustering fallback.
pub const MAX_PROJECT_CHARS: usize = 160_000;

/// Window (±N days) used by the clustering fallback to group memories by
/// temporal proximity. Two memories land in the same cluster when they
/// share at least [`MIN_TAG_OVERLAP`] tags AND their `updated_at` values
/// fall within this window. 14 days is wide enough to pull together the
/// "parent summary + per-file subdocs" pattern observed on the traderx
/// refactor and narrow enough to keep temporally-unrelated work separate.
pub const CLUSTER_WINDOW_DAYS: i64 = 14;

/// Minimum tag overlap required to cluster two memories together. Two
/// shared tags is the empirical floor that kept milestone entries
/// together without fusing unrelated per-tag clouds.
pub const MIN_TAG_OVERLAP: usize = 2;

/// Max output tokens requested from the model for the project review
/// response. Each memory needs ~1-3 lines of JSON; allowing 32 tokens per
/// memory at the [`MAX_PROJECT_MEMORIES`] cap leaves headroom for
/// commentary-free replies while still bounding runaway generation.
pub const MAX_OUTPUT_TOKENS: u32 = 8192;

/// Top-level errors surfaced by the project review pass.
#[derive(Debug, Error)]
pub enum ProjectReviewError {
    #[error("db error: {0}")]
    Db(#[from] MemoryError),

    #[error("inference backend failed: {0}")]
    Inference(#[from] InferenceError),

    #[error("failed to parse model response: {0}")]
    Parse(String),

    #[error("embedding failed: {0}")]
    Embed(String),

    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

/// Per-memory decision emitted by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Leave the memory in place; it flows into Stage A / B unchanged.
    Keep,
    /// Delete the memory outright (reconstructable from git, pure noise,
    /// etc.). No replacement is created.
    Drop,
    /// Fold into another memory from the same review batch. The target
    /// id must also appear in the decision map with a non-`Drop` verdict;
    /// dangling pointers get downgraded to `Keep` and a warning logged.
    MergeInto { target_id: String },
    /// Replace this memory's body with a new one. The new body flows
    /// into Stage B on the same run so tag normalization and headline
    /// discipline stay centralized in [`crate::dream::condense`].
    SupersedeBy {
        content: String,
        tags: Option<Vec<String>>,
    },
    /// Extract a durable insight into a NEW memory and drop the original.
    /// Used when the original memory mixes reconstructable noise (e.g.
    /// "vX.Y shipped") with a genuine architectural note buried inside.
    Extract {
        content: String,
        tags: Option<Vec<String>>,
    },
}

/// JSON shape the model is asked to emit. Kept separate from [`Decision`]
/// so the parse stage can distinguish "malformed verdict" from "unknown
/// decision kind" and surface actionable errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum RawDecision {
    Keep,
    Drop,
    MergeInto {
        target_id: String,
    },
    SupersedeBy {
        content: String,
        #[serde(default)]
        tags: Option<Vec<String>>,
    },
    Extract {
        content: String,
        #[serde(default)]
        tags: Option<Vec<String>>,
    },
}

/// Envelope the model returns. A map keyed by memory id keeps the
/// response compact and lets the parser detect dangling ids without
/// walking a flat list twice.
#[derive(Debug, Deserialize)]
struct RawResponse {
    decisions: HashMap<String, RawDecision>,
}

/// Running tallies across a project review pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProjectReviewStats {
    pub kept: usize,
    pub dropped: usize,
    pub merged: usize,
    pub superseded: usize,
    pub extracted: usize,
    pub failed: usize,
}

/// Outcome of a project review pass — both the stats and the survivor
/// list the orchestrator should feed into Stage A + B.
pub struct ProjectReviewOutcome {
    pub stats: ProjectReviewStats,
    pub survivors: Vec<Memory>,
}

/// Run the project review pass for a single project's memory set.
///
/// `apply` toggles between Apply (persist decisions, return rewritten
/// survivors for Stage B) and Dry (emit intent lines, persist nothing,
/// return the original survivors so Stage B dry-runs them verbatim).
///
/// The `embedding_cache_dir` is needed because `supersede_by` and
/// `extract` both create new content whose embedding must match the rest
/// of the store for Stage A cosine similarity to remain meaningful.
pub fn run_project(
    conn: &mut Connection,
    inference: &dyn Inference,
    project: Option<&str>,
    candidates: Vec<Memory>,
    model_name: &str,
    embedding_cache_dir: &std::path::Path,
    apply: bool,
) -> Result<ProjectReviewOutcome, ProjectReviewError> {
    let mut stats = ProjectReviewStats::default();

    if candidates.is_empty() {
        return Ok(ProjectReviewOutcome {
            stats,
            survivors: Vec::new(),
        });
    }

    // Decide whether the project fits in one shot or needs clustering.
    let batches = partition_for_review(candidates);

    let mut survivors: Vec<Memory> = Vec::new();

    for batch in batches {
        // Model round-trip per batch. A batch is either "whole project"
        // (normal path) or one cluster (fallback path).
        //
        // On inference / parse failure we do NOT inflate `stats.failed`
        // here — the downstream Stages A + B will run on the passthrough
        // survivors and tally their own failures. Counting the failure
        // at both layers would double-count "this memory couldn't be
        // processed" into the final summary.
        let decisions = match invoke_review(inference, project, &batch) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(project = ?project, error = %e,
                    "project review inference failed; treating batch as all-keep");
                // Keep everything in this batch so Stage B still runs.
                survivors.extend(batch.into_iter());
                continue;
            }
        };

        apply_decisions(
            conn,
            project,
            &batch,
            decisions,
            model_name,
            embedding_cache_dir,
            apply,
            &mut stats,
            &mut survivors,
        )?;
    }

    Ok(ProjectReviewOutcome { stats, survivors })
}

/// Split `candidates` into model-sized batches.
///
/// Fast path: everything fits under both caps → one batch, original order.
/// Fallback: cluster by `(±14-day window, tag_overlap ≥ MIN_TAG_OVERLAP)`
/// and fall back to single-memory batches for anything that didn't
/// cluster (a lone memory still gets reviewed, it just can't be merged).
pub fn partition_for_review(candidates: Vec<Memory>) -> Vec<Vec<Memory>> {
    let total_chars: usize = candidates.iter().map(|m| m.content.len()).sum();
    if candidates.len() <= MAX_PROJECT_MEMORIES && total_chars <= MAX_PROJECT_CHARS {
        return vec![candidates];
    }

    cluster_memories(candidates)
}

/// Cluster memories for the fallback path.
///
/// Algorithm: greedy single-link clustering on the
/// `(date window, tag overlap)` predicate. Each memory walks every
/// existing cluster; it joins the first cluster whose *representative*
/// (the cluster's first-inserted memory) satisfies the predicate.
/// Deliberately simple — the fallback is a safety net, not an
/// optimization target.
fn cluster_memories(candidates: Vec<Memory>) -> Vec<Vec<Memory>> {
    let mut clusters: Vec<Vec<Memory>> = Vec::new();

    for mem in candidates {
        let placed = clusters.iter_mut().find(|cluster| {
            let rep = &cluster[0];
            memories_cluster_together(rep, &mem)
        });

        match placed {
            Some(c) => c.push(mem),
            None => clusters.push(vec![mem]),
        }
    }

    clusters
}

/// Predicate for the clustering fallback.
///
/// Two memories cluster together when their `updated_at` values are
/// within [`CLUSTER_WINDOW_DAYS`] days AND they share at least
/// [`MIN_TAG_OVERLAP`] tags.
fn memories_cluster_together(a: &Memory, b: &Memory) -> bool {
    if !within_window(&a.updated_at, &b.updated_at, CLUSTER_WINDOW_DAYS) {
        return false;
    }
    tag_overlap(a.tags.as_deref(), b.tags.as_deref()) >= MIN_TAG_OVERLAP
}

/// True when two RFC3339 timestamps are within `days` days of each other.
/// Malformed timestamps (shouldn't happen — we stamp with chrono) fall
/// through as "not in window" so we never silently collapse unrelated
/// memories on bad data.
fn within_window(a: &str, b: &str, days: i64) -> bool {
    let Ok(ta) = chrono::DateTime::parse_from_rfc3339(a) else {
        return false;
    };
    let Ok(tb) = chrono::DateTime::parse_from_rfc3339(b) else {
        return false;
    };
    (ta - tb).num_days().abs() <= days
}

/// Count of shared tags between two optional tag lists.
fn tag_overlap(a: Option<&[String]>, b: Option<&[String]>) -> usize {
    let (Some(a), Some(b)) = (a, b) else {
        return 0;
    };
    let set_a: HashSet<&String> = a.iter().collect();
    b.iter().filter(|t| set_a.contains(t)).count()
}

/// Invoke the inference backend with a batch of memories and parse its
/// response into a decision map.
pub fn invoke_review(
    inference: &dyn Inference,
    project: Option<&str>,
    batch: &[Memory],
) -> Result<HashMap<String, Decision>, ProjectReviewError> {
    let prompt_text = prompt::build_project_review_prompt(project, batch);
    let raw = inference.generate(&prompt_text, MAX_OUTPUT_TOKENS)?;
    parse_response(&raw, batch)
}

/// Parse a raw JSON response from the model, validate every referenced
/// id, and return a map of id → decision.
///
/// Validation rules:
///   * Every id in the response must match a memory in `batch`. Unknown
///     ids are dropped with a tracing::warn so the caller sees the
///     miscue but the pass still makes progress.
///   * `MergeInto { target_id }` with a target that doesn't appear in
///     `batch` (or itself maps to `Drop`) is downgraded to `Keep` with a
///     warning — we refuse to orphan data.
///   * Memories present in `batch` but missing from the response default
///     to `Keep` (defensive: the model mis-classifies absent rows as
///     garbage more often than it drops them on purpose).
pub fn parse_response(
    raw: &str,
    batch: &[Memory],
) -> Result<HashMap<String, Decision>, ProjectReviewError> {
    // Strip optional markdown fences — models frequently add ```json
    // even when the prompt forbids them. Defense-in-depth, same as the
    // sibling condense parser.
    let body = strip_code_fence(raw.trim());
    if body.is_empty() {
        return Err(ProjectReviewError::Parse(
            "empty response from model".to_string(),
        ));
    }

    let parsed: RawResponse = serde_json::from_str(body).map_err(|e| {
        ProjectReviewError::Parse(format!(
            "json decode failed: {e}; first 120 chars of body: {:.120}",
            body
        ))
    })?;

    // Fast lookup by id for validation + defaulting-missing to Keep.
    let known_ids: HashSet<&String> = batch.iter().map(|m| &m.id).collect();
    let mut out: HashMap<String, Decision> = HashMap::new();

    for (id, raw_d) in parsed.decisions {
        if !known_ids.contains(&id) {
            tracing::warn!(id = %id, "project review response referenced unknown memory id; skipping");
            continue;
        }
        out.insert(id, raw_d.into());
    }

    // Validate merge targets — dangling or into-a-dropped-memory pointers
    // get downgraded to Keep so facts aren't silently discarded.
    let mut downgrades: Vec<(String, String)> = Vec::new();
    for (id, d) in &out {
        if let Decision::MergeInto { target_id } = d {
            let target_known = known_ids.contains(target_id);
            let target_alive = matches!(
                out.get(target_id),
                Some(Decision::Keep) | Some(Decision::SupersedeBy { .. })
            );
            if !target_known || !target_alive {
                downgrades.push((id.clone(), target_id.clone()));
            }
        }
    }
    for (id, target) in downgrades {
        tracing::warn!(id = %id, target = %target,
            "merge_into target missing or not alive; downgrading to keep");
        out.insert(id, Decision::Keep);
    }

    // Default missing ids → Keep.
    for mem in batch {
        out.entry(mem.id.clone()).or_insert(Decision::Keep);
    }

    Ok(out)
}

/// Strip common markdown fences from a model response. Mirrors the sibling
/// helper in [`crate::dream::condense`] — the prompt forbids fences but
/// models add them anyway.
fn strip_code_fence(s: &str) -> &str {
    let trimmed = s.trim();
    if let Some(body) = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
    {
        body.trim_end_matches("```").trim()
    } else {
        trimmed
    }
}

impl From<RawDecision> for Decision {
    fn from(r: RawDecision) -> Self {
        match r {
            RawDecision::Keep => Decision::Keep,
            RawDecision::Drop => Decision::Drop,
            RawDecision::MergeInto { target_id } => Decision::MergeInto { target_id },
            RawDecision::SupersedeBy { content, tags } => Decision::SupersedeBy { content, tags },
            RawDecision::Extract { content, tags } => Decision::Extract { content, tags },
        }
    }
}

/// Persist decisions for a batch and push the survivors into `survivors`.
///
/// Runs in two phases so the DB write lock is held only for the brief
/// window needed to apply the writes:
///
/// 1. **Lock-free pre-materialization.** For every `SupersedeBy` /
///    `Extract` decision we build the replacement [`Memory`] (including
///    its embedding via fastembed) BEFORE touching sqlite. Previously
///    this ran inside the batch's `BEGIN IMMEDIATE` tx and held a
///    RESERVED write lock for the full embedding duration × every
///    materialization in the batch. On a dense DB that stalled
///    concurrent `memory store`/`update`/`forget` for minutes per batch.
///
/// 2. **Short tx for writes.** Once all replacements are materialized
///    (or their failures recorded), we open a single `BEGIN IMMEDIATE`
///    and apply only the fast DB statements. A delete+insert failure
///    inside a `SupersedeBy` / `Extract` still rolls the whole batch
///    back rather than leaving orphaned state.
#[allow(clippy::too_many_arguments)]
fn apply_decisions(
    conn: &mut Connection,
    project: Option<&str>,
    batch: &[Memory],
    decisions: HashMap<String, Decision>,
    model_name: &str,
    embedding_cache_dir: &std::path::Path,
    apply: bool,
    stats: &mut ProjectReviewStats,
    survivors: &mut Vec<Memory>,
) -> Result<(), ProjectReviewError> {
    let project_label = project.unwrap_or("(null)");

    // Phase 1 — pre-materialize replacement memories for SupersedeBy /
    // Extract decisions with NO sqlite lock held.
    let mut materialized: HashMap<String, Memory> = HashMap::new();
    let mut materialize_failed: HashSet<String> = HashSet::new();
    for mem in batch {
        let (content, tags) = match decisions.get(&mem.id) {
            Some(Decision::SupersedeBy { content, tags })
            | Some(Decision::Extract { content, tags }) => (content.as_str(), tags.as_deref()),
            _ => continue,
        };
        match materialize_new_memory(mem, content, tags, model_name, embedding_cache_dir) {
            Ok(new_mem) => {
                materialized.insert(mem.id.clone(), new_mem);
            }
            Err(e) => {
                stats.failed += 1;
                tracing::warn!(id = %mem.id, error = %e,
                    "review materialize failed; keeping source memory alive");
                materialize_failed.insert(mem.id.clone());
            }
        }
    }

    // Phase 2 — apply writes inside a short `BEGIN IMMEDIATE` tx.
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

    for mem in batch {
        let decision = decisions
            .get(&mem.id)
            .cloned()
            .unwrap_or(Decision::Keep);

        match decision {
            Decision::Keep => {
                stats.kept += 1;
                survivors.push(mem.clone());
                println!(
                    "{}",
                    render::render_action_result(
                        "review_keep",
                        &[
                            ("id", render::short_id(&mem.id).to_string()),
                            ("project", project_label.to_string()),
                        ]
                    )
                );
            }
            Decision::Drop => {
                stats.dropped += 1;
                if apply {
                    if let Err(e) = q::delete_memory(&tx, &mem.id) {
                        stats.failed += 1;
                        tracing::warn!(id = %mem.id, error = %e, "review drop failed");
                        continue;
                    }
                }
                println!(
                    "{}",
                    render::render_action_result(
                        if apply { "review_drop" } else { "review_would_drop" },
                        &[
                            ("id", render::short_id(&mem.id).to_string()),
                            ("project", project_label.to_string()),
                        ]
                    )
                );
            }
            Decision::MergeInto { target_id } => {
                stats.merged += 1;
                if apply {
                    if let Err(e) = q::mark_superseded(&tx, &mem.id, &target_id) {
                        stats.failed += 1;
                        tracing::warn!(id = %mem.id, target = %target_id, error = %e,
                            "review merge failed");
                        continue;
                    }
                }
                println!(
                    "{}",
                    render::render_action_result(
                        if apply { "review_merge" } else { "review_would_merge" },
                        &[
                            ("id", render::short_id(&mem.id).to_string()),
                            ("target", render::short_id(&target_id).to_string()),
                            ("project", project_label.to_string()),
                        ]
                    )
                );
            }
            Decision::SupersedeBy { .. } | Decision::Extract { .. } => {
                // Materialization ran in Phase 1; if it failed we kept
                // the source alive and already charged `stats.failed`.
                if materialize_failed.contains(&mem.id) {
                    survivors.push(mem.clone());
                    continue;
                }
                let new_mem = materialized
                    .remove(&mem.id)
                    .expect("pre-materialized in phase 1");
                let is_supersede = matches!(decision, Decision::SupersedeBy { .. });
                if is_supersede {
                    stats.superseded += 1;
                } else {
                    stats.extracted += 1;
                }
                let label_action = if is_supersede { "supersede" } else { "extract" };

                if apply {
                    // Delete the old row and insert the new one as a
                    // distinct memory so provenance stays audit-friendly.
                    if let Err(e) = q::delete_memory(&tx, &mem.id) {
                        stats.failed += 1;
                        tracing::warn!(id = %mem.id, error = %e,
                            "review {label_action} delete failed");
                        survivors.push(mem.clone());
                        continue;
                    }
                    if let Err(e) = q::insert_memory(&tx, &new_mem) {
                        stats.failed += 1;
                        tracing::warn!(id = %new_mem.id, error = %e,
                            "review {label_action} insert failed");
                        let _ = tx.rollback();
                        return Ok(());
                    }
                }
                survivors.push(new_mem.clone());
                let tag = if apply {
                    if is_supersede {
                        "review_supersede"
                    } else {
                        "review_extract"
                    }
                } else if is_supersede {
                    "review_would_supersede"
                } else {
                    "review_would_extract"
                };
                println!(
                    "{}",
                    render::render_action_result(
                        tag,
                        &[
                            ("old_id", render::short_id(&mem.id).to_string()),
                            ("new_id", render::short_id(&new_mem.id).to_string()),
                            ("project", project_label.to_string()),
                        ]
                    )
                );
            }
        }
    }

    if apply {
        tx.commit()?;
    } else {
        tx.rollback()?;
    }
    Ok(())
}

/// Build a `Memory` row for a `supersede_by` / `extract` decision.
///
/// Inherits the source memory's `project`, `memory_type`, and `agent`
/// columns so the new row lands in the same scope. Tags default to the
/// source's tags when the model didn't return a replacement list — that
/// keeps the model honest about tag changes (it has to emit them
/// explicitly) while still preserving a useful default.
///
/// Embeddings are computed up front rather than deferred to Stage B so
/// the new memory can participate in the same pass's Stage A dedup if
/// desired. Without a fresh embedding the cosine comparison would skip
/// the row entirely.
fn materialize_new_memory(
    source: &Memory,
    content: &str,
    tags_override: Option<&[String]>,
    model_name: &str,
    embedding_cache_dir: &std::path::Path,
) -> Result<Memory, ProjectReviewError> {
    let tags = tags_override
        .map(|t| t.to_vec())
        .or_else(|| source.tags.clone());
    let mut m = Memory::new(
        content.to_string(),
        tags,
        source.project.clone(),
        source.agent.clone(),
        source.source_file.clone(),
        source.memory_type.clone(),
    );

    let emb = embed_text(content, embedding_cache_dir)
        .map_err(|e| ProjectReviewError::Embed(format!("{e}")))?;
    m.embedding = Some(emb);
    m.embedding_model = Some(crate::dream::EMBEDDING_MODEL_NAME.to_string());
    // Stamp the condenser_version so Stage B's incremental filter can
    // recognise this row as "already reviewed this pass" if it loops.
    m.condenser_version = Some(condenser_version_stamp(model_name));
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FixedInference;

    fn mk_memory(id: &str, content: &str, tags: &[&str], updated_at: &str) -> Memory {
        Memory {
            id: id.to_string(),
            content: content.to_string(),
            tags: Some(tags.iter().map(|s| s.to_string()).collect()),
            project: Some("proj".to_string()),
            agent: None,
            source_file: None,
            created_at: updated_at.to_string(),
            updated_at: updated_at.to_string(),
            access_count: 0,
            embedding: None,
            memory_type: Some("user".to_string()),
            content_raw: None,
            superseded_by: None,
            condenser_version: None,
            embedding_model: Some("all-MiniLM-L6-v2".to_string()),
        }
    }

    #[test]
    fn parse_basic_keep_and_drop() {
        let batch = vec![
            mk_memory("aaa", "first", &["t1"], "2026-04-20T00:00:00Z"),
            mk_memory("bbb", "second", &["t2"], "2026-04-21T00:00:00Z"),
        ];
        let raw = r#"{
            "decisions": {
                "aaa": {"action": "keep"},
                "bbb": {"action": "drop"}
            }
        }"#;
        let out = parse_response(raw, &batch).unwrap();
        assert_eq!(out.get("aaa"), Some(&Decision::Keep));
        assert_eq!(out.get("bbb"), Some(&Decision::Drop));
    }

    #[test]
    fn parse_merge_into_with_valid_target_is_preserved() {
        let batch = vec![
            mk_memory("aaa", "first", &["t1"], "2026-04-20T00:00:00Z"),
            mk_memory("bbb", "second", &["t2"], "2026-04-21T00:00:00Z"),
        ];
        let raw = r#"{
            "decisions": {
                "aaa": {"action": "merge_into", "target_id": "bbb"},
                "bbb": {"action": "keep"}
            }
        }"#;
        let out = parse_response(raw, &batch).unwrap();
        match out.get("aaa") {
            Some(Decision::MergeInto { target_id }) => assert_eq!(target_id, "bbb"),
            other => panic!("expected MergeInto, got {other:?}"),
        }
    }

    #[test]
    fn parse_merge_into_dangling_target_downgrades_to_keep() {
        // Target id "zzz" isn't in the batch → merge must downgrade.
        let batch = vec![
            mk_memory("aaa", "first", &["t1"], "2026-04-20T00:00:00Z"),
        ];
        let raw = r#"{
            "decisions": {
                "aaa": {"action": "merge_into", "target_id": "zzz"}
            }
        }"#;
        let out = parse_response(raw, &batch).unwrap();
        assert_eq!(out.get("aaa"), Some(&Decision::Keep));
    }

    #[test]
    fn parse_merge_into_dropped_target_downgrades_to_keep() {
        // Target exists but it's going to be Dropped → not alive.
        let batch = vec![
            mk_memory("aaa", "first", &["t1"], "2026-04-20T00:00:00Z"),
            mk_memory("bbb", "second", &["t2"], "2026-04-21T00:00:00Z"),
        ];
        let raw = r#"{
            "decisions": {
                "aaa": {"action": "merge_into", "target_id": "bbb"},
                "bbb": {"action": "drop"}
            }
        }"#;
        let out = parse_response(raw, &batch).unwrap();
        assert_eq!(out.get("aaa"), Some(&Decision::Keep));
        assert_eq!(out.get("bbb"), Some(&Decision::Drop));
    }

    #[test]
    fn parse_missing_id_defaults_to_keep() {
        let batch = vec![
            mk_memory("aaa", "first", &["t1"], "2026-04-20T00:00:00Z"),
            mk_memory("bbb", "second", &["t2"], "2026-04-21T00:00:00Z"),
        ];
        // Response omits bbb entirely.
        let raw = r#"{
            "decisions": {
                "aaa": {"action": "drop"}
            }
        }"#;
        let out = parse_response(raw, &batch).unwrap();
        assert_eq!(out.get("aaa"), Some(&Decision::Drop));
        assert_eq!(out.get("bbb"), Some(&Decision::Keep));
    }

    #[test]
    fn parse_unknown_id_is_dropped_from_response() {
        let batch = vec![mk_memory("aaa", "first", &["t1"], "2026-04-20T00:00:00Z")];
        let raw = r#"{
            "decisions": {
                "aaa": {"action": "keep"},
                "unknown_id_the_model_invented": {"action": "drop"}
            }
        }"#;
        let out = parse_response(raw, &batch).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out.contains_key("aaa"));
    }

    #[test]
    fn parse_empty_response_is_error() {
        let batch = vec![mk_memory("aaa", "x", &[], "2026-04-20T00:00:00Z")];
        let err = parse_response("   ", &batch).unwrap_err();
        match err {
            ProjectReviewError::Parse(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn parse_malformed_json_is_error() {
        let batch = vec![mk_memory("aaa", "x", &[], "2026-04-20T00:00:00Z")];
        let err = parse_response("{not json at all}", &batch).unwrap_err();
        assert!(matches!(err, ProjectReviewError::Parse(_)));
    }

    #[test]
    fn parse_code_fence_wrapped_response_is_unwrapped() {
        let batch = vec![mk_memory("aaa", "x", &[], "2026-04-20T00:00:00Z")];
        let raw = "```json\n{\"decisions\": {\"aaa\": {\"action\": \"keep\"}}}\n```";
        let out = parse_response(raw, &batch).unwrap();
        assert_eq!(out.get("aaa"), Some(&Decision::Keep));
    }

    #[test]
    fn parse_supersede_and_extract_carry_content_and_tags() {
        let batch = vec![
            mk_memory("aaa", "bloated", &["old"], "2026-04-20T00:00:00Z"),
            mk_memory("bbb", "mixed", &["old"], "2026-04-21T00:00:00Z"),
        ];
        let raw = r#"{
            "decisions": {
                "aaa": {
                    "action": "supersede_by",
                    "content": "canonical form",
                    "tags": ["milestone","ndesign"]
                },
                "bbb": {
                    "action": "extract",
                    "content": "durable architectural note",
                    "tags": ["architecture"]
                }
            }
        }"#;
        let out = parse_response(raw, &batch).unwrap();
        match out.get("aaa") {
            Some(Decision::SupersedeBy { content, tags }) => {
                assert_eq!(content, "canonical form");
                assert_eq!(
                    tags.clone().unwrap(),
                    vec!["milestone".to_string(), "ndesign".to_string()]
                );
            }
            other => panic!("expected SupersedeBy, got {other:?}"),
        }
        match out.get("bbb") {
            Some(Decision::Extract { content, tags }) => {
                assert_eq!(content, "durable architectural note");
                assert_eq!(tags.clone().unwrap(), vec!["architecture".to_string()]);
            }
            other => panic!("expected Extract, got {other:?}"),
        }
    }

    #[test]
    fn partition_small_batch_returns_single_batch() {
        let batch = vec![
            mk_memory("aaa", "x", &["t"], "2026-04-20T00:00:00Z"),
            mk_memory("bbb", "y", &["t"], "2026-04-21T00:00:00Z"),
        ];
        let out = partition_for_review(batch);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn partition_large_batch_triggers_clustering() {
        // Force the count threshold by creating MAX_PROJECT_MEMORIES + 1 memories
        // across two disjoint tag clouds → expect ≥ 2 clusters.
        let mut batch = Vec::with_capacity(MAX_PROJECT_MEMORIES + 2);
        for i in 0..((MAX_PROJECT_MEMORIES / 2) + 1) {
            batch.push(mk_memory(
                &format!("a{i:04}"),
                "x",
                &["cloud-a", "shared"],
                "2026-04-20T00:00:00Z",
            ));
        }
        for i in 0..((MAX_PROJECT_MEMORIES / 2) + 1) {
            batch.push(mk_memory(
                &format!("b{i:04}"),
                "y",
                &["cloud-b", "shared"],
                // Far outside the 14-day window so they can't cluster
                // with the first group on the date predicate.
                "2026-01-01T00:00:00Z",
            ));
        }
        assert!(batch.len() > MAX_PROJECT_MEMORIES, "test precondition");
        let out = partition_for_review(batch);
        assert!(out.len() >= 2, "clustering must yield multiple clusters");
    }

    #[test]
    fn cluster_groups_by_tag_overlap_and_window() {
        let a1 = mk_memory("a1", "x", &["milestone", "ndesign"], "2026-04-20T00:00:00Z");
        let a2 = mk_memory("a2", "y", &["milestone", "ndesign"], "2026-04-22T00:00:00Z");
        let b1 = mk_memory("b1", "z", &["unrelated", "other"], "2026-04-21T00:00:00Z");
        let out = cluster_memories(vec![a1, a2, b1]);
        assert_eq!(out.len(), 2);
        // a1/a2 cluster together; b1 alone.
        let sizes: Vec<_> = out.iter().map(|c| c.len()).collect();
        assert!(sizes.contains(&2));
        assert!(sizes.contains(&1));
    }

    #[test]
    fn cluster_respects_date_window() {
        // Shared tags but far-apart dates → must NOT cluster.
        let a = mk_memory("a", "x", &["t1", "t2"], "2026-04-20T00:00:00Z");
        let b = mk_memory("b", "y", &["t1", "t2"], "2025-01-01T00:00:00Z");
        let out = cluster_memories(vec![a, b]);
        assert_eq!(out.len(), 2, "outside window must split clusters");
    }

    #[test]
    fn cluster_requires_min_tag_overlap() {
        // Same window, only one shared tag → must NOT cluster.
        let a = mk_memory("a", "x", &["shared"], "2026-04-20T00:00:00Z");
        let b = mk_memory("b", "y", &["shared"], "2026-04-21T00:00:00Z");
        let out = cluster_memories(vec![a, b]);
        assert_eq!(out.len(), 2, "single shared tag must not cluster");
    }

    #[test]
    fn invoke_review_round_trip_with_fixed_inference() {
        // End-to-end: stub returns a keep-everything JSON payload and we
        // confirm invoke_review produces the matching decision map.
        let batch = vec![
            mk_memory("aaa", "first", &["t"], "2026-04-20T00:00:00Z"),
            mk_memory("bbb", "second", &["t"], "2026-04-21T00:00:00Z"),
        ];
        let canned = r#"{"decisions": {"aaa": {"action":"keep"}, "bbb": {"action":"keep"}}}"#;
        let inf = FixedInference::new(canned);
        let out = invoke_review(&inf, Some("proj"), &batch).unwrap();
        assert_eq!(out.get("aaa"), Some(&Decision::Keep));
        assert_eq!(out.get("bbb"), Some(&Decision::Keep));
    }
}
