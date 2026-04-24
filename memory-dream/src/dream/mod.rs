//! Dream orchestrator (v1.6.0).
//!
//! Entry point: [`run`]. Called once per `memory-dream` invocation.
//! Per-project pipeline:
//!
//! 1. **Enumerate projects** with at least one live memory.
//! 2. **Stage 0 — project review** ([`project_review::run_project`]).
//!    Primary cross-memory consolidation path. The model sees every
//!    memory in a project (or every memory in a clustered batch when
//!    the project is too large) and emits per-memory
//!    `keep`/`drop`/`merge_into`/`supersede_by`/`extract` decisions.
//!    Catches paraphrased duplicates that Stage A misses because they
//!    share no vocabulary.
//! 3. **Stage A — cosine dedup** ([`dedup::find_duplicate`] + policy).
//!    Secondary signal. Kept in place to catch byte-identical inserts
//!    that slipped through without a model round-trip. See
//!    [`project_review`] module docs for the rationale.
//! 4. **Stage B — per-memory condense**. For every remaining candidate we
//!    invoke the configured inference backend with the strict three-way
//!    prompt contract ([`condense::run_per_memory`]):
//!      * `skip` → no change.
//!      * `forget` → delete via the DB layer (not a `memory forget` shell).
//!      * otherwise → treat as a rewritten body, persist via
//!        `update_content` so `content_raw` preserves provenance.
//! 5. **Stamp** `project_state.last_dream_at` per project (Apply mode).
//!
//! Progress is emitted as light-XML on stdout — same renderer the rest of
//! the project uses — so a caller can pipe `memory-dream` output into a
//! log collector alongside `memory` command output.

pub mod condense;
pub mod dedup;
pub mod project_review;
pub mod prompt;

use std::path::Path;

use agent_memory::db::models::Memory;
use agent_memory::db::queries as q;
use agent_memory::embedding::embed_text;
use agent_memory::error::MemoryError;
use agent_memory::render;
use rusqlite::Connection;
use thiserror::Error;

use crate::inference::Inference;

/// Top-level errors for the dream orchestrator.
#[derive(Debug, Error)]
pub enum DreamError {
    #[error("db error: {0}")]
    Db(#[from] MemoryError),

    #[error("sqlite transaction error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

/// Run mode for a dream pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamMode {
    /// Walk candidate memories and report intended decisions without
    /// writing. Stage A still classifies duplicates (no mutation); Stage B
    /// invokes the LLM but the parsed decision is discarded.
    Dry,
    /// Commit changes.
    Apply,
}

/// Short name used for the embedding model column. Mirrors the fastembed
/// default in `agent_memory::embedding::embed_text`.
pub const EMBEDDING_MODEL_NAME: &str = "all-MiniLM-L6-v2";

/// Configuration for a single dream pass.
pub struct DreamConfig<'a> {
    /// Which execution mode to run in.
    pub mode: DreamMode,
    /// Cap on the number of memories to process. `0` = no limit.
    pub limit: usize,
    /// Short model name to stamp into `condenser_version`.
    pub model_name: &'a str,
    /// Cache directory for fastembed's MiniLM model files.
    pub embedding_cache_dir: &'a Path,
    /// Cosine threshold for Stage A dedup.
    pub cosine_threshold: f32,
    /// Force a full walk, ignoring both the
    /// `project_state.last_dream_at` time cutoff AND the
    /// `condenser_version` freshness check, so every live memory in every
    /// project flows into Stages 0 / A / B regardless of when it was last
    /// processed.
    ///
    /// Set via `memory-dream --full` (historical name) OR
    /// `memory-dream --refresh` (user-facing alias). `main.rs` folds both
    /// CLI flags into this single switch so every orchestrator site only
    /// needs one code path.
    pub full: bool,
    /// Reserved — per-memory condense doesn't batch, but the CLI flag is
    /// retained so existing automation keeps parsing. Ignored by the
    /// v1.5.0 orchestrator.
    pub batch_size_override: usize,
}

impl<'a> DreamConfig<'a> {
    /// Build a config with sensible defaults. The caller is still expected
    /// to fill in `mode` and `embedding_cache_dir`.
    pub fn new(mode: DreamMode, model_name: &'a str, embedding_cache_dir: &'a Path) -> Self {
        Self {
            mode,
            limit: 0,
            model_name,
            embedding_cache_dir,
            cosine_threshold: dedup::DEFAULT_COSINE_THRESHOLD,
            full: false,
            batch_size_override: 0,
        }
    }
}

/// Summary of a completed dream pass. Returned to the CLI layer so the
/// binary can emit a single `<result .../>` line at the end.
///
/// Counters split across the three stages:
///   - Stage 0 (project review): `review_*` fields tally per-project
///     LLM decisions.
///   - Stage A (cosine dedup): `superseded`.
///   - Stage B (per-memory condense): `kept`, `rewritten`, `forgot`.
///   - `failed` is shared across stages.
#[derive(Debug, Default)]
pub struct DreamSummary {
    pub total_walked: usize,
    pub kept: usize,
    pub rewritten: usize,
    pub forgot: usize,
    pub superseded: usize,
    pub failed: usize,
    // -- Stage 0 (project review) counters ---------------------------------
    pub review_kept: usize,
    pub review_dropped: usize,
    pub review_merged: usize,
    pub review_superseded: usize,
    pub review_extracted: usize,
}

/// Run a full dream pass.
pub fn run(
    conn: &mut Connection,
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
) -> Result<DreamSummary, DreamError> {
    let mut summary = DreamSummary::default();

    // Enumerate projects. An empty DB yields zero projects and the pass
    // exits cleanly with zero work.
    let projects = q::list_distinct_projects_for_dream(conn)?;

    for project in &projects {
        let project_label = project.as_deref().unwrap_or("(null)");
        let cutoff = resolve_incremental_cutoff(conn, project.as_deref(), cfg)?;
        let current_stamp = prompt::condenser_version_stamp(cfg.model_name);

        let candidates = q::list_dream_candidates(
            conn,
            project.as_deref(),
            cutoff.as_deref(),
            if cfg.full {
                None
            } else {
                Some(current_stamp.as_str())
            },
            cfg.limit,
        )?;

        let walked = candidates.len();
        summary.total_walked += walked;

        println!(
            "{}",
            render::render_action_result(
                "dream_project",
                &[
                    ("project", project_label.to_string()),
                    ("candidates", walked.to_string()),
                ]
            )
        );

        if walked == 0 {
            continue;
        }

        let mut project_stats = ProjectStats::default();

        // Stage 0 — project-level cross-memory review. Primary
        // consolidation path. Survivors (including newly-materialized
        // supersede/extract memories) flow into Stage A + B for the
        // existing cosine + per-memory polish work.
        let stage0_survivors = run_stage_0_project_review(
            conn,
            inference,
            cfg,
            project.as_deref(),
            candidates,
            &mut project_stats,
            &mut summary,
        );

        // Stage A — cosine dedup over the Stage 0 survivors for this project.
        // Survivors move to Stage B.
        let survivors = run_stage_a_dedup(conn, cfg, &stage0_survivors, &mut project_stats);

        // Stage B — per-memory condense via the three-way contract.
        for mem in &survivors {
            run_stage_b_condense(conn, inference, cfg, mem, &mut project_stats);
        }

        summary.kept += project_stats.kept;
        summary.rewritten += project_stats.rewritten;
        summary.forgot += project_stats.forgot;
        summary.superseded += project_stats.superseded;
        summary.failed += project_stats.failed;

        println!(
            "{}",
            render::render_action_result(
                "dream_project_complete",
                &[
                    ("project", project_label.to_string()),
                    ("kept", project_stats.kept.to_string()),
                    ("rewritten", project_stats.rewritten.to_string()),
                    ("forgot", project_stats.forgot.to_string()),
                    ("superseded", project_stats.superseded.to_string()),
                    ("review_kept", project_stats.review_kept.to_string()),
                    ("review_dropped", project_stats.review_dropped.to_string()),
                    ("review_merged", project_stats.review_merged.to_string()),
                    (
                        "review_superseded",
                        project_stats.review_superseded.to_string()
                    ),
                    (
                        "review_extracted",
                        project_stats.review_extracted.to_string()
                    ),
                    ("failed", project_stats.failed.to_string()),
                ]
            )
        );

        // Stamp the project's last_dream_at on Apply mode only — Dry runs
        // must not influence the next incremental pass.
        if cfg.mode == DreamMode::Apply {
            let now = chrono::Utc::now().to_rfc3339();
            if let Err(e) = q::set_last_dream_at(conn, project.as_deref(), &now) {
                tracing::warn!(project = %project_label, error = %e,
                    "failed to stamp project_state.last_dream_at");
            }
        }
    }

    println!(
        "{}",
        render::render_action_result(
            "dream_complete",
            &[
                ("walked", summary.total_walked.to_string()),
                ("kept", summary.kept.to_string()),
                ("rewritten", summary.rewritten.to_string()),
                ("forgot", summary.forgot.to_string()),
                ("superseded", summary.superseded.to_string()),
                ("review_kept", summary.review_kept.to_string()),
                ("review_dropped", summary.review_dropped.to_string()),
                ("review_merged", summary.review_merged.to_string()),
                ("review_superseded", summary.review_superseded.to_string()),
                ("review_extracted", summary.review_extracted.to_string()),
                ("failed", summary.failed.to_string()),
            ]
        )
    );

    Ok(summary)
}

/// Per-project tallies accumulated across Stages 0+A+B. Rolled up into
/// [`DreamSummary`] at the end of each project's iteration.
///
/// Stage 0 (project review) fills the `review_*` fields; Stage A fills
/// `superseded`; Stage B fills `kept`, `rewritten`, `forgot`. `failed`
/// is shared across stages — any transaction-rollback or parse-error
/// hit lands there.
#[derive(Debug, Default, Clone, Copy)]
struct ProjectStats {
    kept: usize,
    rewritten: usize,
    forgot: usize,
    superseded: usize,
    failed: usize,
    // Stage 0 tallies.
    review_kept: usize,
    review_dropped: usize,
    review_merged: usize,
    review_superseded: usize,
    review_extracted: usize,
}

/// Decide which `last_dream_at` cutoff applies for `project`.
///
/// Three cases, collapsed into one return value:
///   - `cfg.full == true`              → `None` (no cutoff; re-walk all).
///   - No prior pass (NULL row)        → `None` (first pass processes everything).
///   - Prior pass timestamp present    → `Some(ts)`.
fn resolve_incremental_cutoff(
    conn: &Connection,
    project: Option<&str>,
    cfg: &DreamConfig<'_>,
) -> Result<Option<String>, DreamError> {
    if cfg.full {
        return Ok(None);
    }
    Ok(q::get_last_dream_at(conn, project)?)
}

/// Stage 0 — project-level cross-memory review.
///
/// Sends the project's full candidate set (or clustered batches for
/// oversized projects) to the model, applies the returned per-memory
/// decisions, and returns the survivors. Survivors include:
///   - Memories the model flagged `keep`.
///   - Newly-materialized memories from `supersede_by` and `extract`
///     decisions (the original rows are deleted in Apply mode).
///
/// Dropped and merged memories are not returned as survivors — they're
/// gone from the pipeline after Stage 0.
///
/// On inference failure the pass degrades gracefully: all candidates
/// pass through as survivors, failures are counted, and Stages A + B
/// still run. This matches the existing `NoopInference` fallback used
/// when no model is available.
fn run_stage_0_project_review(
    conn: &mut rusqlite::Connection,
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
    project: Option<&str>,
    candidates: Vec<Memory>,
    project_stats: &mut ProjectStats,
    summary: &mut DreamSummary,
) -> Vec<Memory> {
    let apply = cfg.mode == DreamMode::Apply;
    let project_label = project.unwrap_or("(null)");

    match project_review::run_project(
        conn,
        inference,
        project,
        candidates.clone(),
        cfg.model_name,
        cfg.embedding_cache_dir,
        apply,
    ) {
        Ok(outcome) => {
            project_stats.review_kept += outcome.stats.kept;
            project_stats.review_dropped += outcome.stats.dropped;
            project_stats.review_merged += outcome.stats.merged;
            project_stats.review_superseded += outcome.stats.superseded;
            project_stats.review_extracted += outcome.stats.extracted;
            project_stats.failed += outcome.stats.failed;

            summary.review_kept += outcome.stats.kept;
            summary.review_dropped += outcome.stats.dropped;
            summary.review_merged += outcome.stats.merged;
            summary.review_superseded += outcome.stats.superseded;
            summary.review_extracted += outcome.stats.extracted;

            println!(
                "{}",
                render::render_action_result(
                    "review_project_complete",
                    &[
                        ("project", project_label.to_string()),
                        ("kept", outcome.stats.kept.to_string()),
                        ("dropped", outcome.stats.dropped.to_string()),
                        ("merged", outcome.stats.merged.to_string()),
                        ("superseded", outcome.stats.superseded.to_string()),
                        ("extracted", outcome.stats.extracted.to_string()),
                        ("failed", outcome.stats.failed.to_string()),
                    ]
                )
            );
            outcome.survivors
        }
        Err(e) => {
            // A Stage 0 failure is a single "batch didn't parse" event,
            // not a per-memory failure. We pass the candidates through
            // to Stages A + B, which do their own counting; inflating
            // `failed` here would double-count with Stage B's work.
            tracing::warn!(project = %project_label, error = %e,
                "project review pass failed; falling back to all-keep");
            println!(
                "{}",
                render::render_action_result(
                    "review_failed",
                    &[
                        ("project", project_label.to_string()),
                        ("reason", format!("{e}")),
                    ]
                )
            );
            candidates
        }
    }
}

/// Stage A — cosine dedup.
///
/// For each candidate we fetch the project's other live rows that share
/// its `memory_type` + `embedding_model` axis (the dedup key), and run
/// [`dedup::find_duplicate`]. Near-matches above `cfg.cosine_threshold`
/// get superseded. Superseded candidates are dropped from the returned
/// survivor list so Stage B doesn't re-condense them.
fn run_stage_a_dedup(
    conn: &Connection,
    cfg: &DreamConfig<'_>,
    candidates: &[Memory],
    stats: &mut ProjectStats,
) -> Vec<Memory> {
    let mut survivors: Vec<Memory> = Vec::with_capacity(candidates.len());
    for mem in candidates {
        let peers = match q::list_dedup_candidates(
            conn,
            &mem.id,
            mem.project.as_deref(),
            mem.memory_type.as_deref(),
            mem.embedding_model.as_deref(),
        ) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(id = %mem.id, error = %e,
                    "stage A dedup peer lookup failed; keeping memory");
                survivors.push(mem.clone());
                continue;
            }
        };

        let decision = dedup::find_duplicate(mem, &peers, cfg.cosine_threshold);
        match decision {
            dedup::DedupDecision::Distinct => survivors.push(mem.clone()),
            _ => {
                if cfg.mode == DreamMode::Apply {
                    match dedup::apply_policy(conn, mem, &decision) {
                        Ok(Some((older, newer))) => {
                            stats.superseded += 1;
                            println!(
                                "{}",
                                render::render_action_result(
                                    "dedup_superseded",
                                    &[
                                        ("older", render::short_id(&older).to_string()),
                                        ("newer", render::short_id(&newer).to_string()),
                                    ]
                                )
                            );
                            // If `mem` itself was the loser, drop it from
                            // the survivor list (condensing a row that's
                            // already hidden from default reads is wasted
                            // work). Otherwise keep it — the sibling lost,
                            // `mem` stays live.
                            if older == mem.id {
                                continue;
                            }
                            survivors.push(mem.clone());
                        }
                        Ok(None) => survivors.push(mem.clone()),
                        Err(e) => {
                            stats.failed += 1;
                            tracing::warn!(id = %mem.id, error = %e,
                                "stage A apply_policy failed");
                            survivors.push(mem.clone());
                        }
                    }
                } else {
                    // Dry mode: surface the intent but do not mutate.
                    stats.superseded += 1;
                    println!(
                        "{}",
                        render::render_action_result(
                            "dedup_would_supersede",
                            &[("id", render::short_id(&mem.id).to_string())]
                        )
                    );
                    survivors.push(mem.clone());
                }
            }
        }
    }
    survivors
}

/// Stage B — per-memory condense using the three-way response contract.
///
/// Each condensation is wrapped in `BEGIN IMMEDIATE` (Apply mode) so
/// concurrent `memory store` / `memory update` writes serialize behind
/// this row. Dry mode rolls the transaction back.
fn run_stage_b_condense(
    conn: &mut Connection,
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
    source: &Memory,
    stats: &mut ProjectStats,
) {
    // Belt-and-braces — candidate list already filters superseded rows.
    if source.superseded_by.is_some() {
        stats.kept += 1;
        return;
    }

    let tx = match conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate) {
        Ok(tx) => tx,
        Err(e) => {
            stats.failed += 1;
            tracing::warn!(id = %source.id, error = %e, "stage B begin-immediate failed");
            println!(
                "{}",
                render::render_action_result(
                    "condense_failed",
                    &[
                        ("id", render::short_id(&source.id).to_string()),
                        ("reason", format!("begin_tx: {e}")),
                    ]
                )
            );
            return;
        }
    };

    let outcome = condense::run_per_memory(inference, source);

    let commit_tx = |tx: rusqlite::Transaction<'_>| -> Result<(), rusqlite::Error> {
        if cfg.mode == DreamMode::Apply {
            tx.commit()
        } else {
            tx.rollback()
        }
    };

    match outcome {
        Ok(condense::Decision::Skip) => {
            stats.kept += 1;
            if let Err(e) = commit_tx(tx) {
                tracing::warn!(id = %source.id, error = %e, "commit after skip failed");
            }
            println!(
                "{}",
                render::render_action_result(
                    "kept",
                    &[("id", render::short_id(&source.id).to_string())]
                )
            );
        }
        Ok(condense::Decision::Forget) => {
            if cfg.mode == DreamMode::Apply {
                match agent_memory::db::queries::delete_memory(&tx, &source.id) {
                    Ok(_) => {
                        stats.forgot += 1;
                        if let Err(e) = commit_tx(tx) {
                            stats.failed += 1;
                            tracing::warn!(id = %source.id, error = %e, "commit after forget failed");
                            return;
                        }
                        println!(
                            "{}",
                            render::render_action_result(
                                "forgot",
                                &[("id", render::short_id(&source.id).to_string())]
                            )
                        );
                    }
                    Err(e) => {
                        stats.failed += 1;
                        let _ = tx.rollback();
                        println!(
                            "{}",
                            render::render_action_result(
                                "condense_failed",
                                &[
                                    ("id", render::short_id(&source.id).to_string()),
                                    ("reason", format!("delete: {e}")),
                                ]
                            )
                        );
                    }
                }
            } else {
                stats.forgot += 1;
                let _ = tx.rollback();
                println!(
                    "{}",
                    render::render_action_result(
                        "would_forget",
                        &[("id", render::short_id(&source.id).to_string())]
                    )
                );
            }
        }
        Ok(condense::Decision::Rewrite { text }) => {
            let bytes_before = source.content.len();
            let bytes_after = text.len();

            // Re-embed the condensed content so vector search doesn't
            // drift from the visible text. Embedding runs outside the
            // transaction because fastembed doesn't touch SQLite.
            let new_emb = match embed_text(&text, cfg.embedding_cache_dir) {
                Ok(v) => v,
                Err(e) => {
                    stats.failed += 1;
                    let _ = tx.rollback();
                    println!(
                        "{}",
                        render::render_action_result(
                            "condense_failed",
                            &[
                                ("id", render::short_id(&source.id).to_string()),
                                ("reason", format!("embed: {e}")),
                            ]
                        )
                    );
                    return;
                }
            };

            if cfg.mode == DreamMode::Apply {
                // Preserve the ORIGINAL raw body: update_content uses
                // COALESCE(content_raw, content) so a re-condensation
                // never chains through an intermediate condensed form.
                match q::update_condensation(
                    &tx,
                    &source.id,
                    &text,
                    source.content_raw.as_deref().unwrap_or(&source.content),
                    &prompt::condenser_version_stamp(cfg.model_name),
                    &new_emb,
                    EMBEDDING_MODEL_NAME,
                ) {
                    Ok(()) => {
                        stats.rewritten += 1;
                        if let Err(e) = commit_tx(tx) {
                            stats.failed += 1;
                            tracing::warn!(id = %source.id, error = %e, "commit after rewrite failed");
                            return;
                        }
                        println!(
                            "{}",
                            render::render_action_result(
                                "rewritten",
                                &[
                                    ("id", render::short_id(&source.id).to_string()),
                                    ("bytes_before", bytes_before.to_string()),
                                    ("bytes_after", bytes_after.to_string()),
                                ]
                            )
                        );
                    }
                    Err(e) => {
                        stats.failed += 1;
                        let _ = tx.rollback();
                        println!(
                            "{}",
                            render::render_action_result(
                                "condense_failed",
                                &[
                                    ("id", render::short_id(&source.id).to_string()),
                                    ("reason", format!("update: {e}")),
                                ]
                            )
                        );
                    }
                }
            } else {
                stats.rewritten += 1;
                let _ = tx.rollback();
                println!(
                    "{}",
                    render::render_action_result(
                        "would_rewrite",
                        &[
                            ("id", render::short_id(&source.id).to_string()),
                            ("bytes_before", bytes_before.to_string()),
                            ("bytes_after", bytes_after.to_string()),
                        ]
                    )
                );
            }
        }
        Err(e) => {
            stats.failed += 1;
            let _ = tx.rollback();
            tracing::info!(id = %source.id, error = %e, "stage B condense failed");
            println!(
                "{}",
                render::render_action_result(
                    "condense_failed",
                    &[
                        ("id", render::short_id(&source.id).to_string()),
                        ("reason", format!("{e}")),
                    ]
                )
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FixedInference;
    use std::path::PathBuf;

    fn open_mem_db() -> Connection {
        agent_memory::db::open_database(&PathBuf::from(":memory:")).expect("open in-memory db")
    }

    fn insert(conn: &Connection, id: &str, content: &str, project: Option<&str>) {
        let mut m = Memory::new(
            content.to_string(),
            None,
            project.map(String::from),
            None,
            None,
            Some("user".to_string()),
        );
        m.id = id.to_string();
        m.embedding_model = Some(EMBEDDING_MODEL_NAME.to_string());
        q::insert_memory(conn, &m).expect("insert");
    }

    /// Zero-memory DB: the orchestrator exits cleanly with zero counts.
    #[test]
    fn empty_db_exits_cleanly() {
        let mut conn = open_mem_db();
        let inf = FixedInference::new("skip");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.total_walked, 0);
        assert_eq!(summary.kept, 0);
        assert_eq!(summary.rewritten, 0);
        assert_eq!(summary.forgot, 0);
    }

    /// Incremental filter: after a successful pass, re-running with no new
    /// writes should yield zero candidates.
    #[test]
    fn incremental_filter_skips_processed_projects() {
        let mut conn = open_mem_db();
        let stamp = prompt::condenser_version_stamp("sonnet");
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type,
                                   created_at, updated_at, condenser_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                "aaaaaaaa-0000-1111-2222-000000000001",
                "first",
                "p1",
                "user",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                stamp,
            ],
        )
        .unwrap();
        q::set_last_dream_at(&conn, Some("p1"), "2099-01-01T00:00:00Z").unwrap();

        let inf = FixedInference::new("skip");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.total_walked, 0, "incremental filter must skip rows");
    }

    /// --full flag overrides the incremental cutoff — every row is walked.
    #[test]
    fn full_flag_re_walks_everything() {
        let mut conn = open_mem_db();
        insert(
            &conn,
            "aaaaaaaa-0000-1111-2222-000000000001",
            "first",
            Some("p1"),
        );
        q::set_last_dream_at(&conn, Some("p1"), "2099-01-01T00:00:00Z").unwrap();

        let inf = FixedInference::new("skip");
        let tmp = std::env::temp_dir();
        let mut cfg = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        cfg.full = true;
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.total_walked, 1, "--full must ignore cutoff");
    }

    /// Helper for the --refresh stage tests: insert a memory that would
    /// normally be skipped by the incremental gate (stamped with the
    /// current condenser_version AND updated_at older than
    /// project_state.last_dream_at). Returns the memory id.
    ///
    /// The three tests below drive this shape through the pipeline with
    /// and without `cfg.full = true` to confirm every stage honors the
    /// --refresh override (which folds into `cfg.full` in `main.rs`).
    fn insert_already_processed(conn: &Connection, id: &str, content: &str, project: &str) {
        let stamp = prompt::condenser_version_stamp("sonnet");
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type,
                                   created_at, updated_at, condenser_version,
                                   embedding_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                id,
                content,
                project,
                "user",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                stamp,
                EMBEDDING_MODEL_NAME,
            ],
        )
        .unwrap();
        q::set_last_dream_at(conn, Some(project), "2099-01-01T00:00:00Z").unwrap();
    }

    /// --refresh (via `cfg.full`) re-feeds an already-processed memory
    /// into Stage 0 (project review). Without it, the memory is skipped
    /// by the incremental gate in `list_dream_candidates` and Stage 0
    /// never sees it. The test asserts Stage 0 actually ran by checking
    /// `review_kept > 0` in the summary — that counter is only bumped
    /// inside project_review::run_project.
    #[test]
    fn refresh_flag_reruns_stage_0_project_review() {
        let mut conn = open_mem_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000010";
        insert_already_processed(&conn, id, "already reviewed", "p1");

        // Canned response that keeps the memory so it survives into
        // Stage A and B (we only care that Stage 0 fired).
        let canned = format!(
            r#"{{"decisions": {{"{id}": {{"action": "keep"}}}}}}"#,
            id = id
        );
        let inf = FixedInference::new(canned);
        let tmp = std::env::temp_dir();

        // Sanity: without --refresh the memory is skipped entirely.
        let cfg_skip = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary_skip = run(&mut conn, &inf, &cfg_skip).expect("dream ok");
        assert_eq!(
            summary_skip.total_walked, 0,
            "precondition: incremental gate must hide the memory by default"
        );
        assert_eq!(summary_skip.review_kept, 0, "Stage 0 must not fire by default");

        // With --refresh the memory re-enters the pipeline and Stage 0
        // sees it.
        let mut cfg_refresh = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        cfg_refresh.full = true;
        let summary_refresh = run(&mut conn, &inf, &cfg_refresh).expect("dream ok");
        assert!(
            summary_refresh.review_kept >= 1,
            "--refresh must let Stage 0 (project review) reprocess the memory; got review_kept={}",
            summary_refresh.review_kept
        );
    }

    /// --refresh re-feeds already-processed memories into Stage A
    /// (cosine dedup). Two byte-identical rows are pre-stamped so the
    /// default gate skips them; with refresh on, Stage A fires and
    /// supersedes the older row.
    #[test]
    fn refresh_flag_reruns_stage_a_dedup() {
        let mut conn = open_mem_db();
        let older = "aaaaaaaa-0000-1111-2222-000000000020";
        let newer = "aaaaaaaa-0000-1111-2222-000000000021";

        let stamp = prompt::condenser_version_stamp("sonnet");
        // Two byte-identical rows in the same project / memory_type /
        // embedding_model so Stage A's exact-match short-circuit fires.
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type,
                                   created_at, updated_at, condenser_version,
                                   embedding_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                older,
                "identical content",
                "p2",
                "user",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                stamp,
                EMBEDDING_MODEL_NAME,
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories (id, content, project, memory_type,
                                   created_at, updated_at, condenser_version,
                                   embedding_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                newer,
                "identical content",
                "p2",
                "user",
                "2026-01-02T00:00:00Z",
                "2026-01-02T00:00:00Z",
                stamp,
                EMBEDDING_MODEL_NAME,
            ],
        )
        .unwrap();
        q::set_last_dream_at(&conn, Some("p2"), "2099-01-01T00:00:00Z").unwrap();

        // Stage 0 response must keep both memories so they survive into
        // Stage A where dedup actually runs.
        let canned = format!(
            r#"{{"decisions": {{
                "{older}": {{"action": "keep"}},
                "{newer}": {{"action": "keep"}}
            }}}}"#
        );
        let inf = FixedInference::new(canned);
        let tmp = std::env::temp_dir();

        // Without --refresh: incremental gate hides both → no dedup work.
        let cfg_skip = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary_skip = run(&mut conn, &inf, &cfg_skip).expect("dream ok");
        assert_eq!(summary_skip.superseded, 0,
            "precondition: Stage A must not fire by default on pre-stamped rows");

        // With --refresh: Stage A fires and supersedes the older row.
        let mut cfg_refresh = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        cfg_refresh.full = true;
        let summary_refresh = run(&mut conn, &inf, &cfg_refresh).expect("dream ok");
        assert!(
            summary_refresh.superseded >= 1,
            "--refresh must let Stage A (cosine dedup) reprocess; got superseded={}",
            summary_refresh.superseded
        );
    }

    /// --refresh re-feeds an already-processed memory into Stage B
    /// (per-memory condense).
    ///
    /// Wiring note: the orchestrator uses one [`FixedInference`] stub for
    /// both Stage 0 (JSON project-review response) and Stage B (free-text
    /// three-way contract). We pad the memory body with enough filler so
    /// the JSON payload is strictly shorter than the input — Stage B's
    /// parser then accepts it as a rewrite, incrementing `rewritten`.
    /// That bump is the signal Stage B fired; without --refresh it
    /// stays at zero because the incremental gate hides the memory.
    #[test]
    fn refresh_flag_reruns_stage_b_condense() {
        let mut conn = open_mem_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000030";

        // Pad the memory body so any plausible JSON payload is shorter.
        // The actual content doesn't matter — only the length does,
        // because Stage B rejects rewrites that aren't strictly shorter
        // than the input. 300 chars is comfortably above the ~70-char
        // keep-decision JSON below.
        let long_content = "x".repeat(300);
        insert_already_processed(&conn, id, &long_content, "p3");

        // Stage 0 keeps the memory; Stage B gets the same string and
        // treats it as a (shorter than input) rewrite body.
        let canned = format!(
            r#"{{"decisions": {{"{id}": {{"action": "keep"}}}}}}"#,
            id = id
        );
        let inf = FixedInference::new(canned);
        let tmp = std::env::temp_dir();

        // Without --refresh: Stage B never fires.
        let cfg_skip = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary_skip = run(&mut conn, &inf, &cfg_skip).expect("dream ok");
        let stage_b_touched_default =
            summary_skip.kept + summary_skip.rewritten + summary_skip.forgot;
        assert_eq!(
            stage_b_touched_default, 0,
            "precondition: Stage B must not fire by default"
        );

        // With --refresh: Stage B fires. Some non-failure outcome lands
        // in kept/rewritten/forgot; we don't care which — only that
        // Stage B ran at all.
        let mut cfg_refresh = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        cfg_refresh.full = true;
        let summary_refresh = run(&mut conn, &inf, &cfg_refresh).expect("dream ok");
        let stage_b_touched =
            summary_refresh.kept + summary_refresh.rewritten + summary_refresh.forgot;
        assert!(
            stage_b_touched >= 1,
            "--refresh must let Stage B (per-memory condense) reprocess; \
             kept={}, rewritten={}, forgot={}, failed={}",
            summary_refresh.kept,
            summary_refresh.rewritten,
            summary_refresh.forgot,
            summary_refresh.failed,
        );
    }

    /// `skip` response keeps the memory untouched.
    #[test]
    fn skip_response_keeps_memory() {
        let mut conn = open_mem_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id, "already concise", Some("p1"));

        let inf = FixedInference::new("skip");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.kept, 1);
        assert_eq!(summary.rewritten, 0);
        assert_eq!(summary.forgot, 0);

        // Memory still present.
        let got = q::get_memory_by_id(&conn, id).unwrap();
        assert_eq!(got.content, "already concise");
    }

    /// `forget` response deletes the memory via the DB layer.
    #[test]
    fn forget_response_deletes_memory() {
        let mut conn = open_mem_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id, "CI notification noise", Some("p1"));

        let inf = FixedInference::new("forget");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.forgot, 1);
        assert_eq!(summary.kept, 0);

        // Memory is gone.
        let err = q::get_memory_by_id(&conn, id).unwrap_err();
        assert!(matches!(err, MemoryError::NotFound(_)));
    }

    /// `forget` dry-run surfaces intent without deleting.
    #[test]
    fn forget_response_dry_run_does_not_delete() {
        let mut conn = open_mem_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id, "CI notification noise", Some("p1"));

        let inf = FixedInference::new("forget");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Dry, "sonnet", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.forgot, 1, "counts still reflect intent");

        // Row must still exist — dry run doesn't delete.
        assert!(q::get_memory_by_id(&conn, id).is_ok());
    }

    /// Apply-mode pass stamps `project_state.last_dream_at`.
    #[test]
    fn apply_mode_stamps_project_state() {
        let mut conn = open_mem_db();
        insert(
            &conn,
            "aaaaaaaa-0000-1111-2222-000000000001",
            "first",
            Some("p1"),
        );

        let inf = FixedInference::new("skip");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        run(&mut conn, &inf, &cfg).expect("dream ok");

        let ts = q::get_last_dream_at(&conn, Some("p1"))
            .unwrap()
            .expect("project_state row must exist after apply");
        assert!(!ts.is_empty());
    }

    /// Dry-mode pass does NOT stamp `project_state`.
    #[test]
    fn dry_mode_does_not_stamp_project_state() {
        let mut conn = open_mem_db();
        insert(
            &conn,
            "aaaaaaaa-0000-1111-2222-000000000001",
            "first",
            Some("p1"),
        );

        let inf = FixedInference::new("skip");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Dry, "sonnet", &tmp);
        run(&mut conn, &inf, &cfg).expect("dream ok");

        let ts = q::get_last_dream_at(&conn, Some("p1")).unwrap();
        assert!(ts.is_none(), "dry-run must not stamp project_state");
    }

    /// A malformed response (neither `skip`/`forget` nor shorter than
    /// input) lands in the failed bucket and the row survives unchanged.
    #[test]
    fn malformed_response_is_counted_as_failed() {
        let mut conn = open_mem_db();
        let id = "aaaaaaaa-0000-1111-2222-000000000001";
        insert(&conn, id, "short", Some("p1"));

        // Inference stub returns a string LONGER than the input — the
        // length-guard in condense should reject it.
        let inf = FixedInference::new("this is a much longer response than the raw input");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "sonnet", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.failed, 1);

        // Row still there with original content.
        let got = q::get_memory_by_id(&conn, id).unwrap();
        assert_eq!(got.content, "short");
    }
}
