//! Dream orchestrator (Release 2.3).
//!
//! Entry point: [`run`]. Called once per `memory-dream` invocation.
//! Responsible for:
//!
//! 1. **Probe** the backend for shell-tool support ([`agentic::probe_tool_support`]).
//!    The result is cached for the whole pass — we never re-probe.
//! 2. **Enumerate projects** with at least one live memory.
//! 3. **For each project**, pull incremental candidates (rows newer than
//!    `project_state.last_dream_at`, or with a stale `condenser_version`)
//!    and route to either:
//!      - **Agentic path** ([`agentic::run_agentic_batch`]) — one batched
//!        LLM call per N memories; the LLM invokes `memory` CLI tools
//!        directly to curate. No parsing; DB is the output.
//!      - **Non-agentic path** ([`process_one_non_agentic`]) — per-memory
//!        plain-bullets condensation. No batching, no discards, no scope
//!        reclassify. Local-candle backends always take this path.
//! 4. **Stamp** `project_state.last_dream_at` per project after the batch
//!    completes so the next incremental run skips untouched rows.
//!
//! Progress is emitted as light-XML on stdout — same renderer the rest of
//! the project uses — so a caller can pipe `memory-dream` output into a
//! log collector alongside `memory` command output.

pub mod agentic;
pub mod condense;
pub mod dedup;
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

use self::agentic::probe_tool_support;
use self::condense::condense;

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
    /// Walk candidate memories and report intended decisions. In agentic
    /// mode this means "probe + enumerate + print candidate counts", NOT
    /// "invoke the LLM in read-only mode" — the LLM has no dry-run bit we
    /// can flip, so dry-run for agentic is "what WOULD we hand to the
    /// model". Non-agentic dry-run condenses in-memory but doesn't persist.
    Dry,
    /// Commit changes.
    Apply,
}

/// Short name used for the embedding model column. Mirrors the fastembed
/// default in `agent_memory::embedding::embed_text`.
pub const EMBEDDING_MODEL_NAME: &str = "all-MiniLM-L6-v2";

/// Default maximum memories per agentic batch. Tuned for Claude's context
/// window (~200k tokens): 100 memories × ~200 chars preview each ≈ 20k
/// tokens, leaving plenty of room for tool-call round trips.
pub const DEFAULT_AGENTIC_BATCH_SIZE: usize = 100;

/// Default batch size for non-agentic per-memory condensation. One at a
/// time — there is no cross-memory reasoning to amortize.
pub const DEFAULT_NON_AGENTIC_BATCH_SIZE: usize = 1;

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
    /// Cosine threshold (reserved for future reuse; non-agentic path
    /// currently does not run dedup).
    pub cosine_threshold: f32,
    /// Force a full walk, ignoring `project_state.last_dream_at`.
    /// Set via `memory-dream --full`.
    pub full: bool,
    /// Override for agentic batch size (0 = use default).
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

    /// Resolve the batch size for a given mode based on the override (if
    /// set) and the agentic/non-agentic defaults.
    fn effective_batch_size(&self, agentic: bool) -> usize {
        if self.batch_size_override > 0 {
            self.batch_size_override
        } else if agentic {
            DEFAULT_AGENTIC_BATCH_SIZE
        } else {
            DEFAULT_NON_AGENTIC_BATCH_SIZE
        }
    }
}

/// Summary of a completed dream pass. Returned to the CLI layer so the
/// binary can emit a single `<result .../>` line at the end.
#[derive(Debug, Default)]
pub struct DreamSummary {
    pub total_walked: usize,
    pub condensed: usize,
    pub agentic_batches: usize,
    pub skipped: usize,
    pub errors: usize,
}

/// Run a full dream pass.
pub fn run(
    conn: &mut Connection,
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
) -> Result<DreamSummary, DreamError> {
    // One-shot probe. Cached for the remainder of the pass — re-probing
    // per batch would double latency with no benefit since a backend's
    // tool support doesn't change mid-run.
    let probe = probe_tool_support(inference);
    let agentic = probe.supports_tools();

    println!(
        "{}",
        render::render_action_result(
            "dream_probe",
            &[
                (
                    "mode",
                    if agentic { "agentic" } else { "non_agentic" }.to_string()
                ),
                ("outcome", format!("{probe:?}").to_lowercase()),
            ]
        )
    );

    let mut summary = DreamSummary::default();

    // Enumerate projects. An empty DB yields zero projects and the pass
    // exits cleanly with zero work.
    let projects = q::list_distinct_projects_for_dream(conn)?;

    for project in &projects {
        let project_label = project.as_deref().unwrap_or("(null)");
        let cutoff = resolve_incremental_cutoff(conn, project.as_deref(), cfg)?;
        let current_stamp = prompt::condenser_version_stamp(cfg.model_name);

        // `list_dream_candidates` handles the --full branch internally:
        // passing `last_dream_at = None` disables the time filter, which
        // is exactly what we want when `cfg.full` is true.
        let candidates = q::list_dream_candidates(
            conn,
            project.as_deref(),
            cutoff.as_deref(),
            // On --full, also drop the stamp-based filter so every row
            // re-enters the pipeline regardless of its condenser_version.
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

        if agentic {
            process_project_agentic(inference, cfg, project_label, &candidates, &mut summary);
        } else {
            for mem in &candidates {
                match process_one_non_agentic(conn, inference, cfg, mem) {
                    Ok(Outcome::Condensed) => summary.condensed += 1,
                    Ok(Outcome::Skipped) => summary.skipped += 1,
                    Err(e) => {
                        summary.errors += 1;
                        tracing::warn!(id = %mem.id, error = %e, "dream failure on memory");
                        println!(
                            "{}",
                            render::render_action_result(
                                "dream_error",
                                &[
                                    ("id", render::short_id(&mem.id).to_string()),
                                    ("error", format!("{e}")),
                                ]
                            )
                        );
                    }
                }
            }
        }

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
                ("condensed", summary.condensed.to_string()),
                ("agentic_batches", summary.agentic_batches.to_string()),
                ("skipped", summary.skipped.to_string()),
                ("errors", summary.errors.to_string()),
                (
                    "mode",
                    if agentic { "agentic" } else { "non_agentic" }.to_string()
                ),
            ]
        )
    );

    Ok(summary)
}

/// Per-memory disposition in the non-agentic path.
enum Outcome {
    Condensed,
    Skipped,
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

/// Agentic batch driver. Slices `candidates` into batches of
/// `cfg.effective_batch_size(true)` and invokes the LLM once per batch.
fn process_project_agentic(
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
    project_label: &str,
    candidates: &[Memory],
    summary: &mut DreamSummary,
) {
    let batch_size = cfg.effective_batch_size(/* agentic */ true);
    for chunk in candidates.chunks(batch_size.max(1)) {
        println!(
            "{}",
            render::render_action_result(
                "dream_batch_start",
                &[
                    ("project", project_label.to_string()),
                    ("size", chunk.len().to_string()),
                ]
            )
        );

        if cfg.mode == DreamMode::Dry {
            // Dry-run for agentic: do NOT invoke the model. The LLM has
            // no read-only mode, so the safest dry-run surfaces the
            // candidate list and stops.
            println!(
                "{}",
                render::render_action_result(
                    "dream_batch_dry_run",
                    &[
                        ("project", project_label.to_string()),
                        ("would_process", chunk.len().to_string()),
                    ]
                )
            );
            continue;
        }

        match agentic::run_agentic_batch(inference, project_label, chunk) {
            Ok(report) => {
                summary.agentic_batches += 1;
                println!(
                    "{}",
                    render::render_action_result(
                        "dream_batch_complete",
                        &[
                            ("project", project_label.to_string()),
                            ("memories", report.memories_in_batch.to_string()),
                            ("reply_bytes", report.reply_bytes.to_string()),
                        ]
                    )
                );
            }
            Err(e) => {
                summary.errors += 1;
                tracing::warn!(project = %project_label, error = %e, "agentic batch failed");
                println!(
                    "{}",
                    render::render_action_result(
                        "dream_batch_error",
                        &[
                            ("project", project_label.to_string()),
                            ("error", format!("{e}")),
                        ]
                    )
                );
            }
        }
    }
}

/// Non-agentic per-memory condensation.
///
/// Wrapped in a `BEGIN IMMEDIATE` transaction in Apply mode so concurrent
/// `memory store` / `memory update` writes serialize behind this row. Dry
/// mode rolls the transaction back so no side effect reaches the DB.
fn process_one_non_agentic(
    conn: &mut Connection,
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
    source: &Memory,
) -> Result<Outcome, DreamError> {
    if source.superseded_by.is_some() {
        // Belt-and-braces; the candidate query already filters these out.
        return Ok(Outcome::Skipped);
    }

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

    let want_stamp = prompt::condenser_version_stamp(cfg.model_name);
    let needs_condensation = source.content_raw.is_none()
        || source
            .condenser_version
            .as_deref()
            .map(|v| v != want_stamp)
            .unwrap_or(true);

    let mut outcome = Outcome::Skipped;

    if needs_condensation {
        match condense(inference, cfg.model_name, &source.content) {
            Ok(c) => {
                let new_emb =
                    embed_text(&c.text, cfg.embedding_cache_dir).map_err(DreamError::from)?;

                if cfg.mode == DreamMode::Apply {
                    q::update_condensation(
                        &tx,
                        &source.id,
                        &c.text,
                        // First-pass condensation moves `content` to
                        // `content_raw`. Re-condensations must keep the
                        // very first raw form — never chain through a
                        // prior condensed body.
                        source.content_raw.as_deref().unwrap_or(&source.content),
                        &c.version,
                        &new_emb,
                        EMBEDDING_MODEL_NAME,
                    )?;
                }
                outcome = Outcome::Condensed;
            }
            Err(e) => {
                tracing::info!(id = %source.id, error = %e,
                    "non-agentic condense skipped; keeping raw content");
            }
        }
    }

    if cfg.mode == DreamMode::Apply {
        tx.commit()?;
    } else {
        tx.rollback()?;
    }

    Ok(outcome)
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

    /// Zero-memory DB: the orchestrator runs the probe and exits cleanly.
    #[test]
    fn empty_db_exits_cleanly() {
        let mut conn = open_mem_db();
        let inf = FixedInference::new("NO_TOOLS");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "gemma3", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.total_walked, 0);
        assert_eq!(summary.condensed, 0);
        assert_eq!(summary.agentic_batches, 0);
    }

    /// Incremental filter: after a successful pass, re-running with no new
    /// writes should yield zero candidates.
    ///
    /// The memory must also have a `condenser_version` matching the current
    /// stamp — otherwise the stale-stamp branch keeps it in the candidate
    /// pool. This mirrors real behavior where the first agentic pass writes
    /// both the last_dream_at row AND stamps each condensed memory.
    #[test]
    fn incremental_filter_skips_processed_projects() {
        let mut conn = open_mem_db();

        // Seed a memory pre-stamped with the current condenser version so
        // the stale branch of the incremental filter doesn't pull it back
        // in. This mirrors the state a memory would be in AFTER a
        // successful non-agentic condensation.
        let stamp = prompt::condenser_version_stamp("gemma3");
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

        // Cutoff is after the row's updated_at, so the time filter rejects
        // it too.
        q::set_last_dream_at(&conn, Some("p1"), "2099-01-01T00:00:00Z").unwrap();

        let inf = FixedInference::new("NO_TOOLS");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "gemma3", &tmp);
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

        let inf = FixedInference::new("NO_TOOLS");
        let tmp = std::env::temp_dir();
        let mut cfg = DreamConfig::new(DreamMode::Apply, "gemma3", &tmp);
        cfg.full = true;
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.total_walked, 1, "--full must ignore cutoff");
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

        let inf = FixedInference::new("NO_TOOLS");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "gemma3", &tmp);
        run(&mut conn, &inf, &cfg).expect("dream ok");

        let ts = q::get_last_dream_at(&conn, Some("p1"))
            .unwrap()
            .expect("project_state row must exist after apply");
        assert!(!ts.is_empty());
    }

    /// Dry-mode pass does NOT stamp `project_state` — the next run must
    /// re-walk as if the dry pass never happened.
    #[test]
    fn dry_mode_does_not_stamp_project_state() {
        let mut conn = open_mem_db();
        insert(
            &conn,
            "aaaaaaaa-0000-1111-2222-000000000001",
            "first",
            Some("p1"),
        );

        let inf = FixedInference::new("NO_TOOLS");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Dry, "gemma3", &tmp);
        run(&mut conn, &inf, &cfg).expect("dream ok");

        let ts = q::get_last_dream_at(&conn, Some("p1")).unwrap();
        assert!(ts.is_none(), "dry-run must not stamp project_state");
    }

    /// Agentic probe + batch: a supported probe routes into the agentic
    /// driver, which records one batch per chunk.
    #[test]
    fn supported_probe_runs_agentic_batch() {
        let mut conn = open_mem_db();
        insert(
            &conn,
            "aaaaaaaa-0000-1111-2222-000000000001",
            "m1",
            Some("p1"),
        );
        insert(
            &conn,
            "bbbbbbbb-0000-1111-2222-000000000002",
            "m2",
            Some("p1"),
        );

        let inf = FixedInference::new("memory 1.3.0");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Apply, "gemma3", &tmp);
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.agentic_batches, 1);
        assert_eq!(summary.total_walked, 2);
    }

    /// --batch-size override splits a large candidate list into multiple
    /// agentic batches. Seeding 5 rows with batch_size=2 must produce 3
    /// batches.
    #[test]
    fn batch_size_override_splits_agentic_work() {
        let mut conn = open_mem_db();
        for i in 1..=5 {
            insert(
                &conn,
                &format!("00000000-0000-1111-2222-00000000000{i}"),
                &format!("m{i}"),
                Some("p1"),
            );
        }

        let inf = FixedInference::new("memory 1.3.0");
        let tmp = std::env::temp_dir();
        let mut cfg = DreamConfig::new(DreamMode::Apply, "gemma3", &tmp);
        cfg.batch_size_override = 2;
        let summary = run(&mut conn, &inf, &cfg).expect("dream ok");
        assert_eq!(summary.total_walked, 5);
        assert_eq!(summary.agentic_batches, 3, "5/2 should produce 3 batches");
    }
}
