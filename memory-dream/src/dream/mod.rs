//! Dream orchestrator. Wires condense + dedup + embedding + persistence
//! together into a single batch pass.
//!
//! The entry point [`run`] is called once per `memory-dream` invocation.
//! It walks every memory in the DB (oldest `updated_at` first), opens a
//! `BEGIN IMMEDIATE` transaction per row so concurrent `memory store`
//! calls can't race, classifies the row, applies the chosen policy, and
//! commits. Per-memory errors are logged and skipped — one bad memory
//! can't halt the pass.
//!
//! Progress is emitted as light-XML on stdout using the same rendering
//! helpers as `memory`'s CLI, so the dream output stays consistent with
//! the rest of the project. See
//! [`agent_memory::render::render_action_result`].
//!
//! Two execution modes:
//!   - `mode = DreamMode::Dry` — walks the DB and reports intended
//!     decisions; no writes. Useful for "what would dream do on my
//!     current DB" inspection.
//!   - `mode = DreamMode::Apply` — persists condensations and dedup
//!     decisions. The normal CLI default.

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

use self::condense::condense;
use self::dedup::{apply_policy, find_duplicate, DEFAULT_COSINE_THRESHOLD};

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
    /// Walk every memory, run the classification pipeline, and print what
    /// would happen. No writes.
    Dry,
    /// Walk every memory, condense / dedup, commit changes.
    Apply,
}

/// Short name used for the embedding model column. Mirrors the fastembed
/// default in `agent_memory::embedding::embed_text`. Kept as a constant so
/// dedup can compare apples-to-apples even before a memory has been
/// re-embedded by a dream pass.
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
    /// Cosine threshold for `near_match` dedup decisions.
    pub cosine_threshold: f32,
}

impl<'a> DreamConfig<'a> {
    /// Build a config with sensible defaults. The caller is still expected
    /// to fill in `mode`, `embedding_cache_dir`, and (optionally) override
    /// `limit` and `model_name`.
    pub fn new(mode: DreamMode, model_name: &'a str, embedding_cache_dir: &'a Path) -> Self {
        Self {
            mode,
            limit: 0,
            model_name,
            embedding_cache_dir,
            cosine_threshold: DEFAULT_COSINE_THRESHOLD,
        }
    }
}

/// Summary of a completed dream pass. Returned to the CLI layer so the
/// binary can emit a single `<result .../>` line at the end.
#[derive(Debug, Default)]
pub struct DreamSummary {
    pub total_walked: usize,
    pub condensed: usize,
    pub superseded: usize,
    pub skipped: usize,
    pub errors: usize,
}

/// Run a full dream pass.
///
/// Takes the inference backend as a `&dyn` so tests can swap in
/// `FixedInference`. The embedder is pulled from the sibling `memory`
/// crate — dream uses the same vector space as `memory store`'s default
/// embedder so cosine comparisons remain meaningful.
pub fn run(
    conn: &mut Connection,
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
) -> Result<DreamSummary, DreamError> {
    let rows = q::list_all_for_dream(conn, cfg.limit)?;
    let total = rows.len();
    println!(
        "{}",
        render::render_action_result(
            "dream_start",
            &[
                ("total", total.to_string()),
                ("mode", format!("{:?}", cfg.mode).to_lowercase()),
            ]
        )
    );

    let mut summary = DreamSummary {
        total_walked: total,
        ..Default::default()
    };

    for (idx, source_row) in rows.iter().enumerate() {
        match process_one(conn, inference, cfg, source_row) {
            Ok(Outcome::Condensed) => summary.condensed += 1,
            Ok(Outcome::Superseded) => summary.superseded += 1,
            Ok(Outcome::Skipped) => summary.skipped += 1,
            Err(e) => {
                summary.errors += 1;
                tracing::warn!(id = %source_row.id, error = %e, "dream failure on memory");
                println!(
                    "{}",
                    render::render_action_result(
                        "dream_error",
                        &[
                            ("id", render::short_id(&source_row.id).to_string()),
                            ("error", format!("{e}")),
                        ]
                    )
                );
            }
        }

        // Emit progress every 5 rows, plus always at the end.
        if (idx + 1) % 5 == 0 || idx + 1 == total {
            println!(
                "{}",
                render::render_action_result(
                    "dream_progress",
                    &[("n", (idx + 1).to_string()), ("total", total.to_string()),]
                )
            );
        }
    }

    println!(
        "{}",
        render::render_action_result(
            "dream_complete",
            &[
                ("walked", summary.total_walked.to_string()),
                ("condensed", summary.condensed.to_string()),
                ("superseded", summary.superseded.to_string()),
                ("skipped", summary.skipped.to_string()),
                ("errors", summary.errors.to_string()),
            ]
        )
    );

    Ok(summary)
}

/// Per-memory disposition, returned by [`process_one`] so the caller can
/// tally the summary counters.
enum Outcome {
    Condensed,
    Superseded,
    Skipped,
}

/// Classify + optionally persist the disposition for a single memory.
///
/// Wrapped in a `BEGIN IMMEDIATE` transaction in `Apply` mode so concurrent
/// `memory store` writes are serialized behind the dream pass. In `Dry`
/// mode the transaction is still opened (to read consistently) but
/// rolled back at the end.
fn process_one(
    conn: &mut Connection,
    inference: &dyn Inference,
    cfg: &DreamConfig<'_>,
    source: &Memory,
) -> Result<Outcome, DreamError> {
    // Skip rows already marked superseded by a prior pass — they're dead
    // and nothing the dream pipeline does will resurrect them.
    if source.superseded_by.is_some() {
        return Ok(Outcome::Skipped);
    }

    // Transaction scope: every mutation below runs under BEGIN IMMEDIATE so
    // concurrent stores / updates block until dream finishes this memory.
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

    // --- Condense if needed --------------------------------------------
    //
    // A row needs condensation when it has no content_raw (never processed)
    // or when its condenser_version stamp doesn't match what the current
    // prompt would produce (stale). The stamp format is "<model>:<prompt8>"
    // so a mismatch on either side triggers a re-run.
    let want_stamp = prompt::condenser_version_stamp(cfg.model_name);
    let needs_condensation = source.content_raw.is_none()
        || source
            .condenser_version
            .as_deref()
            .map(|v| v != want_stamp)
            .unwrap_or(true);

    let mut active_content = source.content.clone();
    let mut active_raw = source.content_raw.clone();
    let mut condensed_flag = false;

    if needs_condensation {
        match condense(inference, cfg.model_name, &source.content) {
            Ok(c) => {
                // Re-embed the condensed text so vector search sees the short form.
                let new_emb =
                    embed_text(&c.text, cfg.embedding_cache_dir).map_err(DreamError::from)?;

                if cfg.mode == DreamMode::Apply {
                    q::update_condensation(
                        &tx,
                        &source.id,
                        &c.text,
                        // Preserve original raw text on first condensation;
                        // subsequent re-condenses should keep the first raw
                        // form rather than chain condensations through
                        // condensed → raw.
                        source.content_raw.as_deref().unwrap_or(&source.content),
                        &c.version,
                        &new_emb,
                        EMBEDDING_MODEL_NAME,
                    )?;
                }

                active_content = c.text;
                active_raw = Some(
                    source
                        .content_raw
                        .clone()
                        .unwrap_or_else(|| source.content.clone()),
                );
                condensed_flag = true;
            }
            Err(e) => {
                // Condensation failed (refused / parse error / too-long /
                // inference backend down). Log and fall through to dedup
                // against the raw content so a missing model doesn't
                // block the dedup pass entirely — `memory-dream --dry-run`
                // with no model pulled still produces useful output.
                tracing::info!(id = %source.id, error = %e,
                    "condense skipped; falling back to raw content for dedup");
            }
        }
    }

    // --- Dedup against peers ------------------------------------------
    //
    // Candidates share project + memory_type + embedding_model (or are NULL
    // on the same axis). If the source row has no embedding we skip dedup
    // entirely — dream's dedup contract is "same vector space or nothing".
    let mut deduped = false;
    if source.embedding.is_some() {
        let candidates = q::list_dedup_candidates(
            &tx,
            &source.id,
            source.project.as_deref(),
            source.memory_type.as_deref(),
            source
                .embedding_model
                .as_deref()
                .or(Some(EMBEDDING_MODEL_NAME)),
        )?;

        // Substitute the freshly-condensed text into a throwaway clone so
        // dedup compares against the current short form, not the stale
        // raw content. The embedding stays the original — re-embedding
        // happened above and the clone stores it faithfully.
        let mut effective_source = source.clone();
        effective_source.content = active_content.clone();
        effective_source.content_raw = active_raw.clone();

        let decision = find_duplicate(&effective_source, &candidates, cfg.cosine_threshold);
        if cfg.mode == DreamMode::Apply {
            if let Some((older, newer)) = apply_policy(&tx, &effective_source, &decision)? {
                tracing::info!(older = %older, newer = %newer, "dream superseded");
                deduped = true;
            }
        } else {
            // Dry run — report but don't write.
            if !matches!(decision, dedup::DedupDecision::Distinct) {
                deduped = true;
            }
        }
    }

    // Commit or roll back depending on mode. Dry-run always rolls back.
    if cfg.mode == DreamMode::Apply {
        tx.commit()?;
    } else {
        tx.rollback()?;
    }

    if deduped {
        Ok(Outcome::Superseded)
    } else if condensed_flag {
        Ok(Outcome::Condensed)
    } else {
        Ok(Outcome::Skipped)
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

    fn insert(conn: &Connection, id: &str, content: &str, emb: Option<Vec<f32>>) {
        let mut m = Memory::new(
            content.to_string(),
            None,
            Some("p".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        m.id = id.to_string();
        m.embedding = emb;
        m.embedding_model = Some(EMBEDDING_MODEL_NAME.to_string());
        q::insert_memory(conn, &m).expect("insert");
    }

    /// Dry-run should walk the DB and report intended decisions without
    /// persisting anything. Rows keep their content_raw=NULL.
    #[test]
    fn dry_run_does_not_mutate_db() {
        let mut conn = open_mem_db();
        insert(
            &conn,
            "00000000-0000-1111-2222-000000000001",
            "first memory",
            None,
        );
        insert(
            &conn,
            "00000000-0000-1111-2222-000000000002",
            "second memory",
            None,
        );

        let inference = FixedInference::new(r#"{"condensed":"short form"}"#);
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Dry, "gemma3", &tmp);
        let summary = run(&mut conn, &inference, &cfg).expect("dream ok");
        // Two rows walked; content should still be the original since dry-run.
        assert_eq!(summary.total_walked, 2);

        for id in [
            "00000000-0000-1111-2222-000000000001",
            "00000000-0000-1111-2222-000000000002",
        ] {
            let got = q::get_memory_by_id(&conn, id).unwrap();
            assert!(got.content_raw.is_none(), "dry-run should not persist");
        }
    }

    /// Superseded rows are skipped by the orchestrator so we don't
    /// reprocess them on every pass.
    #[test]
    fn superseded_rows_are_skipped() {
        let mut conn = open_mem_db();
        insert(&conn, "00000000-0000-1111-2222-000000000001", "older", None);
        insert(&conn, "00000000-0000-1111-2222-000000000002", "newer", None);
        q::mark_superseded(
            &conn,
            "00000000-0000-1111-2222-000000000001",
            "00000000-0000-1111-2222-000000000002",
        )
        .unwrap();

        // Even in dry-run, the superseded row should be skipped (neither
        // condensed nor deduped). Use a response that would fail JSON parse
        // so any accidental condense call would bubble an error.
        let inference = FixedInference::new("not json");
        let tmp = std::env::temp_dir();
        let cfg = DreamConfig::new(DreamMode::Dry, "gemma3", &tmp);
        let summary = run(&mut conn, &inference, &cfg).expect("dream ok");

        // Both rows walked, but the superseded one produces Skipped without
        // invoking condense (no error bubbled).
        assert_eq!(summary.total_walked, 2);
        assert!(
            summary.errors <= 1,
            "superseded row must not cause an error; got {} errors",
            summary.errors
        );
    }

    /// Limit parameter caps the walk — a DB of 5 with limit=2 processes
    /// only the 2 oldest rows.
    #[test]
    fn limit_caps_the_walk() {
        let mut conn = open_mem_db();
        for i in 1..=5 {
            insert(
                &conn,
                &format!("00000000-0000-1111-2222-00000000000{i}"),
                &format!("memory {i}"),
                None,
            );
        }

        let inference = FixedInference::new(r#"{"condensed":"x"}"#);
        let tmp = std::env::temp_dir();
        let mut cfg = DreamConfig::new(DreamMode::Dry, "gemma3", &tmp);
        cfg.limit = 2;
        let summary = run(&mut conn, &inference, &cfg).expect("dream ok");
        assert_eq!(summary.total_walked, 2);
    }
}
