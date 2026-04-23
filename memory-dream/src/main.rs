//! `memory-dream` — offline batch compactor for the agent-memory database.
//!
//! One-shot CLI. Loads the configured gemma3 model via candle, walks the
//! SQLite DB that `memory` also uses, condenses verbose rows, deduplicates
//! near-identical rows via cosine similarity, and exits. Never runs as a
//! daemon, never opens a network port, never listens for anything.
//!
//! The real logic lives in the sibling library crate (`src/lib.rs`); this
//! binary parses CLI args, wires up the DB / inference / config, and
//! dispatches into [`memory_dream::dream::run`].

use agent_memory::config::Config;
use agent_memory::db::open_database;
use agent_memory::render;
use anyhow::Context;
use clap::Parser;
use memory_dream::cli::Cli;
use memory_dream::dream::{DreamConfig, DreamMode};
use memory_dream::inference::{CandleInference, Inference, NoopInference};
use memory_dream::model_manager;
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    // Logs go to stderr (stdout is reserved for the light-XML report so a
    // caller can pipe dream output into a log collector).
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let config = Config::load().context("load config")?;
    config
        .ensure_dirs()
        .context("ensure data directory exists")?;

    // --pull runs to completion and exits — no DB work.
    if cli.pull {
        return run_pull(&cli, &config);
    }

    run_compaction(&cli, &config)
}

/// Handle `memory-dream --pull`. Downloads the model into
/// `$AGENT_MEMORY_DIR/models/<name>/`, emitting per-file progress lines
/// on stdout, then exits.
///
/// Uses a fresh tokio runtime rather than `#[tokio::main]` on the top-level
/// `fn main` because compaction itself is synchronous; spinning up the
/// runtime only for --pull keeps cold-start cost off the common path.
fn run_pull(cli: &Cli, config: &Config) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new().context("construct tokio runtime for --pull")?;
    let cache_root = &config.model_cache_dir;

    let token = model_manager::hf_token_from_env();
    let client =
        model_manager::HfHubClient::new(cache_root, token).context("construct hf-hub client")?;

    let progress = build_progress_renderer();

    match rt.block_on(model_manager::pull_model(
        &client, cache_root, &cli.model, progress,
    )) {
        Ok(report) => {
            // Final summary line; per-file lines were emitted during the
            // pull via the progress callback.
            println!(
                "{}",
                render::render_action_result(
                    if report.skipped {
                        "pull_skipped"
                    } else {
                        "pull_complete"
                    },
                    &[
                        ("model", report.model),
                        ("files", report.files.to_string()),
                        ("bytes_total", report.bytes_total.to_string()),
                        ("dest", cache_root.display().to_string()),
                    ]
                )
            );
            Ok(())
        }
        Err(e) => {
            render_pull_error(&cli.model, &e);
            Err(anyhow::anyhow!("pull failed: {e}"))
        }
    }
}

/// Build a progress callback that emits throttled light-XML lines to
/// stdout. Throttling rule: emit every ~5% OR every 16 MiB delta,
/// whichever is sparser, so multi-GB downloads don't flood the console.
/// Start and Done events always emit.
fn build_progress_renderer() -> memory_dream::model_manager::ProgressFn {
    use memory_dream::model_manager::ProgressEvent;
    use std::sync::Mutex;

    // Per-file last-emitted byte count, so throttling is computed
    // independently for each file in the pull batch.
    let last_emit: std::sync::Arc<Mutex<std::collections::HashMap<String, u64>>> =
        std::sync::Arc::new(Mutex::new(std::collections::HashMap::new()));

    const MIN_DELTA_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB
    const MIN_DELTA_PCT: u64 = 5;

    std::sync::Arc::new(move |ev| match ev {
        ProgressEvent::Start { file, bytes_total } => {
            println!(
                "{}",
                render::render_action_result(
                    "pull_file_start",
                    &[
                        ("file", file.clone()),
                        ("bytes_total", bytes_total.to_string()),
                    ]
                )
            );
            last_emit.lock().unwrap().insert(file, 0);
        }
        ProgressEvent::Update {
            file,
            bytes_done,
            bytes_total,
        } => {
            let should_emit = {
                let mut map = last_emit.lock().unwrap();
                let prev = map.get(&file).copied().unwrap_or(0);
                let byte_delta = bytes_done.saturating_sub(prev);
                let pct_delta = if bytes_total > 0 {
                    (byte_delta * 100) / bytes_total
                } else {
                    100
                };
                // Emit when BOTH thresholds pass — sparser of the two
                // wins, so tiny files (<16MiB total) emit on percent
                // and huge files emit on absolute delta.
                if byte_delta >= MIN_DELTA_BYTES && pct_delta >= MIN_DELTA_PCT {
                    map.insert(file.clone(), bytes_done);
                    true
                } else {
                    false
                }
            };
            if should_emit {
                let pct = if bytes_total > 0 {
                    (bytes_done * 100 / bytes_total).to_string()
                } else {
                    "0".to_string()
                };
                println!(
                    "{}",
                    render::render_action_result(
                        "pull_progress",
                        &[
                            ("file", file),
                            ("bytes_done", bytes_done.to_string()),
                            ("bytes_total", bytes_total.to_string()),
                            ("pct", pct),
                        ]
                    )
                );
            }
        }
        ProgressEvent::Done { file, bytes_total } => {
            println!(
                "{}",
                render::render_action_result(
                    "pull_file_done",
                    &[("file", file), ("bytes_total", bytes_total.to_string())]
                )
            );
        }
    })
}

/// Render a typed `ModelManagerError` as a structured light-XML line
/// plus, for `AuthRequired`, a human-readable multi-line hint on stderr
/// so scripts reading stdout still get the parseable status while
/// interactive users see the full remediation.
fn render_pull_error(model: &str, err: &model_manager::ModelManagerError) {
    use model_manager::ModelManagerError;
    let status = err.status_token();
    let mut attrs: Vec<(&str, String)> = vec![("model", model.to_string())];
    match err {
        ModelManagerError::AuthRequired { repo, .. } => {
            attrs.push(("repo", repo.clone()));
        }
        ModelManagerError::NotFound { repo, file } => {
            attrs.push(("repo", repo.clone()));
            attrs.push(("file", file.clone()));
        }
        ModelManagerError::NetworkError { attempts, .. } => {
            attrs.push(("attempts", attempts.to_string()));
        }
        _ => {}
    }
    attrs.push(("reason", format!("{err}")));
    println!(
        "{}",
        render::render_action_result(status, &attrs_as_refs(&attrs))
    );

    if let ModelManagerError::AuthRequired { repo, .. } = err {
        eprintln!(
            "\nError: model '{model}' ({repo}) is gated on HuggingFace.\n\
             1. Visit https://huggingface.co/{repo} and accept the license.\n\
             2. Create an access token at https://huggingface.co/settings/tokens.\n\
             3. Export HF_TOKEN=<your_token> and re-run `memory-dream --pull`."
        );
    }
}

/// Helper — `render_action_result` takes `&[(&str, String)]` and the
/// error path builds an owned `Vec<(&str, String)>` that needs to be
/// borrowed as-is. Inlined once so the call sites stay tidy.
fn attrs_as_refs<'a>(attrs: &'a [(&'a str, String)]) -> Vec<(&'a str, String)> {
    attrs.iter().map(|(k, v)| (*k, v.clone())).collect()
}

/// Handle the default compaction pass.
///
/// Opens the same SQLite DB `memory` uses, constructs the candle
/// inference backend (which may fail with `ModelMissing` when no
/// model has been pulled yet), and dispatches to `dream::run`. When
/// candle init fails, dream is still invoked in dedup-only fallback
/// mode — the orchestrator handles `InferenceFailed` per row.
fn run_compaction(cli: &Cli, config: &Config) -> anyhow::Result<()> {
    let mut conn = open_database(&config.db_path).context("open memory database")?;

    // Inference backend. A missing model directory is non-fatal here:
    // `dream::run` can still do exact-match + cosine dedup without any
    // model loaded — condense() surfaces InferenceFailed and the
    // orchestrator's fallback path keeps going. We swap in NoopInference
    // so every condense attempt deterministically fails; the orchestrator
    // then proceeds straight to dedup for each row.
    let model_dir = model_manager::resolve_model_path(&config.model_cache_dir, &cli.model);
    let inference: Box<dyn Inference> = match CandleInference::new(&model_dir) {
        Ok(i) => Box::new(i),
        Err(e) => {
            eprintln!(
                "[WARN] {e}. Falling back to dedup-only pass. Run \
                 `memory-dream --pull` to enable condensation."
            );
            Box::new(NoopInference::new(format!("{e}")))
        }
    };

    let mode = if cli.dry_run {
        DreamMode::Dry
    } else {
        DreamMode::Apply
    };

    let mut cfg = DreamConfig::new(mode, &cli.model, &config.model_cache_dir);
    cfg.limit = cli.limit;

    let summary = memory_dream::dream::run(&mut conn, inference.as_ref(), &cfg)
        .map_err(|e| anyhow::anyhow!("dream pass failed: {e}"))?;

    tracing::info!(
        walked = summary.total_walked,
        condensed = summary.condensed,
        superseded = summary.superseded,
        skipped = summary.skipped,
        errors = summary.errors,
        "dream pass finished"
    );

    Ok(())
}
