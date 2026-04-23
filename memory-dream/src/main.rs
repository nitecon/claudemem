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
/// `$AGENT_MEMORY_DIR/models/<name>/`, emits a single `<result/>` line,
/// exits.
///
/// Uses a fresh tokio runtime rather than `#[tokio::main]` on the top-level
/// `fn main` because compaction itself is synchronous; spinning up the
/// runtime only for --pull keeps cold-start cost off the common path.
fn run_pull(cli: &Cli, config: &Config) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new().context("construct tokio runtime for --pull")?;
    let cache_root = &config.model_cache_dir;

    match rt.block_on(model_manager::pull_model(cache_root, &cli.model)) {
        Ok(_) => {
            println!(
                "{}",
                render::render_action_result(
                    "pulled",
                    &[
                        ("model", cli.model.clone()),
                        ("dest", cache_root.display().to_string()),
                    ]
                )
            );
            Ok(())
        }
        Err(e) => {
            // Surface a structured `<result status="pull_failed" .../>` line
            // so scripts can parse it, then propagate the error so the
            // process exit code is non-zero.
            println!(
                "{}",
                render::render_action_result(
                    "pull_failed",
                    &[("model", cli.model.clone()), ("error", format!("{e}"))]
                )
            );
            Err(anyhow::anyhow!("pull failed: {e}"))
        }
    }
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
