//! clap CLI for `memory-dream`.
//!
//! One top-level command, no subcommands — `memory-dream` is a one-shot
//! utility. Flags control execution mode; bare invocation runs a full
//! compaction pass in `Apply` mode.
//!
//! Usage:
//!   memory-dream                    # run a full pass
//!   memory-dream --dry-run          # walk + report, no writes
//!   memory-dream --limit N          # cap the pass at N memories
//!   memory-dream --model <name>     # override the default (gemma3)
//!   memory-dream --pull             # download / update the model

use clap::Parser;

/// Default model short-name matched by `--pull` and stamped into
/// `condenser_version`. Mirrors `crate::model_manager::DEFAULT_MODEL`.
pub const DEFAULT_MODEL_NAME: &str = "gemma3";

/// Parsed CLI arguments.
#[derive(Debug, Parser)]
#[command(
    name = "memory-dream",
    version = env!("CARGO_PKG_VERSION"),
    about = "Offline batch compactor for agent-memory: condense, dedup, archive."
)]
pub struct Cli {
    /// Model short-name to use for condensation (default: gemma3).
    /// Maps through `model_manager::resolve_repo_id` for download.
    #[arg(long, default_value = DEFAULT_MODEL_NAME)]
    pub model: String,

    /// Download (or update) the configured model from HuggingFace into
    /// the local cache directory. Runs to completion and exits without
    /// performing a compaction pass.
    #[arg(long)]
    pub pull: bool,

    /// Walk the DB and report intended decisions, but do not persist
    /// any changes. Row counts remain unchanged afterwards.
    #[arg(long)]
    pub dry_run: bool,

    /// Cap the number of memories processed this pass. `0` (the default)
    /// means no limit.
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_sensible() {
        let cli = Cli::parse_from(["memory-dream"]);
        assert_eq!(cli.model, DEFAULT_MODEL_NAME);
        assert!(!cli.pull);
        assert!(!cli.dry_run);
        assert_eq!(cli.limit, 0);
    }

    #[test]
    fn dry_run_flag_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--dry-run"]);
        assert!(cli.dry_run);
    }

    #[test]
    fn limit_flag_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--limit", "42"]);
        assert_eq!(cli.limit, 42);
    }

    #[test]
    fn pull_flag_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--pull"]);
        assert!(cli.pull);
    }

    #[test]
    fn model_override_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--model", "myorg/custom"]);
        assert_eq!(cli.model, "myorg/custom");
    }
}
