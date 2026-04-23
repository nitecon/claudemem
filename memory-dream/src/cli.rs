//! clap CLI for `memory-dream`.
//!
//! Top-level command has an optional subcommand. When no subcommand is
//! given, the binary runs a dream pass using the configured settings
//! (preserving the pre-2.2 `memory-dream` / `memory-dream --pull` /
//! `memory-dream --dry-run` / `memory-dream --limit N` invocations).
//!
//! Subcommands are 2.2 additions:
//!   * `config show|set`     — inspect / mutate `dream.toml`.
//!   * `use <model>|--headless|--disabled` — flip the active backend.
//!   * `rm <model>`          — delete a local model from cache + TOML.
//!   * `list`                — dump the effective configuration.
//!   * `test <id>`           — preview condensation on a single memory.
//!
//! Flag overrides (`--backend`, `--model`, `--command`) apply to the bare
//! dream pass AND to `test`. They never mutate `dream.toml` — the
//! override flow is strictly single-invocation.

use clap::{Parser, Subcommand};

use crate::settings::BackendMode;

/// Default model short-name matched by `--pull` and stamped into
/// `condenser_version`. Mirrors `crate::model_manager::DEFAULT_MODEL`.
pub const DEFAULT_MODEL_NAME: &str = "gemma3";

/// Parsed CLI arguments for `memory-dream`.
///
/// clap's `Subcommand` derive treats `command: Option<Commands>` as "optional
/// subcommand", and the remaining top-level flags stay valid on the bare
/// invocation so `memory-dream --pull --model gemma3` keeps working.
#[derive(Debug, Parser)]
#[command(
    name = "memory-dream",
    version = env!("CARGO_PKG_VERSION"),
    about = "Offline batch compactor for agent-memory: condense, dedup, archive."
)]
pub struct Cli {
    /// Optional subcommand. When absent, the binary runs a dream pass
    /// (equivalent to `memory-dream run`).
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Model short-name (used by --pull and by the run path's local backend).
    /// When the local backend is selected, this overrides `local.active_model`
    /// from `dream.toml` for a single invocation.
    #[arg(long, global = true, default_value = DEFAULT_MODEL_NAME)]
    pub model: String,

    /// Download (or update) the configured model from HuggingFace into
    /// the local cache directory. Implies a non-run invocation — exits
    /// after the download completes.
    #[arg(long, global = false)]
    pub pull: bool,

    /// Walk the DB and report intended decisions without writing.
    #[arg(long, global = true)]
    pub dry_run: bool,

    /// Cap the number of memories processed this pass. `0` means no limit.
    #[arg(long, global = true, default_value_t = 0)]
    pub limit: usize,

    /// Override the backend for a single invocation
    /// (headless | local | disabled). Does NOT mutate `dream.toml`.
    #[arg(long, global = true, value_parser = parse_backend_mode)]
    pub backend: Option<BackendMode>,

    /// Override the headless command template for a single invocation.
    /// Example: --command "echo '{prompt}'". Does NOT mutate `dream.toml`.
    #[arg(long, global = true)]
    pub command_override: Option<String>,
}

/// Parse `--backend` values. Lives on the clap value_parser rather than
/// `BackendMode: FromStr` so the error surface formats as clap's standard
/// `invalid value` message.
fn parse_backend_mode(s: &str) -> Result<BackendMode, String> {
    s.parse::<BackendMode>()
}

/// Subcommands added in Release 2.2. The bare (None) invocation remains
/// "run a dream pass" so existing automation is untouched.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Explicit alias for the bare invocation. Useful when combining with
    /// other flags in scripts for readability.
    Run,

    /// Manage the `dream.toml` settings file.
    #[command(subcommand)]
    Config(ConfigCmd),

    /// Set the active model (also flips backend to local), or switch to a
    /// non-local backend entirely.
    Use(UseArgs),

    /// Delete a local model's cache directory and remove it from
    /// `local.downloaded_models` in `dream.toml`.
    Rm(RmArgs),

    /// Dump effective settings (backend, active model, downloaded models,
    /// PATH-detected CLIs) as light-XML on stdout.
    List,

    /// Preview condensation for a single memory. Does NOT write to the DB.
    /// Honors `--backend` / `--model` / `--command` overrides so the same
    /// memory can be A/B'd across backends.
    Test(TestArgs),
}

/// `memory-dream config ...` variants.
#[derive(Debug, Subcommand)]
pub enum ConfigCmd {
    /// Dump `dream.toml` as light-XML on stdout.
    Show,
    /// Mutate a dotted key: `config set headless.timeout_ms 60000`.
    Set(ConfigSetArgs),
}

#[derive(Debug, Parser)]
pub struct ConfigSetArgs {
    /// Dotted key path. Supported:
    ///   - backend.mode
    ///   - local.active_model
    ///   - headless.command
    ///   - headless.timeout_ms
    pub key: String,
    /// New value. Type-checked by `Settings::apply_dotted_set`.
    pub value: String,
}

/// `memory-dream use ...` — exactly one of the mutually-exclusive flags
/// or the positional `<model>` argument must be supplied.
///
/// The positional field is named `model_name` (not `model`) deliberately:
/// the top-level `--model <x>` flag on [`Cli`] is global and defaults to
/// `"gemma3"`, which would otherwise collide with the same-named positional
/// and leak the default into `UseArgs` when only `--headless` is passed.
#[derive(Debug, Parser)]
#[command(group = clap::ArgGroup::new("target").required(true).multiple(false))]
pub struct UseArgs {
    /// Set `local.active_model` to this short-name AND flip `backend.mode`
    /// to `local`. The short-name doesn't have to already be in
    /// `downloaded_models` — use this together with `--pull` for new models.
    #[arg(group = "target", value_name = "MODEL")]
    pub model_name: Option<String>,

    /// Flip backend to `headless` without touching `active_model`.
    #[arg(long, group = "target")]
    pub headless: bool,

    /// Flip backend to `disabled` (dedup-only pass).
    #[arg(long, group = "target")]
    pub disabled: bool,
}

#[derive(Debug, Parser)]
pub struct RmArgs {
    /// Short-name of the local model to remove. The cache directory
    /// `$AGENT_MEMORY_DIR/models/<name>/` is deleted and the name is
    /// removed from `downloaded_models` in `dream.toml`.
    pub model: String,
}

#[derive(Debug, Parser)]
pub struct TestArgs {
    /// Memory ID (full UUID or 8-char prefix).
    pub id_prefix: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_invocation_has_no_subcommand() {
        let cli = Cli::parse_from(["memory-dream"]);
        assert!(cli.command.is_none());
        assert_eq!(cli.model, DEFAULT_MODEL_NAME);
        assert!(!cli.pull);
        assert!(!cli.dry_run);
        assert_eq!(cli.limit, 0);
        assert!(cli.backend.is_none());
        assert!(cli.command_override.is_none());
    }

    #[test]
    fn bare_pull_flag_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--pull"]);
        assert!(cli.pull);
        assert!(cli.command.is_none());
    }

    #[test]
    fn bare_dry_run_flag_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--dry-run"]);
        assert!(cli.dry_run);
    }

    #[test]
    fn bare_limit_flag_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--limit", "42"]);
        assert_eq!(cli.limit, 42);
    }

    #[test]
    fn bare_model_override_is_parsed() {
        let cli = Cli::parse_from(["memory-dream", "--model", "myorg/custom"]);
        assert_eq!(cli.model, "myorg/custom");
    }

    #[test]
    fn backend_override_parses() {
        let cli = Cli::parse_from(["memory-dream", "--backend", "local"]);
        assert_eq!(cli.backend, Some(BackendMode::Local));
    }

    #[test]
    fn backend_override_rejects_bad_value() {
        let err = Cli::try_parse_from(["memory-dream", "--backend", "llama"]).unwrap_err();
        assert!(err.to_string().contains("invalid value"));
    }

    #[test]
    fn command_override_parses() {
        let cli = Cli::parse_from(["memory-dream", "--command-override", "echo {prompt}"]);
        assert_eq!(cli.command_override.as_deref(), Some("echo {prompt}"));
    }

    #[test]
    fn config_show_subcommand() {
        let cli = Cli::parse_from(["memory-dream", "config", "show"]);
        matches!(cli.command, Some(Commands::Config(ConfigCmd::Show)));
    }

    #[test]
    fn config_set_subcommand() {
        let cli = Cli::parse_from([
            "memory-dream",
            "config",
            "set",
            "headless.timeout_ms",
            "60000",
        ]);
        match cli.command {
            Some(Commands::Config(ConfigCmd::Set(args))) => {
                assert_eq!(args.key, "headless.timeout_ms");
                assert_eq!(args.value, "60000");
            }
            other => panic!("expected config set, got {other:?}"),
        }
    }

    #[test]
    fn use_positional_model() {
        let cli = Cli::parse_from(["memory-dream", "use", "tinyllama"]);
        match cli.command {
            Some(Commands::Use(args)) => {
                assert_eq!(args.model_name.as_deref(), Some("tinyllama"));
                assert!(!args.headless);
                assert!(!args.disabled);
            }
            other => panic!("expected use, got {other:?}"),
        }
    }

    #[test]
    fn use_headless_flag() {
        let cli = Cli::parse_from(["memory-dream", "use", "--headless"]);
        match cli.command {
            Some(Commands::Use(args)) => {
                assert!(
                    args.model_name.is_none(),
                    "model_name should be None, got {:?}",
                    args.model_name
                );
                assert!(args.headless);
                assert!(!args.disabled);
            }
            other => panic!("expected use, got {other:?}"),
        }
    }

    #[test]
    fn use_disabled_flag() {
        let cli = Cli::parse_from(["memory-dream", "use", "--disabled"]);
        match cli.command {
            Some(Commands::Use(args)) => {
                assert!(args.disabled);
            }
            other => panic!("expected use, got {other:?}"),
        }
    }

    #[test]
    fn use_requires_a_target() {
        // Empty `use` must fail — the ArgGroup enforces one of the three.
        let err = Cli::try_parse_from(["memory-dream", "use"]).unwrap_err();
        assert!(err.to_string().contains("required"));
    }

    #[test]
    fn rm_requires_model() {
        let cli = Cli::parse_from(["memory-dream", "rm", "gemma3"]);
        match cli.command {
            Some(Commands::Rm(args)) => assert_eq!(args.model, "gemma3"),
            other => panic!("expected rm, got {other:?}"),
        }
    }

    #[test]
    fn list_subcommand_parses() {
        let cli = Cli::parse_from(["memory-dream", "list"]);
        matches!(cli.command, Some(Commands::List));
    }

    #[test]
    fn test_subcommand_parses() {
        let cli = Cli::parse_from(["memory-dream", "test", "4c82c482"]);
        match cli.command {
            Some(Commands::Test(args)) => assert_eq!(args.id_prefix, "4c82c482"),
            other => panic!("expected test, got {other:?}"),
        }
    }

    #[test]
    fn run_subcommand_alias_parses() {
        let cli = Cli::parse_from(["memory-dream", "run"]);
        matches!(cli.command, Some(Commands::Run));
    }

    #[test]
    fn test_with_backend_override() {
        let cli = Cli::parse_from(["memory-dream", "--backend", "headless", "test", "4c82c482"]);
        assert_eq!(cli.backend, Some(BackendMode::Headless));
        matches!(cli.command, Some(Commands::Test(_)));
    }
}
