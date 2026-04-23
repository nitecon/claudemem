//! `memory-dream` — offline batch compactor for the agent-memory database.
//!
//! One-shot CLI. Dispatches into three execution paths:
//!   1. `--pull`                     → download model weights, exit.
//!   2. One of the Release 2.2 subcommands (`config`/`use`/`rm`/`list`/`test`).
//!   3. Bare invocation (or `run`)   → open the DB, run a dream pass.
//!
//! Loads `dream.toml` eagerly so first-run auto-detect fires regardless of
//! which path the user takes. CLI flags (`--backend` / `--model` /
//! `--command-override`) layer over the persisted settings *without*
//! mutating them — mutation is reserved for the `config set` / `use` /
//! `rm` subcommands and for `--pull`'s success path.

use agent_memory::config::Config;
use agent_memory::db::open_database;
use agent_memory::render;
use anyhow::Context;
use clap::Parser;
use memory_dream::cli::{Cli, Commands, ConfigCmd, ConfigSetArgs, RmArgs, TestArgs, UseArgs};
use memory_dream::dream::{DreamConfig, DreamMode};
use memory_dream::inference::{
    CandleInference, DevicePreference, HeadlessInference, Inference, NoopInference,
};
use memory_dream::model_manager;
use memory_dream::settings::{BackendMode, CliOverrides, Settings};
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

    // Load settings eagerly so first-run auto-detect fires on every entry
    // point. Subcommands that only read settings (list, config show) still
    // materialize `dream.toml` on their first invocation — consistent with
    // the rest of the UX.
    let mut settings = Settings::load(&config.data_dir).context("load dream.toml settings")?;

    // Subcommands take precedence over the bare-invocation flags. --pull
    // remains a top-level flag because it predates the subcommand split.
    if cli.pull {
        return run_pull(&cli, &config, &mut settings);
    }

    match &cli.command {
        None | Some(Commands::Run) => run_compaction(&cli, &config, &settings),
        Some(Commands::Config(sub)) => run_config(&config, &mut settings, sub),
        Some(Commands::Use(args)) => run_use(&config, &mut settings, args),
        Some(Commands::Rm(args)) => run_rm(&config, &mut settings, args),
        Some(Commands::List) => run_list(&config, &settings),
        Some(Commands::Test(args)) => run_test(&cli, &config, &settings, args),
    }
}

// ---------------------------------------------------------------------------
// --pull
// ---------------------------------------------------------------------------

/// Handle `memory-dream --pull`. Downloads the model into
/// `$AGENT_MEMORY_DIR/models/<name>/`, appends to `downloaded_models` in
/// `dream.toml` on success, emits per-file progress lines on stdout, then
/// exits.
///
/// Uses a fresh tokio runtime rather than `#[tokio::main]` on the top-level
/// `fn main` because compaction itself is synchronous; spinning up the
/// runtime only for --pull keeps cold-start cost off the common path.
fn run_pull(cli: &Cli, config: &Config, settings: &mut Settings) -> anyhow::Result<()> {
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
            // Persist the short-name in dream.toml so `list` / `use` / `rm`
            // can round-trip it. Skipped-idempotent pulls still register
            // the short-name — the cache is present, the TOML should agree.
            settings.add_downloaded_model(&cli.model);
            if let Err(e) = settings.save(&config.data_dir) {
                eprintln!("[WARN] model downloaded but failed to persist settings: {e}");
            }

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
    println!("{}", render::render_action_result(status, &attrs));

    if let ModelManagerError::AuthRequired { repo, .. } = err {
        eprintln!(
            "\nError: model '{model}' ({repo}) is gated on HuggingFace.\n\
             1. Visit https://huggingface.co/{repo} and accept the license.\n\
             2. Create an access token at https://huggingface.co/settings/tokens.\n\
             3. Export HF_TOKEN=<your_token> and re-run `memory-dream --pull`."
        );
    }
}

// ---------------------------------------------------------------------------
// Dream pass (bare invocation / `run`)
// ---------------------------------------------------------------------------

/// Handle the default compaction pass.
///
/// Opens the same SQLite DB `memory` uses, resolves the effective backend
/// (CLI flags over file settings), constructs the inference impl, and
/// dispatches to `dream::run`.
fn run_compaction(cli: &Cli, config: &Config, settings: &Settings) -> anyhow::Result<()> {
    let mut conn = open_database(&config.db_path).context("open memory database")?;

    let overrides = cli_overrides(cli);
    let effective = settings.effective(&overrides);
    let inference = build_inference(&effective, &config.model_cache_dir);

    let mode = if cli.dry_run {
        DreamMode::Dry
    } else {
        DreamMode::Apply
    };

    let mut cfg = DreamConfig::new(mode, &effective.active_model, &config.model_cache_dir);
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

/// Assemble a `CliOverrides` from clap-parsed flags. Single-entry helper
/// kept here rather than on the `Cli` struct so the lib crate doesn't
/// need to know about `BackendMode -> EffectiveConfig`.
fn cli_overrides(cli: &Cli) -> CliOverrides {
    CliOverrides {
        backend: cli.backend,
        // `--model` has a default value baked in (`gemma3`) so we only
        // surface it as an override when the user explicitly set it
        // *different* from that default. Otherwise we fall through to
        // `local.active_model` from dream.toml. Ideally clap would
        // report "was this flag present" but the default-value pattern
        // doesn't give us that signal; the pragmatic workaround is that
        // passing --model gemma3 and not passing --model both produce
        // the same behavior (use gemma3 as the active local model).
        model: if cli.model == memory_dream::cli::DEFAULT_MODEL_NAME {
            None
        } else {
            Some(cli.model.clone())
        },
        command: cli.command_override.clone(),
    }
}

/// Build an [`Inference`] implementation matching the effective backend
/// mode. Failures (missing model, bad template) fall back to
/// [`NoopInference`] so the dream orchestrator can still run dedup-only.
fn build_inference(
    effective: &memory_dream::settings::EffectiveConfig,
    model_cache_dir: &std::path::Path,
) -> Box<dyn Inference> {
    match effective.mode {
        BackendMode::Headless => {
            match HeadlessInference::new(&effective.headless_command, effective.headless_timeout_ms)
            {
                Ok(h) => Box::new(h),
                Err(e) => {
                    eprintln!(
                        "[WARN] headless backend unavailable: {e}. Falling back to \
                         dedup-only pass."
                    );
                    Box::new(NoopInference::new(format!("{e}")))
                }
            }
        }
        BackendMode::Local => {
            let model_dir =
                model_manager::resolve_model_path(model_cache_dir, &effective.active_model);
            // Parse the device preference here so misconfigured dream.toml
            // surfaces a clean error before we try to mmap weights. A bad
            // string falls back to `auto` so a corrupted file doesn't halt
            // the pass outright — the warning is enough.
            let pref = match effective.device.parse::<DevicePreference>() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "[WARN] invalid local.device {:?} in dream.toml ({e}); \
                         falling back to 'auto'.",
                        effective.device
                    );
                    DevicePreference::Auto
                }
            };
            match CandleInference::new(&model_dir, pref) {
                Ok(i) => Box::new(i),
                Err(e) => {
                    eprintln!(
                        "[WARN] {e}. Falling back to dedup-only pass. Run \
                         `memory-dream --pull` to enable condensation."
                    );
                    Box::new(NoopInference::new(format!("{e}")))
                }
            }
        }
        BackendMode::Disabled => Box::new(NoopInference::new("backend disabled in dream.toml")),
    }
}

// ---------------------------------------------------------------------------
// config subcommands
// ---------------------------------------------------------------------------

/// `memory-dream config show | set`.
///
/// `show` renders the current `dream.toml` contents as a light-XML
/// `<settings>` block. `set` mutates a dotted key and persists atomically.
fn run_config(config: &Config, settings: &mut Settings, sub: &ConfigCmd) -> anyhow::Result<()> {
    match sub {
        ConfigCmd::Show => {
            println!("{}", render_settings(settings));
            Ok(())
        }
        ConfigCmd::Set(ConfigSetArgs { key, value }) => {
            settings
                .apply_dotted_set(key, value)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            settings.save(&config.data_dir).context("save dream.toml")?;
            println!(
                "{}",
                render::render_action_result(
                    "config_set",
                    &[("key", key.clone()), ("value", value.clone())]
                )
            );
            Ok(())
        }
    }
}

/// Render the full settings document as a light-XML `<settings>` block.
///
/// We intentionally do NOT just dump raw TOML — the rest of `memory-dream`
/// output is light-XML and scripts parsing stdout benefit from consistency.
/// The shape mirrors `dream.toml` 1:1 so humans reading either format see
/// the same keys.
fn render_settings(s: &Settings) -> String {
    let mut out = String::new();
    out.push_str("<settings>\n");
    out.push_str(&format!(
        "  <backend mode=\"{}\"/>\n",
        s.backend.mode.as_str()
    ));
    out.push_str(&format!(
        "  <local active_model=\"{}\" downloaded_models=\"{}\"/>\n",
        escape_attr(&s.local.active_model),
        escape_attr(&s.local.downloaded_models.join(","))
    ));
    out.push_str(&format!(
        "  <headless command=\"{}\" timeout_ms=\"{}\"/>\n",
        escape_attr(&s.headless.command),
        s.headless.timeout_ms
    ));
    out.push_str("</settings>");
    out
}

/// Escape a string for use inside a `"..."` attribute. Mirrors the minimal
/// escape policy in `agent_memory::render` — only `"` must be neutralized
/// so the attribute delimiter survives literal quotes in command templates.
fn escape_attr(s: &str) -> String {
    if s.contains('"') {
        s.replace('"', "&quot;")
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// use / rm / list
// ---------------------------------------------------------------------------

/// `memory-dream use <model> | --headless | --disabled`.
///
/// Exactly one of the mutually-exclusive targets is present (enforced by
/// clap's `ArgGroup` in `UseArgs`). The positional form sets
/// `local.active_model` AND flips `backend.mode = local` — the common path
/// when the user explicitly wants to condense through a local model.
fn run_use(config: &Config, settings: &mut Settings, args: &UseArgs) -> anyhow::Result<()> {
    if let Some(name) = &args.model_name {
        settings.local.active_model = name.clone();
        settings.backend.mode = BackendMode::Local;
        settings.save(&config.data_dir).context("save dream.toml")?;
        if !settings.local.downloaded_models.iter().any(|m| m == name) {
            eprintln!(
                "[WARN] '{name}' is not in local.downloaded_models; run \
                 `memory-dream --pull --model {name}` before the next dream pass."
            );
        }
        println!(
            "{}",
            render::render_action_result(
                "use",
                &[
                    ("backend", BackendMode::Local.as_str().to_string()),
                    ("model", name.clone()),
                ]
            )
        );
    } else if args.headless {
        settings.backend.mode = BackendMode::Headless;
        settings.save(&config.data_dir).context("save dream.toml")?;
        println!(
            "{}",
            render::render_action_result(
                "use",
                &[("backend", BackendMode::Headless.as_str().to_string())]
            )
        );
    } else if args.disabled {
        settings.backend.mode = BackendMode::Disabled;
        settings.save(&config.data_dir).context("save dream.toml")?;
        println!(
            "{}",
            render::render_action_result(
                "use",
                &[("backend", BackendMode::Disabled.as_str().to_string())]
            )
        );
    } else {
        // clap enforces the ArgGroup so this branch is unreachable in
        // practice; defend anyway.
        return Err(anyhow::anyhow!(
            "internal: use called with no target (clap ArgGroup should prevent this)"
        ));
    }
    Ok(())
}

/// `memory-dream rm <model>`. Deletes the cache directory and removes the
/// short-name from `local.downloaded_models`. Non-destructive when the
/// directory is already gone (e.g. user deleted it manually) — just prunes
/// the TOML entry.
fn run_rm(config: &Config, settings: &mut Settings, args: &RmArgs) -> anyhow::Result<()> {
    let model_dir = model_manager::resolve_model_path(&config.model_cache_dir, &args.model);
    let mut actions: Vec<&str> = Vec::new();

    if model_dir.exists() {
        std::fs::remove_dir_all(&model_dir)
            .with_context(|| format!("remove model dir {}", model_dir.display()))?;
        actions.push("cache_deleted");
    } else {
        actions.push("cache_absent");
    }

    let removed = settings.remove_downloaded_model(&args.model);
    if removed {
        settings.save(&config.data_dir).context("save dream.toml")?;
        actions.push("toml_updated");
    } else {
        actions.push("toml_no_entry");
    }

    println!(
        "{}",
        render::render_action_result(
            "rm",
            &[
                ("model", args.model.clone()),
                ("actions", actions.join(",")),
            ]
        )
    );

    // If the user just removed the currently-active model, nudge them
    // toward a follow-up. We don't auto-switch — silent backend flips
    // create confusing state.
    if settings.local.active_model == args.model {
        eprintln!(
            "[WARN] removed the currently-active local model '{}'. \
             Set a new one with `memory-dream use <model>` or switch to headless.",
            args.model
        );
    }
    Ok(())
}

/// `memory-dream list` — dump the effective configuration + PATH-detection
/// status so the user can see at a glance what the next dream pass will do.
fn run_list(_config: &Config, settings: &Settings) -> anyhow::Result<()> {
    let claude_on_path = which::which("claude").is_ok();
    let gemini_on_path = which::which("gemini").is_ok();

    let mut out = String::new();
    out.push_str("<list>\n");
    out.push_str(&format!(
        "  <backend mode=\"{}\"/>\n",
        settings.backend.mode.as_str()
    ));
    out.push_str(&format!(
        "  <local active_model=\"{}\" downloaded_models=\"{}\"/>\n",
        escape_attr(&settings.local.active_model),
        escape_attr(&settings.local.downloaded_models.join(","))
    ));
    out.push_str(&format!(
        "  <headless command=\"{}\" timeout_ms=\"{}\"/>\n",
        escape_attr(&settings.headless.command),
        settings.headless.timeout_ms
    ));
    out.push_str(&format!(
        "  <path_probe claude=\"{}\" gemini=\"{}\"/>\n",
        claude_on_path, gemini_on_path
    ));
    out.push_str("</list>");
    println!("{out}");
    Ok(())
}

// ---------------------------------------------------------------------------
// test
// ---------------------------------------------------------------------------

/// `memory-dream test <id-prefix>` — condense a single memory through the
/// effective backend and print the before/after block. Writes nothing to
/// the DB.
fn run_test(
    cli: &Cli,
    config: &Config,
    settings: &Settings,
    args: &TestArgs,
) -> anyhow::Result<()> {
    let conn = open_database(&config.db_path).context("open memory database")?;

    let resolved = agent_memory::db::queries::resolve_id_prefix(&conn, &args.id_prefix)
        .context("resolve memory id prefix")?;

    let memory = match resolved {
        agent_memory::db::queries::ResolvedId::Exact(id) => {
            agent_memory::db::queries::get_memory_by_id(&conn, &id)
                .context("fetch resolved memory")?
        }
        agent_memory::db::queries::ResolvedId::Ambiguous(candidates) => {
            println!("{}", render::render_ambiguous(&args.id_prefix, &candidates));
            return Ok(());
        }
        agent_memory::db::queries::ResolvedId::NotFound => {
            println!(
                "{}",
                render::render_action_result("not_found", &[("id_prefix", args.id_prefix.clone())])
            );
            return Ok(());
        }
    };

    let overrides = cli_overrides(cli);
    let effective = settings.effective(&overrides);
    let inference = build_inference(&effective, &config.model_cache_dir);

    // `condense` returns either a `Condensed` or a typed fallback reason.
    // For `test` we want to surface both outcomes cleanly — no silent
    // fallback to raw (that behavior belongs to the real dream pass).
    let original = memory.content.clone();
    match memory_dream::dream::condense::condense(
        inference.as_ref(),
        &effective.active_model,
        &original,
    ) {
        Ok(out) => {
            let ratio_pct = if original.chars().count() == 0 {
                0
            } else {
                (out.text.chars().count() * 100) / original.chars().count()
            };
            let backend_name = effective.mode.as_str();
            let model_label = match effective.mode {
                BackendMode::Headless => {
                    // Best-effort: use argv[0] of the command template as
                    // the "model" label so the output still reads naturally
                    // (`model="claude"`). Falls back to the short-name.
                    shlex::split(&effective.headless_command)
                        .and_then(|v| v.into_iter().next())
                        .map(|bin| {
                            std::path::Path::new(&bin)
                                .file_name()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or(bin)
                        })
                        .unwrap_or_else(|| effective.active_model.clone())
                }
                _ => effective.active_model.clone(),
            };
            println!(
                "<test_result memory_id=\"{id}\" backend=\"{backend}\" model=\"{model}\">\n\
                   <original bytes=\"{orig_bytes}\">{orig}</original>\n\
                   <condensed bytes=\"{cond_bytes}\">{cond}</condensed>\n\
                   <ratio pct=\"{pct}\"/>\n\
                 </test_result>",
                id = agent_memory::render::short_id(&memory.id),
                backend = backend_name,
                model = escape_attr(&model_label),
                orig_bytes = original.len(),
                orig = original,
                cond_bytes = out.text.len(),
                cond = out.text,
                pct = ratio_pct,
            );
        }
        Err(e) => {
            println!(
                "{}",
                render::render_action_result(
                    "test_failed",
                    &[
                        ("id", agent_memory::render::short_id(&memory.id).to_string()),
                        ("backend", effective.mode.as_str().to_string()),
                        ("reason", format!("{e}")),
                    ]
                )
            );
        }
    }

    Ok(())
}
