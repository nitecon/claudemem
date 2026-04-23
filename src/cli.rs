use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use rusqlite::Connection;
use serde_json::{json, Value};

use crate::config::Config;
use crate::db::models::Memory;
use crate::db::queries;
use crate::embedding;
use crate::error::MemoryError;
use crate::project;
use crate::search::{self, SearchOptions, SearchResult};
use crate::setup::{menu, rules, skill};

/// Score multiplier applied to memories tagged with the current project.
/// Strong cross-project matches can still out-rank weak current-project hits;
/// the boost tilts ties toward local context without hard-filtering prior art.
pub const PROJECT_BOOST: f32 = 1.5;
/// Score multiplier applied to memories tagged with the global-scope sentinel
/// project. Intentionally smaller than `PROJECT_BOOST` so universal user
/// preferences surface in every repo while still losing ties to strong local
/// context.
pub const GLOBAL_BOOST: f32 = 1.25;
/// Reserved `project` ident for global-scoped memories. Users interact with
/// this value via `--scope global`, never by name. Guarded at the store/move
/// boundaries to prevent accidental collisions (e.g. a repo literally named
/// `__global__`).
pub const GLOBAL_PROJECT_IDENT: &str = "__global__";
pub const DEFAULT_PREVIEW_CHARS: usize = 160;

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    /// Compact: id, tags, project, quality flags, and a content preview.
    Brief,
    /// Full content plus all metadata.
    Full,
}

/// Logical scope for a new memory. Projects receive the 1.5× current-project
/// boost during retrieval from the matching cwd; `Global` memories are
/// stored under the reserved sentinel project and receive a 1.25× boost from
/// every cwd so universal preferences surface across all repos.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum MemoryScope {
    /// Scoped to the current project (cwd-derived ident, overridable via
    /// `--project`). This is the default.
    Project,
    /// Universal — applies across every repo. Stored under the reserved
    /// sentinel project ident and boosted in every `context`/`search` call.
    Global,
}

#[derive(Parser)]
#[command(name = "memory", about = "Persistent hybrid-search memory system for AI coding agents", version = env!("AGENT_MEMORY_VERSION"))]
pub enum Cli {
    /// Save a memory with auto-embedding and BM25 indexing.
    ///
    /// If --project is omitted, the current project is auto-detected from the
    /// working directory's git remote or canonical path. Use `--scope global`
    /// to store a universal preference that applies across every repo.
    Store {
        /// Memory content text.
        content: String,
        /// Comma-separated tags.
        #[arg(short, long)]
        tags: Option<String>,
        /// Project identifier (defaults to cwd-derived ident).
        #[arg(short, long)]
        project: Option<String>,
        /// Agent identifier.
        #[arg(short, long)]
        agent: Option<String>,
        /// Source file path.
        #[arg(short = 'f', long)]
        source_file: Option<String>,
        /// Memory type: user, feedback, project, reference.
        #[arg(short = 'm', long, default_value = "user")]
        memory_type: String,
        /// Store without any project tag (skips cwd auto-detection).
        #[arg(long)]
        no_project: bool,
        /// Scope for the memory. `project` (default) stores under the cwd
        /// ident; `global` stores under the reserved sentinel so the memory
        /// is boosted across every repo. Left as `Option` (no default) so
        /// the dispatch layer can tell whether the user chose deliberately
        /// and only emit reflection hints when they didn't.
        #[arg(long, value_enum)]
        scope: Option<MemoryScope>,
    },
    /// Hybrid BM25 + vector search with current-project boost.
    ///
    /// By default, the current project (from cwd) is boosted but cross-project
    /// results can still surface. Use --only to hard-filter, or
    /// --no-project-boost for flat ranking.
    Search {
        /// Search query.
        query: String,
        /// Number of results.
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,
        /// Project to boost (defaults to cwd-derived ident).
        #[arg(short, long)]
        project: Option<String>,
        /// Hard filter: only return memories from this project.
        #[arg(long)]
        only: Option<String>,
        /// Disable the current-project boost entirely.
        #[arg(long)]
        no_project_boost: bool,
        /// Output format (default: brief).
        #[arg(long, value_enum, default_value_t = OutputFormat::Brief)]
        format: OutputFormat,
        /// Preview length for brief output.
        #[arg(long, default_value_t = DEFAULT_PREVIEW_CHARS)]
        preview_chars: usize,
    },
    /// Filter memories by project/agent/tags.
    Recall {
        /// Filter by project.
        #[arg(short, long)]
        project: Option<String>,
        /// Filter by agent.
        #[arg(short, long)]
        agent: Option<String>,
        /// Comma-separated tags to filter by.
        #[arg(short, long)]
        tags: Option<String>,
        /// Filter by memory type.
        #[arg(short = 'm', long)]
        memory_type: Option<String>,
        /// Number of results.
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,
        /// Output format (default: brief).
        #[arg(long, value_enum, default_value_t = OutputFormat::Brief)]
        format: OutputFormat,
        /// Preview length for brief output.
        #[arg(long, default_value_t = DEFAULT_PREVIEW_CHARS)]
        preview_chars: usize,
    },
    /// Remove memories by ID or search.
    Forget {
        /// Memory ID to remove.
        #[arg(short, long)]
        id: Option<String>,
        /// Search query to find and remove memories.
        #[arg(short, long)]
        query: Option<String>,
    },
    /// Decay stale/low-access memories.
    Prune {
        /// Maximum age in days before pruning.
        #[arg(short, long, default_value = "90")]
        max_age_days: u64,
        /// Minimum access count to keep.
        #[arg(short = 'c', long, default_value = "0")]
        min_access_count: i64,
        /// Show what would be pruned without deleting.
        #[arg(long)]
        dry_run: bool,
    },
    /// Return top-K relevant memories for a task, with current-project boost.
    Context {
        /// Task description.
        description: String,
        /// Number of results.
        #[arg(short = 'k', long, default_value = "5")]
        limit: usize,
        /// Project to boost (defaults to cwd-derived ident).
        #[arg(short, long)]
        project: Option<String>,
        /// Hard filter: only return memories from this project.
        #[arg(long)]
        only: Option<String>,
        /// Disable the current-project boost entirely.
        #[arg(long)]
        no_project_boost: bool,
        /// Output format (default: brief).
        #[arg(long, value_enum, default_value_t = OutputFormat::Brief)]
        format: OutputFormat,
        /// Preview length for brief output.
        #[arg(long, default_value_t = DEFAULT_PREVIEW_CHARS)]
        preview_chars: usize,
    },
    /// Fetch full content for one or more memory IDs.
    ///
    /// Pair with `memory search --brief` for a cheap two-stage flow: scan
    /// lightweight hits, then pull the full content of the handful you want.
    Get {
        /// Memory IDs to fetch.
        #[arg(required = true)]
        ids: Vec<String>,
        /// Output format (default: full).
        #[arg(long, value_enum, default_value_t = OutputFormat::Full)]
        format: OutputFormat,
        /// Preview length when --format brief is used.
        #[arg(long, default_value_t = DEFAULT_PREVIEW_CHARS)]
        preview_chars: usize,
    },
    /// List all stored memories.
    List {
        /// Number of results.
        #[arg(short = 'k', long, default_value = "50")]
        limit: usize,
        /// Filter by project.
        #[arg(short, long)]
        project: Option<String>,
        /// Filter by memory type.
        #[arg(short = 'm', long)]
        memory_type: Option<String>,
        /// Output format (default: brief).
        #[arg(long, value_enum, default_value_t = OutputFormat::Brief)]
        format: OutputFormat,
        /// Preview length for brief output.
        #[arg(long, default_value_t = DEFAULT_PREVIEW_CHARS)]
        preview_chars: usize,
    },
    /// Reassign the `project` ident on one or more memories.
    ///
    /// Common use: migrate memories that were tagged under a legacy project
    /// name (e.g. `trading-platform-sre`) to the canonical git-remote ident
    /// (e.g. `github.com/nitecon/SRE.git`) the cwd-resolver returns now.
    ///
    /// Selectors:
    ///   --id <ID>        move a single memory
    ///   --from <PROJ>    move every memory currently tagged with <PROJ>
    ///                    (pass `--from ""` to target memories with no project)
    ///
    /// Target:
    ///   --to <PROJ>      new project ident (pass `--to ""` to clear)
    Move {
        /// Move a single memory by ID.
        #[arg(long, conflicts_with = "from")]
        id: Option<String>,
        /// Move all memories whose current project equals this value.
        /// Use an empty string ("") to target memories with no project.
        #[arg(long)]
        from: Option<String>,
        /// New project ident. Use an empty string ("") to clear the project tag.
        #[arg(long)]
        to: String,
        /// Show the memories that would be moved without writing.
        #[arg(long)]
        dry_run: bool,
    },
    /// Duplicate one or more memories under a new project ident.
    ///
    /// Content, tags, agent, source_file, memory_type, and the cached
    /// embedding are all preserved on the copy. A new UUID is minted and
    /// timestamps reset; the source row is left untouched.
    ///
    /// Selectors mirror `memory move`:
    ///   --id <ID>        copy a single memory
    ///   --from <PROJ>    copy every memory currently tagged with <PROJ>
    Copy {
        /// Copy a single memory by ID.
        #[arg(long, conflicts_with = "from")]
        id: Option<String>,
        /// Copy all memories whose current project equals this value.
        /// Use an empty string ("") to target memories with no project.
        #[arg(long)]
        from: Option<String>,
        /// New project ident for the copies. Use an empty string ("") to
        /// create copies with no project tag.
        #[arg(long)]
        to: String,
        /// Show the memories that would be copied without writing.
        #[arg(long)]
        dry_run: bool,
    },
    /// List distinct project idents with memory counts.
    ///
    /// Useful for spotting alias mismatches (e.g. `trading-platform-sre` vs
    /// `github.com/nitecon/SRE.git`) before running `memory move --from … --to …`.
    Projects,
    /// Start MCP stdio server.
    Serve,
    /// Check for updates and install the latest version.
    Update,
    /// Setup and configuration commands (run with no subcommand for an
    /// interactive checklist of available components).
    Setup {
        #[command(subcommand)]
        command: Option<SetupCommands>,
    },
}

#[derive(Subcommand)]
pub enum SetupCommands {
    /// Inject the memory usage protocols into known agent rule files.
    ///
    /// Detects `~/.claude/CLAUDE.md`, `~/.gemini/GEMINI.md`,
    /// `~/.codex/AGENTS.md`, and `~/.config/codex/AGENTS.md`, then writes a
    /// `<memory-rules>…</memory-rules>` block so the agent knows how to call
    /// the `memory` CLI. Idempotent — re-runs replace the existing block in
    /// place. A `.bak` sibling is written before each modification.
    ///
    /// If an `<agent-tools-rules>` block is already present (written by the
    /// sibling `agent-tools setup rules` command), the memory block is
    /// inserted directly after it; otherwise it is prepended.
    Rules {
        /// Update a specific file instead of running detection.
        #[arg(long)]
        target: Option<PathBuf>,
        /// Update every detected file without prompting.
        #[arg(long)]
        all: bool,
        /// Show the resulting file content without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Print the rules block to stdout and exit (no file IO).
        #[arg(long)]
        print: bool,
    },

    /// Install a Claude Code skill at `~/.claude/skills/agent-memory/SKILL.md`
    /// so the `memory` CLI is auto-advertised to sessions via the always-loaded
    /// skill description (~100 tokens). The full body only loads on demand.
    Skill {
        /// Show the resulting file content without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Print the SKILL.md to stdout and exit (no file IO).
        #[arg(long)]
        print: bool,
    },

    /// Run rules → skill non-interactively.
    All {
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

pub fn execute(cmd: Cli, config: Config, conn: &Connection) -> Result<(), MemoryError> {
    let cwd_project = project::project_ident_from_cwd().ok();

    match cmd {
        Cli::Store {
            content,
            tags,
            project,
            agent,
            source_file,
            memory_type,
            no_project,
            scope,
        } => {
            let tag_list = tags.map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

            // Reject accidental sentinel collision on --project before we
            // resolve the scope. Users who want global scope must say so
            // explicitly via `--scope global` so the intent is recorded in
            // shell history rather than hiding behind a suspicious string.
            if matches!(scope, Some(MemoryScope::Project) | None)
                && project.as_deref() == Some(GLOBAL_PROJECT_IDENT)
            {
                return Err(MemoryError::Config(format!(
                    "`{GLOBAL_PROJECT_IDENT}` is reserved for global-scoped memories. \
                     Use `--scope global` instead."
                )));
            }

            // Normalize scope: None means "user didn't pass --scope" — treat
            // it as project-scoped for storage, but remember the fact so the
            // reflection hint below can nudge them.
            let scope_explicit = scope.is_some();
            let resolved_scope = scope.unwrap_or(MemoryScope::Project);

            let resolved_project = match resolved_scope {
                MemoryScope::Global => {
                    // --scope global forces the sentinel regardless of any
                    // --project override or cwd auto-detect. --no-project is
                    // incompatible with global scope (a global memory *must*
                    // live under the sentinel).
                    if no_project {
                        return Err(MemoryError::Config(
                            "--no-project is incompatible with --scope global".to_string(),
                        ));
                    }
                    Some(GLOBAL_PROJECT_IDENT.to_string())
                }
                MemoryScope::Project => {
                    if no_project {
                        None
                    } else {
                        project.or(cwd_project.clone())
                    }
                }
            };

            let mut memory = Memory::new(
                content.clone(),
                tag_list,
                resolved_project,
                agent,
                source_file,
                Some(memory_type.clone()),
            );

            let emb = embedding::embed_text(&content, &config.model_cache_dir)?;
            memory.embedding = Some(emb);

            queries::insert_memory(conn, &memory)?;

            let mut out = serde_json::Map::new();
            out.insert("status".into(), Value::String("stored".into()));
            out.insert("id".into(), Value::String(memory.id.clone()));
            out.insert(
                "project".into(),
                memory
                    .project
                    .as_deref()
                    .map(|p| Value::String(p.to_string()))
                    .unwrap_or(Value::Null),
            );
            out.insert(
                "scope".into(),
                Value::String(scope_label(resolved_scope).to_string()),
            );
            // Reflection hint: nudge the agent to reconsider scope only when
            // the memory type is most likely to be cross-cutting (user or
            // feedback) AND the user didn't pick a scope deliberately. No
            // noise on `project`/`reference` stores or when --scope was
            // passed explicitly — silence is the default.
            if !scope_explicit
                && resolved_scope == MemoryScope::Project
                && matches!(memory_type.as_str(), "user" | "feedback")
            {
                out.insert(
                    "hint".into(),
                    Value::String(store_scope_hint()),
                );
            }
            println!("{}", serde_json::to_string_pretty(&Value::Object(out))?);
        }
        Cli::Search {
            query,
            limit,
            project,
            only,
            no_project_boost,
            format,
            preview_chars,
        } => {
            let boosts =
                resolve_boosts(project.as_deref(), cwd_project.as_deref(), no_project_boost);
            let opts = SearchOptions {
                limit,
                current_project: boosts.current_project,
                boost_factor: boosts.project_boost,
                only_project: only.as_deref(),
                global_project: boosts.global_project,
                global_boost_factor: boosts.global_boost,
            };
            let results = search::hybrid_search(conn, &query, opts, &config.model_cache_dir)?;
            let output = render_ranked_output(&results, &boosts, format, preview_chars);
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Context {
            description,
            limit,
            project,
            only,
            no_project_boost,
            format,
            preview_chars,
        } => {
            let boosts =
                resolve_boosts(project.as_deref(), cwd_project.as_deref(), no_project_boost);
            let opts = SearchOptions {
                limit,
                current_project: boosts.current_project,
                boost_factor: boosts.project_boost,
                only_project: only.as_deref(),
                global_project: boosts.global_project,
                global_boost_factor: boosts.global_boost,
            };
            let results = search::hybrid_search(conn, &description, opts, &config.model_cache_dir)?;
            let output = render_ranked_output(&results, &boosts, format, preview_chars);
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Recall {
            project,
            agent,
            tags,
            memory_type,
            limit,
            format,
            preview_chars,
        } => {
            let tag_list: Option<Vec<String>> =
                tags.map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

            let memories = queries::list_memories(
                conn,
                project.as_deref(),
                agent.as_deref(),
                tag_list.as_deref(),
                memory_type.as_deref(),
                limit,
            )?;

            let output =
                render_memory_list(&memories, cwd_project.as_deref(), format, preview_chars);
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Forget { id, query } => {
            if let Some(id) = id {
                let deleted = queries::delete_memory(conn, &id)?;
                if deleted {
                    println!(r#"{{"status": "deleted", "id": "{}"}}"#, id);
                } else {
                    println!(r#"{{"status": "not_found", "id": "{}"}}"#, id);
                }
            } else if let Some(query) = query {
                let opts = SearchOptions::new(5);
                let results = search::hybrid_search(conn, &query, opts, &config.model_cache_dir)?;
                if results.is_empty() {
                    println!(r#"{{"status": "no_matches"}}"#);
                } else {
                    let mut deleted_ids = Vec::new();
                    for r in &results {
                        if queries::delete_memory(conn, &r.memory.id)? {
                            deleted_ids.push(r.memory.id.clone());
                        }
                    }
                    let output = json!({
                        "status": "deleted",
                        "count": deleted_ids.len(),
                        "ids": deleted_ids,
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
            } else {
                eprintln!("Either --id or --query must be provided");
            }
        }
        Cli::Prune {
            max_age_days,
            min_access_count,
            dry_run,
        } => {
            let pruned = queries::prune_memories(conn, max_age_days, min_access_count, dry_run)?;

            let output = json!({
                "status": if dry_run { "dry_run" } else { "pruned" },
                "count": pruned.len(),
                "memories": pruned.iter().map(|m| json!({
                    "id": m.id,
                    "content": m.content.chars().take(100).collect::<String>(),
                    "access_count": m.access_count,
                    "updated_at": m.updated_at,
                })).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Get {
            ids,
            format,
            preview_chars,
        } => {
            let mut fetched: Vec<Memory> = Vec::with_capacity(ids.len());
            let mut missing: Vec<String> = Vec::new();
            for id in &ids {
                match queries::get_memory_by_id(conn, id) {
                    Ok(m) => fetched.push(m),
                    Err(MemoryError::NotFound(_)) => missing.push(id.clone()),
                    Err(e) => return Err(e),
                }
            }
            let hit_ids: Vec<String> = fetched.iter().map(|m| m.id.clone()).collect();
            queries::increment_access(conn, &hit_ids)?;

            let results: Vec<Value> = fetched
                .iter()
                .map(|m| render_memory(m, cwd_project.as_deref(), format, preview_chars))
                .collect();
            let output = json!({
                "results": results,
                "missing": missing,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::List {
            limit,
            project,
            memory_type,
            format,
            preview_chars,
        } => {
            let memories = queries::list_memories(
                conn,
                project.as_deref(),
                None,
                None,
                memory_type.as_deref(),
                limit,
            )?;
            let output =
                render_memory_list(&memories, cwd_project.as_deref(), format, preview_chars);
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Move {
            id,
            from,
            to,
            dry_run,
        } => {
            // Guard: writing the reserved sentinel via `--to` would bypass
            // the `--scope global` contract. If a user legitimately wants to
            // promote an existing memory to global scope, the follow-up
            // `memory promote`/`memory demote` helpers (deferred) or a
            // future `--allow-sentinel` escape hatch are the right surface.
            if to == GLOBAL_PROJECT_IDENT {
                return Err(MemoryError::Config(format!(
                    "`{GLOBAL_PROJECT_IDENT}` is reserved for global-scoped memories. \
                     Use `memory store --scope global <content>` for new globals, or \
                     re-run with a different `--to` value."
                )));
            }
            let new_project = empty_to_none(&to);
            let output = run_move(conn, id.as_deref(), from.as_deref(), new_project, dry_run)?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Copy {
            id,
            from,
            to,
            dry_run,
        } => {
            let new_project = empty_to_none(&to);
            let output = run_copy(conn, id.as_deref(), from.as_deref(), new_project, dry_run)?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Projects => {
            let rows = queries::list_projects(conn)?;
            let items: Vec<Value> = rows
                .iter()
                .map(|(p, c)| {
                    let is_current = cwd_project
                        .as_deref()
                        .map(|cp| p.as_deref() == Some(cp))
                        .unwrap_or(false);
                    json!({
                        "project": p,
                        "count": c,
                        "is_current_project": is_current,
                    })
                })
                .collect();
            let mut out = serde_json::Map::new();
            out.insert("projects".into(), Value::Array(items));
            if let Some(cp) = cwd_project.as_deref() {
                out.insert("current_project".into(), Value::String(cp.to_string()));
            }
            println!("{}", serde_json::to_string_pretty(&Value::Object(out))?);
        }
        Cli::Serve => {
            unreachable!("Serve is handled in main.rs");
        }
        Cli::Update => {
            crate::updater::manual_update()?;
        }
        Cli::Setup { command } => {
            execute_setup(command).map_err(|e| MemoryError::Config(format!("{e:#}")))?;
        }
    }
    Ok(())
}

/// Dispatch the `memory setup` family. Returns `anyhow::Result` so the
/// installer subcommands can use `anyhow::Context` freely; the caller wraps
/// the error into `MemoryError::Config` for the unified CLI exit path.
fn execute_setup(command: Option<SetupCommands>) -> anyhow::Result<()> {
    match command {
        None => menu::run_interactive(),
        Some(SetupCommands::Rules {
            target,
            all,
            dry_run,
            print,
        }) => rules::run(target, all, dry_run, print),
        Some(SetupCommands::Skill { dry_run, print }) => skill::run(dry_run, print),
        Some(SetupCommands::All { yes }) => menu::run_all(yes),
    }
}

/// Resolved boost configuration for one `search`/`context` call.
///
/// The two boosts are independent: disabling the current-project boost via
/// `--no-project-boost` leaves the global boost intact so universal
/// preferences continue to surface across every repo even when local
/// boosting is off. Global scope is always on unless a future
/// `--no-global-boost` flag is added; the sentinel identifier is hardcoded
/// so the retrieval behavior is deterministic.
pub(crate) struct BoostConfig<'a> {
    pub current_project: Option<&'a str>,
    pub global_project: Option<&'a str>,
    pub project_boost: f32,
    pub global_boost: f32,
}

fn resolve_boosts<'a>(
    explicit: Option<&'a str>,
    cwd: Option<&'a str>,
    disable: bool,
) -> BoostConfig<'a> {
    if disable {
        // --no-project-boost disables BOTH boosts so `memory search --no-project-boost`
        // gives a genuinely flat ranking. Users who want "flat current, but
        // still boost global" are a sufficiently niche case that we wait
        // for them to ask before plumbing a third flag.
        BoostConfig {
            current_project: None,
            global_project: None,
            project_boost: 1.0,
            global_boost: 1.0,
        }
    } else {
        BoostConfig {
            current_project: explicit.or(cwd),
            global_project: Some(GLOBAL_PROJECT_IDENT),
            project_boost: PROJECT_BOOST,
            global_boost: GLOBAL_BOOST,
        }
    }
}

fn render_ranked_output(
    results: &[SearchResult],
    boosts: &BoostConfig<'_>,
    format: OutputFormat,
    preview_chars: usize,
) -> Value {
    let total = results.len();
    let global_scope_count = results.iter().filter(|r| r.is_global).count();
    let cross_project_count = if boosts.current_project.is_some() {
        // "cross-project" means "not current-project"; global-scope hits
        // are still counted as cross-project because they aren't tagged
        // with the current repo's ident. The hint helper splits them out
        // so the agent can reason about each class separately.
        results.iter().filter(|r| !r.is_current_project).count()
    } else {
        0
    };

    let items: Vec<Value> = results
        .iter()
        .map(|r| render_ranked(r, format, preview_chars))
        .collect();

    let mut out = serde_json::Map::new();
    out.insert("results".into(), Value::Array(items));
    if let Some(cp) = boosts.current_project {
        out.insert("current_project".into(), Value::String(cp.to_string()));
        out.insert(
            "cross_project_count".into(),
            Value::Number(cross_project_count.into()),
        );
    }
    out.insert(
        "global_scope_count".into(),
        Value::Number(global_scope_count.into()),
    );
    if let Some(hint) = results_hint(
        cross_project_count,
        global_scope_count,
        total,
        boosts.current_project,
    ) {
        out.insert("hint".into(), Value::String(hint));
    }
    Value::Object(out)
}

/// Assemble the reflection `hint` string for `context`/`search` responses.
///
/// Three independent conditions, each additive:
///
/// 1. **Cross-project results present** — original behavior: prefix the hint
///    with the cross-project ratio so the agent knows to treat them as
///    prior-art rather than direct context.
/// 2. **Zero global-scope matches in top-K** — reflection prompt nudging the
///    agent to confirm no universal user preference applies before acting.
///    Emitted whenever `current_project` is set (i.e. we're doing a scoped
///    retrieval, not a flat query), even if everything else lines up —
///    silence here is the dangerous case, not noise.
/// 3. **Global-scope matches present** — surface the count so the agent
///    treats them as directives rather than suggestions.
///
/// Returns `None` when nothing useful would be said (no current-project
/// context AND no global hits AND no cross-project hits).
fn results_hint(
    cross: usize,
    globals: usize,
    total: usize,
    current: Option<&str>,
) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(cp) = current {
        if cross > 0 {
            parts.push(if cross == total && globals == 0 {
                format!(
                    "All {total} results are from other projects (no memories tagged '{cp}' matched). Treat as prior-art or general guidance, not direct context for the current project."
                )
            } else {
                format!(
                    "{cross} of {total} results are cross-project (is_current_project=false). Use those as prior-art or general guidance, not direct context for '{cp}'. Use `memory get <id>` for full content."
                )
            });
        }
    }

    if globals > 0 {
        parts.push(format!(
            "{globals} of {total} results are global-scope preferences (apply across all projects). Treat them as directives, not suggestions."
        ));
    } else if current.is_some() {
        // Reflection prompt fires only when we would otherwise have surfaced
        // globals (i.e. during a scoped retrieval). In flat `--no-project-boost`
        // mode the global boost is off too, so the absence isn't meaningful.
        parts.push(
            "No global-scope preferences matched this task. If the user has stated a general rule relevant to this domain, it did not surface — consider asking before acting if you suspect one exists."
                .to_string(),
        );
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn scope_label(scope: MemoryScope) -> &'static str {
    match scope {
        MemoryScope::Project => "project",
        MemoryScope::Global => "global",
    }
}

fn store_scope_hint() -> String {
    "Stored as project-scoped. If this preference applies across all projects, re-run with `--scope global` — a silent mis-classification means future sessions in other projects won't see it.".to_string()
}

fn render_ranked(r: &SearchResult, format: OutputFormat, preview_chars: usize) -> Value {
    match format {
        OutputFormat::Brief => json!({
            "id": r.memory.id,
            "tags": r.memory.tags,
            "project": r.memory.project,
            "memory_type": r.memory.memory_type,
            "match_quality": r.match_quality.as_str(),
            "is_current_project": r.is_current_project,
            "preview": preview(&r.memory.content, preview_chars),
            "content_len": r.memory.content.chars().count(),
        }),
        OutputFormat::Full => json!({
            "id": r.memory.id,
            "content": r.memory.content,
            "tags": r.memory.tags,
            "project": r.memory.project,
            "agent": r.memory.agent,
            "memory_type": r.memory.memory_type,
            "match_quality": r.match_quality.as_str(),
            "is_current_project": r.is_current_project,
            "score": r.rank_info.score,
            "bm25_rank": r.rank_info.bm25_rank,
            "vector_rank": r.rank_info.vector_rank,
            "access_count": r.memory.access_count,
            "created_at": r.memory.created_at,
        }),
    }
}

fn render_memory_list(
    memories: &[Memory],
    cwd_project: Option<&str>,
    format: OutputFormat,
    preview_chars: usize,
) -> Value {
    let items: Vec<Value> = memories
        .iter()
        .map(|m| render_memory(m, cwd_project, format, preview_chars))
        .collect();
    Value::Array(items)
}

fn render_memory(
    m: &Memory,
    cwd_project: Option<&str>,
    format: OutputFormat,
    preview_chars: usize,
) -> Value {
    let is_current = cwd_project
        .map(|cp| m.project.as_deref() == Some(cp))
        .unwrap_or(false);
    match format {
        OutputFormat::Brief => json!({
            "id": m.id,
            "tags": m.tags,
            "project": m.project,
            "memory_type": m.memory_type,
            "is_current_project": is_current,
            "preview": preview(&m.content, preview_chars),
            "content_len": m.content.chars().count(),
            "access_count": m.access_count,
            "updated_at": m.updated_at,
        }),
        OutputFormat::Full => json!({
            "id": m.id,
            "content": m.content,
            "tags": m.tags,
            "project": m.project,
            "agent": m.agent,
            "memory_type": m.memory_type,
            "is_current_project": is_current,
            "access_count": m.access_count,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
        }),
    }
}

fn preview(content: &str, n: usize) -> String {
    let mut out: String = content.chars().take(n).collect();
    if content.chars().count() > n {
        out.push('…');
    }
    out
}

/// Treat empty strings as "no project" for the move/copy `--from`/`--to` flags.
/// Lets users explicitly target or assign a NULL project without a second flag.
fn empty_to_none(s: &str) -> Option<&str> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn run_move(
    conn: &Connection,
    id: Option<&str>,
    from: Option<&str>,
    to: Option<&str>,
    dry_run: bool,
) -> Result<Value, MemoryError> {
    match (id, from) {
        (Some(id), _) => {
            if dry_run {
                let mem = queries::get_memory_by_id(conn, id)?;
                Ok(json!({
                    "status": "dry_run",
                    "would_move": 1,
                    "id": mem.id,
                    "from_project": mem.project,
                    "to_project": to,
                }))
            } else {
                let changed = queries::move_memory_by_id(conn, id, to)?;
                Ok(json!({
                    "status": if changed { "moved" } else { "not_found" },
                    "id": id,
                    "to_project": to,
                }))
            }
        }
        (None, Some(from)) => {
            let from_opt = empty_to_none(from);
            if dry_run {
                let mems = queries::list_memories_by_project(conn, from_opt)?;
                Ok(json!({
                    "status": "dry_run",
                    "would_move": mems.len(),
                    "from_project": from_opt,
                    "to_project": to,
                    "ids": mems.iter().map(|m| &m.id).collect::<Vec<_>>(),
                }))
            } else {
                let count = queries::move_memories_by_project(conn, from_opt, to)?;
                Ok(json!({
                    "status": "moved",
                    "count": count,
                    "from_project": from_opt,
                    "to_project": to,
                }))
            }
        }
        (None, None) => Ok(json!({
            "status": "error",
            "message": "Either --id or --from must be provided",
        })),
    }
}

fn run_copy(
    conn: &Connection,
    id: Option<&str>,
    from: Option<&str>,
    to: Option<&str>,
    dry_run: bool,
) -> Result<Value, MemoryError> {
    match (id, from) {
        (Some(id), _) => {
            if dry_run {
                let mem = queries::get_memory_by_id(conn, id)?;
                Ok(json!({
                    "status": "dry_run",
                    "would_copy": 1,
                    "source_id": mem.id,
                    "from_project": mem.project,
                    "to_project": to,
                }))
            } else {
                let new_id = queries::copy_memory_by_id(conn, id, to)?;
                Ok(json!({
                    "status": "copied",
                    "source_id": id,
                    "new_id": new_id,
                    "to_project": to,
                }))
            }
        }
        (None, Some(from)) => {
            let from_opt = empty_to_none(from);
            if dry_run {
                let mems = queries::list_memories_by_project(conn, from_opt)?;
                Ok(json!({
                    "status": "dry_run",
                    "would_copy": mems.len(),
                    "from_project": from_opt,
                    "to_project": to,
                    "source_ids": mems.iter().map(|m| &m.id).collect::<Vec<_>>(),
                }))
            } else {
                let new_ids = queries::copy_memories_by_project(conn, from_opt, to)?;
                Ok(json!({
                    "status": "copied",
                    "count": new_ids.len(),
                    "from_project": from_opt,
                    "to_project": to,
                    "new_ids": new_ids,
                }))
            }
        }
        (None, None) => Ok(json!({
            "status": "error",
            "message": "Either --id or --from must be provided",
        })),
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// `--scope global` should parse to `MemoryScope::Global` and leave other
    /// flags untouched.
    #[test]
    fn parse_store_scope_global() {
        let cli = Cli::try_parse_from(["memory", "store", "hello", "--scope", "global"]).unwrap();
        match cli {
            Cli::Store { scope, content, .. } => {
                assert_eq!(scope, Some(MemoryScope::Global));
                assert_eq!(content, "hello");
            }
            _ => panic!("expected Store variant"),
        }
    }

    /// Omitting `--scope` must leave the field `None` (not a default) so the
    /// dispatch layer can distinguish "not chosen" from "explicitly project".
    #[test]
    fn parse_store_scope_absent_is_none() {
        let cli = Cli::try_parse_from(["memory", "store", "hello"]).unwrap();
        match cli {
            Cli::Store { scope, .. } => assert_eq!(scope, None),
            _ => panic!("expected Store variant"),
        }
    }

    /// Explicit `--scope project` is distinguishable from the default and
    /// suppresses the reflection hint.
    #[test]
    fn parse_store_scope_project_explicit() {
        let cli =
            Cli::try_parse_from(["memory", "store", "hi", "--scope", "project"]).unwrap();
        match cli {
            Cli::Store { scope, .. } => assert_eq!(scope, Some(MemoryScope::Project)),
            _ => panic!("expected Store variant"),
        }
    }

    /// `resolve_boosts` with the normal path returns both current-project and
    /// global sentinel wired up with their respective multipliers.
    #[test]
    fn resolve_boosts_default_wires_both_scopes() {
        let boosts = resolve_boosts(None, Some("agent-memory"), false);
        assert_eq!(boosts.current_project, Some("agent-memory"));
        assert_eq!(boosts.global_project, Some(GLOBAL_PROJECT_IDENT));
        assert!((boosts.project_boost - PROJECT_BOOST).abs() < f32::EPSILON);
        assert!((boosts.global_boost - GLOBAL_BOOST).abs() < f32::EPSILON);
    }

    /// `--no-project-boost` must disable BOTH boosts so users get a genuinely
    /// flat ranking when they ask for one.
    #[test]
    fn resolve_boosts_disabled_kills_both() {
        let boosts = resolve_boosts(None, Some("agent-memory"), true);
        assert_eq!(boosts.current_project, None);
        assert_eq!(boosts.global_project, None);
        assert_eq!(boosts.project_boost, 1.0);
        assert_eq!(boosts.global_boost, 1.0);
    }

    /// Explicit `--project foo` overrides cwd-derived ident for the boost.
    #[test]
    fn resolve_boosts_explicit_project_overrides_cwd() {
        let boosts = resolve_boosts(Some("foo"), Some("agent-memory"), false);
        assert_eq!(boosts.current_project, Some("foo"));
    }

    /// Hint: with at least one global match, emit the directive-style count
    /// and suppress the zero-global reflection prompt.
    #[test]
    fn results_hint_global_present_emits_directive_count() {
        let h = results_hint(0, 2, 5, Some("agent-memory")).expect("hint should be present");
        assert!(h.contains("2 of 5 results are global-scope preferences"));
        assert!(!h.contains("No global-scope preferences matched"));
    }

    /// Hint: zero global matches during a scoped retrieval fires the
    /// reflection prompt asking the agent to pause.
    #[test]
    fn results_hint_zero_global_fires_reflection_prompt() {
        let h = results_hint(0, 0, 3, Some("agent-memory")).expect("hint should be present");
        assert!(h.contains("No global-scope preferences matched"));
    }

    /// Hint: cross-project + zero globals should include both the
    /// prior-art warning and the reflection prompt.
    #[test]
    fn results_hint_combines_cross_project_and_zero_global() {
        let h = results_hint(3, 0, 5, Some("agent-memory")).expect("hint should be present");
        assert!(h.contains("cross-project"));
        assert!(h.contains("No global-scope preferences matched"));
    }

    /// Hint: flat retrieval (no current-project) and no globals → no hint
    /// at all (nothing useful to say).
    #[test]
    fn results_hint_flat_ranking_no_globals_emits_nothing() {
        assert!(results_hint(0, 0, 5, None).is_none());
    }

    /// Hint: flat retrieval with globals still surfaces the directive-count
    /// — universal preferences are always worth flagging.
    #[test]
    fn results_hint_flat_ranking_with_globals_surfaces_count() {
        let h = results_hint(0, 1, 3, None).expect("hint should be present");
        assert!(h.contains("global-scope preferences"));
    }

    /// `store_scope_hint` mentions both `--scope global` and the silent
    /// mis-classification risk (so the agent knows *why* to reclassify).
    #[test]
    fn store_scope_hint_mentions_global_and_risk() {
        let h = store_scope_hint();
        assert!(h.contains("--scope global"));
        assert!(h.contains("silent mis-classification"));
    }

    #[test]
    fn scope_label_round_trip() {
        assert_eq!(scope_label(MemoryScope::Project), "project");
        assert_eq!(scope_label(MemoryScope::Global), "global");
    }
}
