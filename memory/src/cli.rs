use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use rusqlite::Connection;

use crate::config::Config;
use crate::db::models::Memory;
use crate::db::queries::{self, ResolvedId};
use crate::embedding;
use crate::error::MemoryError;
use crate::project;
use crate::render;
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
    /// Dual-mode command:
    ///
    ///   - `memory update`                  → check for and install the
    ///     latest `memory` release (self-updater; original behavior).
    ///   - `memory update <id> --content X` → atomically re-author the
    ///     memory at `<id>`: replace content, archive the prior body to
    ///     `content_raw`, bump `updated_at`, clear `superseded_by`, and
    ///     re-embed. Supports short ID prefixes (≥4 chars).
    ///
    /// The two modes are distinguished purely by presence of the positional
    /// `<id>` argument. This lets the agentic dream pass use the CLI contract
    /// documented in `docs/plan` while preserving the pre-2.3 self-updater
    /// invocation verbatim. Primary writer of the content-update form is
    /// `memory-dream` running under a tool-enabled LLM backend.
    Update {
        /// Memory ID (full UUID or ≥4-char prefix). When omitted, the
        /// command runs the self-updater. When provided, `--content` is
        /// required.
        id: Option<String>,
        /// New content body. Required when `id` is supplied; ignored by the
        /// self-updater path.
        #[arg(long)]
        content: Option<String>,
        /// Optional comma-separated tag replacement. Omit to preserve
        /// existing tags. Ignored by the self-updater path.
        #[arg(long)]
        tags: Option<String>,
        /// Optional memory type replacement (user | feedback | project | reference).
        /// Omit to preserve the existing type. Ignored by the self-updater path.
        #[arg(short = 'm', long)]
        memory_type: Option<String>,
    },
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
        /// Strip the `<memory-rules>` block from detected rule files and
        /// remove the paired `autoMemoryEnabled` key from any Claude
        /// `settings.json`. Inverse of the default install.
        #[arg(long)]
        remove: bool,
    },

    /// Install an Agent Skill so the `memory` CLI is auto-advertised to
    /// sessions via the always-loaded skill description (~100 tokens). The
    /// full body only loads on demand.
    ///
    /// Writes `SKILL.md` to every known skill-root unconditionally:
    ///   - `~/.claude/skills/agent-memory/SKILL.md` (Claude Code, tool-native)
    ///   - `~/.agents/skills/agent-memory/SKILL.md` (cross-agent alias — read
    ///     by Gemini CLI, Codex, and any future frontend honoring the shared
    ///     `.agents/skills/` convention)
    ///
    /// Every frontend reads the same YAML frontmatter + Markdown body, so
    /// identical byte contents are written to each target. No auto-detection
    /// of whether the agent is installed — running `memory setup skill` is
    /// the opt-in signal. Legacy install paths (v1.4.0/v1.4.1's
    /// `~/.gemini/skills/agent-memory/SKILL.md`) are scrubbed on every
    /// install and every `--remove`.
    Skill {
        /// Show the resulting file content without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Print the SKILL.md to stdout and exit (no file IO).
        #[arg(long)]
        print: bool,
        /// Delete the installed `SKILL.md` from every known target. Missing
        /// files are a silent no-op per target. Inverse of the default
        /// install, matching `setup rules --remove`.
        #[arg(long)]
        remove: bool,
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

            let mut attrs: Vec<(&str, String)> = vec![
                ("id", render::short_id(&memory.id).to_string()),
                ("scope", scope_label(resolved_scope).to_string()),
            ];
            if let Some(p) = memory.project.as_deref() {
                attrs.push(("project", p.to_string()));
            }
            println!("{}", render::render_action_result("stored", &attrs));
            // Reflection hint: nudge the agent to reconsider scope only when
            // the memory type is most likely to be cross-cutting (user or
            // feedback) AND the user didn't pick a scope deliberately. No
            // noise on `project`/`reference` stores or when --scope was
            // passed explicitly — silence is the default.
            if !scope_explicit
                && resolved_scope == MemoryScope::Project
                && matches!(memory_type.as_str(), "user" | "feedback")
            {
                println!("{}", render::render_hint(&store_scope_hint()));
            }
        }
        Cli::Search {
            query,
            limit,
            project,
            only,
            no_project_boost,
            format: _,
            preview_chars: _,
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
            print_ranked(&results, &boosts);
        }
        Cli::Context {
            description,
            limit,
            project,
            only,
            no_project_boost,
            format: _,
            preview_chars: _,
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
            print_ranked(&results, &boosts);
        }
        Cli::Recall {
            project,
            agent,
            tags,
            memory_type,
            limit,
            format: _,
            preview_chars: _,
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

            println!(
                "{}",
                render::render_memory_list(&memories, cwd_project.as_deref())
            );
            println!("{}", render::render_usage_legend());
        }
        Cli::Forget { id, query } => {
            if let Some(id) = id {
                match queries::resolve_id_prefix(conn, &id)? {
                    ResolvedId::Exact(full_id) => {
                        let deleted = queries::delete_memory(conn, &full_id)?;
                        let status = if deleted { "forgot" } else { "not_found" };
                        let short = render::short_id(&full_id).to_string();
                        println!("{}", render::render_action_result(status, &[("id", short)]));
                    }
                    ResolvedId::Ambiguous(cands) => {
                        println!("{}", render::render_ambiguous(&id, &cands));
                    }
                    ResolvedId::NotFound => {
                        println!(
                            "{}",
                            render::render_action_result("not_found", &[("id", id)])
                        );
                    }
                }
            } else if let Some(query) = query {
                let opts = SearchOptions::new(5);
                let results = search::hybrid_search(conn, &query, opts, &config.model_cache_dir)?;
                if results.is_empty() {
                    println!("{}", render::render_action_result("no_matches", &[]));
                } else {
                    let mut deleted = 0usize;
                    for r in &results {
                        if queries::delete_memory(conn, &r.memory.id)? {
                            deleted += 1;
                        }
                    }
                    println!(
                        "{}",
                        render::render_action_result("forgot", &[("count", deleted.to_string())])
                    );
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
            let status = if dry_run { "dry_run" } else { "pruned" };
            println!(
                "{}",
                render::render_action_result(status, &[("count", pruned.len().to_string())])
            );
        }
        Cli::Get {
            ids,
            format: _,
            preview_chars: _,
        } => {
            // Each arg is resolved independently: a short prefix that maps to
            // one memory produces a <memory>, an ambiguous prefix produces an
            // <ambiguous> block, and a miss produces a `<result status="not_found"
            // id="..."/>` — all routed through the renderer so the surface stays
            // consistent.
            let mut fetched: Vec<Memory> = Vec::with_capacity(ids.len());
            let mut out_lines: Vec<String> = Vec::with_capacity(ids.len());
            for id in &ids {
                match queries::resolve_id_prefix(conn, id)? {
                    ResolvedId::Exact(full_id) => match queries::get_memory_by_id(conn, &full_id) {
                        Ok(m) => {
                            out_lines.push(render::render_memory(&m));
                            fetched.push(m);
                        }
                        Err(MemoryError::NotFound(_)) => {
                            out_lines.push(render::render_action_result(
                                "not_found",
                                &[("id", id.clone())],
                            ));
                        }
                        Err(e) => return Err(e),
                    },
                    ResolvedId::Ambiguous(cands) => {
                        out_lines.push(render::render_ambiguous(id, &cands));
                    }
                    ResolvedId::NotFound => {
                        out_lines.push(render::render_action_result(
                            "not_found",
                            &[("id", id.clone())],
                        ));
                    }
                }
            }
            let hit_ids: Vec<String> = fetched.iter().map(|m| m.id.clone()).collect();
            queries::increment_access(conn, &hit_ids)?;
            for line in &out_lines {
                println!("{line}");
            }
        }
        Cli::List {
            limit,
            project,
            memory_type,
            format: _,
            preview_chars: _,
        } => {
            let memories = queries::list_memories(
                conn,
                project.as_deref(),
                None,
                None,
                memory_type.as_deref(),
                limit,
            )?;
            println!(
                "{}",
                render::render_memory_list(&memories, cwd_project.as_deref())
            );
            println!("{}", render::render_usage_legend());
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
            let resolved_id = resolve_id_arg(conn, id.as_deref())?;
            run_move(
                conn,
                resolved_id.as_deref(),
                from.as_deref(),
                new_project,
                dry_run,
            )?;
        }
        Cli::Copy {
            id,
            from,
            to,
            dry_run,
        } => {
            let new_project = empty_to_none(&to);
            let resolved_id = resolve_id_arg(conn, id.as_deref())?;
            run_copy(
                conn,
                resolved_id.as_deref(),
                from.as_deref(),
                new_project,
                dry_run,
            )?;
        }
        Cli::Projects => {
            let rows = queries::list_projects(conn)?;
            println!("{}", render::render_projects(&rows, cwd_project.as_deref()));
        }
        Cli::Serve => {
            unreachable!("Serve is handled in main.rs");
        }
        Cli::Update {
            id,
            content,
            tags,
            memory_type,
        } => match id {
            None => {
                if content.is_some() || tags.is_some() || memory_type.is_some() {
                    return Err(MemoryError::Config(
                        "`memory update --content ...` requires a positional <id>. \
                             Run `memory update <id> --content \"...\"` to re-author, or \
                             `memory update` alone to run the self-updater."
                            .to_string(),
                    ));
                }
                crate::updater::manual_update()?;
            }
            Some(raw_id) => {
                let new_content = content.ok_or_else(|| {
                    MemoryError::Config(
                        "`memory update <id>` requires --content \"...\"".to_string(),
                    )
                })?;
                run_update_content(
                    conn,
                    &config,
                    &raw_id,
                    &new_content,
                    tags.as_deref(),
                    memory_type.as_deref(),
                )?;
            }
        },
        Cli::Setup { command } => {
            execute_setup(command).map_err(|e| MemoryError::Config(format!("{e:#}")))?;
        }
    }
    Ok(())
}

/// Print a ranked result set (`search`/`context`) as grouped light-XML.
/// Delegates to the `render` module, appends the reflection hint, and emits
/// the `<usage>` legend at the bottom so a cold agent knows how to consume
/// the output (short-ID semantics, section meanings, how to fetch full
/// content). The legend ships unconditionally — even on zero-result runs —
/// because that's when a new caller most needs the guidance.
fn print_ranked(results: &[SearchResult], boosts: &BoostConfig<'_>) {
    let total = results.len();
    let globals = results.iter().filter(|r| r.is_global).count();
    let cross = if boosts.current_project.is_some() {
        results.iter().filter(|r| !r.is_current_project).count()
    } else {
        0
    };
    let hint = results_hint(cross, globals, total, boosts.current_project);
    let rendered = render::render_search_results(results, boosts.current_project, hint.as_deref());
    // Empty input yields an empty render; emit an explicit empty marker so
    // callers can tell "query ran, zero hits" from a silent failure.
    if rendered.is_empty() {
        println!("<results count=\"0\"/>");
    } else {
        println!("{rendered}");
    }
    println!("{}", render::render_usage_legend());
}

/// Resolve a user-supplied `--id` argument for move/copy subcommands through
/// the short-prefix resolver. Ambiguous prefixes emit an `<ambiguous>` block
/// and return `Ok(None)` so the caller skips the mutation; missing IDs return
/// a `not_found` result line. Full UUIDs and unique prefixes return the full
/// UUID wrapped in `Some`.
fn resolve_id_arg(conn: &Connection, id: Option<&str>) -> Result<Option<String>, MemoryError> {
    match id {
        None => Ok(None),
        Some(raw) => match queries::resolve_id_prefix(conn, raw)? {
            ResolvedId::Exact(full) => Ok(Some(full)),
            ResolvedId::Ambiguous(cands) => {
                println!("{}", render::render_ambiguous(raw, &cands));
                // Sentinel: returning `None` here would let the caller fall
                // through to the `--from` branch. Instead we pass a token
                // that the mutation fns treat as "id was supplied but
                // unresolved" — achieved by returning an empty string, which
                // `run_move`/`run_copy` short-circuit on below.
                Ok(Some(String::new()))
            }
            ResolvedId::NotFound => {
                println!(
                    "{}",
                    render::render_action_result("not_found", &[("id", raw.to_string())])
                );
                Ok(Some(String::new()))
            }
        },
    }
}

/// Handle `memory update <id> --content ...` — the content-update form.
///
/// Resolves the ID prefix (short or full UUID), re-embeds the new content,
/// and calls `queries::update_content` which swaps `content`↔`content_raw`,
/// bumps `updated_at`, and clears `superseded_by`. Emits a light-XML
/// `<result status="updated" id="..."/>` line on success or the usual
/// `<ambiguous>` / `not_found` shape on resolution failures.
fn run_update_content(
    conn: &Connection,
    config: &Config,
    raw_id: &str,
    new_content: &str,
    tags_csv: Option<&str>,
    memory_type: Option<&str>,
) -> Result<(), MemoryError> {
    if new_content.trim().is_empty() {
        return Err(MemoryError::Config(
            "--content must not be empty".to_string(),
        ));
    }

    let full_id = match queries::resolve_id_prefix(conn, raw_id)? {
        ResolvedId::Exact(id) => id,
        ResolvedId::Ambiguous(cands) => {
            println!("{}", render::render_ambiguous(raw_id, &cands));
            return Ok(());
        }
        ResolvedId::NotFound => {
            println!(
                "{}",
                render::render_action_result("not_found", &[("id", raw_id.to_string())])
            );
            return Ok(());
        }
    };

    let tag_vec: Option<Vec<String>> = tags_csv.map(|t| {
        t.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });

    let changed =
        queries::update_content(conn, &full_id, new_content, tag_vec.as_deref(), memory_type)?;
    if !changed {
        // resolve_id_prefix returned Exact but the UPDATE touched zero rows —
        // implies a TOCTOU delete between resolve and update. Surface it
        // cleanly rather than claiming success.
        println!(
            "{}",
            render::render_action_result("not_found", &[("id", raw_id.to_string())])
        );
        return Ok(());
    }

    // Re-embed after the content swap so vector search picks up the new
    // form on the next query. We intentionally embed *after* the SQL UPDATE
    // so a flaky embedder doesn't block the metadata update — the worst
    // case is a stale embedding that the next dream pass corrects.
    let new_embedding = embedding::embed_text(new_content, &config.model_cache_dir)?;
    let blob = crate::db::models::embedding_to_blob(&new_embedding);
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE memories SET embedding = ?1, embedding_model = ?2, updated_at = ?3
         WHERE id = ?4",
        rusqlite::params![
            blob,
            crate::db::models::EMBEDDING_MODEL_NAME_DEFAULT,
            now,
            full_id
        ],
    )?;

    println!(
        "{}",
        render::render_action_result("updated", &[("id", render::short_id(&full_id).to_string())])
    );
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
            remove,
        }) => rules::run(target, all, dry_run, print, remove),
        Some(SetupCommands::Skill {
            dry_run,
            print,
            remove,
        }) => skill::run(dry_run, print, remove),
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

/// Treat empty strings as "no project" for the move/copy `--from`/`--to` flags.
/// Lets users explicitly target or assign a NULL project without a second flag.
fn empty_to_none(s: &str) -> Option<&str> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Execute a `memory move` and print a light-XML `<result>` line.
///
/// `id` may be `None` (use `from`), `Some(full_uuid)`, or `Some("")` — the
/// empty string is a sentinel from [`resolve_id_arg`] meaning "the user
/// supplied an ID but it was ambiguous or not found; the disambiguation
/// prompt has already been printed, do nothing here".
fn run_move(
    conn: &Connection,
    id: Option<&str>,
    from: Option<&str>,
    to: Option<&str>,
    dry_run: bool,
) -> Result<(), MemoryError> {
    match (id, from) {
        (Some(""), _) => Ok(()), // unresolved sentinel — already handled
        (Some(id), _) => {
            if dry_run {
                let mem = queries::get_memory_by_id(conn, id)?;
                let mut attrs: Vec<(&str, String)> = vec![
                    ("id", render::short_id(&mem.id).to_string()),
                    ("would_move", "1".to_string()),
                ];
                if let Some(p) = mem.project.as_deref() {
                    attrs.push(("from_project", p.to_string()));
                }
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result("dry_run", &attrs));
            } else {
                let changed = queries::move_memory_by_id(conn, id, to)?;
                let status = if changed { "moved" } else { "not_found" };
                let mut attrs: Vec<(&str, String)> = vec![("id", render::short_id(id).to_string())];
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result(status, &attrs));
            }
            Ok(())
        }
        (None, Some(from)) => {
            let from_opt = empty_to_none(from);
            if dry_run {
                let mems = queries::list_memories_by_project(conn, from_opt)?;
                let mut attrs: Vec<(&str, String)> = vec![
                    ("would_move", mems.len().to_string()),
                    ("from_project", from_opt.unwrap_or("").to_string()),
                ];
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result("dry_run", &attrs));
            } else {
                let count = queries::move_memories_by_project(conn, from_opt, to)?;
                let mut attrs: Vec<(&str, String)> = vec![
                    ("count", count.to_string()),
                    ("from_project", from_opt.unwrap_or("").to_string()),
                ];
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result("moved", &attrs));
            }
            Ok(())
        }
        (None, None) => {
            println!(
                "{}",
                render::render_action_result(
                    "error",
                    &[(
                        "message",
                        "Either --id or --from must be provided".to_string()
                    )]
                )
            );
            Ok(())
        }
    }
}

/// Execute a `memory copy` and print a light-XML `<result>` line. See
/// [`run_move`] for the sentinel-handling contract on `id`.
fn run_copy(
    conn: &Connection,
    id: Option<&str>,
    from: Option<&str>,
    to: Option<&str>,
    dry_run: bool,
) -> Result<(), MemoryError> {
    match (id, from) {
        (Some(""), _) => Ok(()), // unresolved sentinel — already handled
        (Some(id), _) => {
            if dry_run {
                let mem = queries::get_memory_by_id(conn, id)?;
                let mut attrs: Vec<(&str, String)> = vec![
                    ("source_id", render::short_id(&mem.id).to_string()),
                    ("would_copy", "1".to_string()),
                ];
                if let Some(p) = mem.project.as_deref() {
                    attrs.push(("from_project", p.to_string()));
                }
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result("dry_run", &attrs));
            } else {
                let new_id = queries::copy_memory_by_id(conn, id, to)?;
                let mut attrs: Vec<(&str, String)> = vec![
                    ("source_id", render::short_id(id).to_string()),
                    ("new_id", render::short_id(&new_id).to_string()),
                ];
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result("copied", &attrs));
            }
            Ok(())
        }
        (None, Some(from)) => {
            let from_opt = empty_to_none(from);
            if dry_run {
                let mems = queries::list_memories_by_project(conn, from_opt)?;
                let mut attrs: Vec<(&str, String)> = vec![
                    ("would_copy", mems.len().to_string()),
                    ("from_project", from_opt.unwrap_or("").to_string()),
                ];
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result("dry_run", &attrs));
            } else {
                let new_ids = queries::copy_memories_by_project(conn, from_opt, to)?;
                let mut attrs: Vec<(&str, String)> = vec![
                    ("count", new_ids.len().to_string()),
                    ("from_project", from_opt.unwrap_or("").to_string()),
                ];
                if let Some(t) = to {
                    attrs.push(("to_project", t.to_string()));
                }
                println!("{}", render::render_action_result("copied", &attrs));
            }
            Ok(())
        }
        (None, None) => {
            println!(
                "{}",
                render::render_action_result(
                    "error",
                    &[(
                        "message",
                        "Either --id or --from must be provided".to_string()
                    )]
                )
            );
            Ok(())
        }
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
        let cli = Cli::try_parse_from(["memory", "store", "hi", "--scope", "project"]).unwrap();
        match cli {
            Cli::Store { scope, .. } => assert_eq!(scope, Some(MemoryScope::Project)),
            _ => panic!("expected Store variant"),
        }
    }

    /// Bare `memory update` still parses as the self-updater (no id, no
    /// content). The dispatch layer routes to the manual-update path.
    #[test]
    fn parse_update_bare_is_self_updater() {
        let cli = Cli::try_parse_from(["memory", "update"]).unwrap();
        match cli {
            Cli::Update {
                id,
                content,
                tags,
                memory_type,
            } => {
                assert!(id.is_none());
                assert!(content.is_none());
                assert!(tags.is_none());
                assert!(memory_type.is_none());
            }
            _ => panic!("expected Update variant"),
        }
    }

    /// `memory update <id> --content "..."` parses the content-update form.
    #[test]
    fn parse_update_content_form() {
        let cli = Cli::try_parse_from([
            "memory",
            "update",
            "aabbccdd",
            "--content",
            "new body\n- fact",
            "--tags",
            "a,b",
            "-m",
            "project",
        ])
        .unwrap();
        match cli {
            Cli::Update {
                id,
                content,
                tags,
                memory_type,
            } => {
                assert_eq!(id.as_deref(), Some("aabbccdd"));
                assert_eq!(content.as_deref(), Some("new body\n- fact"));
                assert_eq!(tags.as_deref(), Some("a,b"));
                assert_eq!(memory_type.as_deref(), Some("project"));
            }
            _ => panic!("expected Update variant"),
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

    // -- Light-XML output shape tests ---------------------------------------
    //
    // These replace the removed JSON-shape assertions. They exercise the same
    // render helpers the CLI dispatch layer prints, so we lock in the text
    // format without having to capture stdout from the `execute` function.

    use crate::db::models::Memory;
    use crate::render;

    fn mk_mem(id: &str, content: &str, project: Option<&str>) -> Memory {
        Memory {
            id: id.to_string(),
            content: content.to_string(),
            tags: None,
            project: project.map(String::from),
            agent: None,
            source_file: None,
            created_at: "2026-04-23T00:00:00Z".to_string(),
            updated_at: "2026-04-23T00:00:00Z".to_string(),
            access_count: 0,
            embedding: None,
            memory_type: Some("user".to_string()),
            content_raw: None,
            superseded_by: None,
            condenser_version: None,
            embedding_model: None,
        }
    }

    /// Store result should emit a single `<result status="stored" .../>` line
    /// carrying the short ID, scope, and project. No JSON braces anywhere.
    #[test]
    fn store_output_is_light_xml_result_line() {
        let s = render::render_action_result(
            "stored",
            &[
                (
                    "id",
                    render::short_id("abcdef12-3456-7890-abcd-ef1234567890").to_string(),
                ),
                ("scope", "global".to_string()),
                ("project", "__global__".to_string()),
            ],
        );
        assert_eq!(
            s,
            r#"<result status="stored" id="abcdef12" scope="global" project="__global__"/>"#
        );
        assert!(!s.contains('{'));
        assert!(!s.contains('}'));
    }

    /// Forget-by-id success emits `<result status="forgot" .../>` with the
    /// short ID. Not-found emits `status="not_found"` instead.
    #[test]
    fn forget_output_uses_forgot_status_on_success() {
        let s_ok = render::render_action_result("forgot", &[("id", "a4936eff".to_string())]);
        assert_eq!(s_ok, r#"<result status="forgot" id="a4936eff"/>"#);

        let s_miss = render::render_action_result("not_found", &[("id", "deadbeef".to_string())]);
        assert_eq!(s_miss, r#"<result status="not_found" id="deadbeef"/>"#);
    }

    /// Forget-by-query with no hits emits `no_matches`; with hits emits the
    /// count in the attribute set.
    #[test]
    fn forget_by_query_status_no_matches_vs_forgot() {
        assert_eq!(
            render::render_action_result("no_matches", &[]),
            r#"<result status="no_matches"/>"#
        );
        let s = render::render_action_result("forgot", &[("count", "3".to_string())]);
        assert_eq!(s, r#"<result status="forgot" count="3"/>"#);
    }

    /// Prune emits `<result status="pruned" count=".."/>` or `dry_run` when
    /// the --dry-run flag is passed. No longer carries the memory content list
    /// (callers can list + prune separately for a preview now).
    #[test]
    fn prune_output_uses_count_attribute() {
        let s = render::render_action_result("pruned", &[("count", "7".to_string())]);
        assert_eq!(s, r#"<result status="pruned" count="7"/>"#);
        let s_dry = render::render_action_result("dry_run", &[("count", "7".to_string())]);
        assert_eq!(s_dry, r#"<result status="dry_run" count="7"/>"#);
    }

    /// Memory get renders a `<memory>` wrapper with metadata attributes and
    /// the full content as inner text.
    #[test]
    fn get_output_is_memory_wrapper_with_full_content() {
        let m = mk_mem(
            "a4936eff-1234-5678-9abc-def012345678",
            "Content line\nwith newline",
            Some("agent-memory"),
        );
        let s = render::render_memory(&m);
        assert!(s.starts_with("<memory id=\"a4936eff\""));
        assert!(s.contains("project=\"agent-memory\""));
        assert!(s.contains("Content line\nwith newline"));
        assert!(s.ends_with("</memory>"));
    }

    /// Projects output uses a `<projects count=".."/>` block with lines
    /// marking the current project with `*`.
    #[test]
    fn projects_output_marks_current_project() {
        let rows = vec![
            (Some("agent-memory".to_string()), 42_i64),
            (Some("colorithmic".to_string()), 7),
        ];
        let s = render::render_projects(&rows, Some("agent-memory"));
        assert!(s.contains("<projects count=\"2\">"));
        assert!(s.contains("*agent-memory (42)"));
        assert!(s.contains(" colorithmic (7)"));
    }

    /// List empty projects surfaces a self-closing `<projects count="0"/>`.
    #[test]
    fn projects_output_empty_is_self_closing() {
        assert_eq!(
            render::render_projects(&[], None),
            "<projects count=\"0\"/>"
        );
    }

    /// Memory list renders `<memories count=".."/>` with the current-project
    /// marker when cwd matches.
    #[test]
    fn list_output_is_memory_list_block() {
        let mems = vec![
            mk_mem("11111111-aaaa", "local mem", Some("agent-memory")),
            mk_mem("22222222-bbbb", "other mem", Some("colorithmic")),
        ];
        let s = render::render_memory_list(&mems, Some("agent-memory"));
        assert!(s.contains("<memories count=\"2\">"));
        assert!(s.contains("1.*(user) agent-memory"));
        assert!(s.contains("2. (user) colorithmic"));
        assert!(s.ends_with("</memories>"));
    }

    /// Verify the output of every render helper contains zero JSON braces —
    /// the user's explicit rule ("JSON output goes away entirely").
    #[test]
    fn no_render_helper_emits_json_braces() {
        let m = mk_mem("11111111-aaaa-bbbb-cccc-dddddddddddd", "hi", Some("p"));
        let checks: Vec<String> = vec![
            render::render_action_result("stored", &[("id", "abc".to_string())]),
            render::render_hint("x"),
            render::render_memory(&m),
            render::render_memory_list(std::slice::from_ref(&m), None),
            render::render_projects(&[(Some("p".to_string()), 1)], None),
            render::render_ambiguous("abcd", std::slice::from_ref(&m)),
        ];
        for s in checks {
            assert!(!s.contains('{'), "unexpected '{{' in output: {s}");
            assert!(!s.contains('}'), "unexpected '}}' in output: {s}");
        }
    }
}
