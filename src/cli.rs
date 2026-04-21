use clap::{Parser, ValueEnum};
use rusqlite::Connection;
use serde_json::{json, Value};

use crate::config::Config;
use crate::db::models::Memory;
use crate::db::queries;
use crate::embedding;
use crate::error::MemoryError;
use crate::project;
use crate::search::{self, SearchOptions, SearchResult};

/// Score multiplier applied to memories tagged with the current project.
/// Strong cross-project matches can still out-rank weak current-project hits;
/// the boost tilts ties toward local context without hard-filtering prior art.
pub const PROJECT_BOOST: f32 = 1.5;
pub const DEFAULT_PREVIEW_CHARS: usize = 160;

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    /// Compact: id, tags, project, quality flags, and a content preview.
    Brief,
    /// Full content plus all metadata.
    Full,
}

#[derive(Parser)]
#[command(name = "memory", about = "Persistent hybrid-search memory system for AI coding agents", version = env!("AGENT_MEMORY_VERSION"))]
pub enum Cli {
    /// Save a memory with auto-embedding and BM25 indexing.
    ///
    /// If --project is omitted, the current project is auto-detected from the
    /// working directory's git remote or canonical path.
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
        } => {
            let tag_list = tags.map(|t| t.split(',').map(|s| s.trim().to_string()).collect());
            let resolved_project = if no_project {
                None
            } else {
                project.or(cwd_project.clone())
            };

            let mut memory = Memory::new(
                content.clone(),
                tag_list,
                resolved_project,
                agent,
                source_file,
                Some(memory_type),
            );

            let emb = embedding::embed_text(&content, &config.model_cache_dir)?;
            memory.embedding = Some(emb);

            queries::insert_memory(conn, &memory)?;

            let output = json!({
                "status": "stored",
                "id": memory.id,
                "project": memory.project,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
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
            let (boost_project, boost_factor) =
                resolve_boost(project.as_deref(), cwd_project.as_deref(), no_project_boost);
            let opts = SearchOptions {
                limit,
                current_project: boost_project,
                boost_factor,
                only_project: only.as_deref(),
            };
            let results = search::hybrid_search(conn, &query, opts, &config.model_cache_dir)?;
            let output = render_ranked_output(&results, boost_project, format, preview_chars);
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
            let (boost_project, boost_factor) =
                resolve_boost(project.as_deref(), cwd_project.as_deref(), no_project_boost);
            let opts = SearchOptions {
                limit,
                current_project: boost_project,
                boost_factor,
                only_project: only.as_deref(),
            };
            let results = search::hybrid_search(conn, &description, opts, &config.model_cache_dir)?;
            let output = render_ranked_output(&results, boost_project, format, preview_chars);
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
    }
    Ok(())
}

fn resolve_boost<'a>(
    explicit: Option<&'a str>,
    cwd: Option<&'a str>,
    disable: bool,
) -> (Option<&'a str>, f32) {
    if disable {
        (None, 1.0)
    } else {
        (explicit.or(cwd), PROJECT_BOOST)
    }
}

fn render_ranked_output(
    results: &[SearchResult],
    current_project: Option<&str>,
    format: OutputFormat,
    preview_chars: usize,
) -> Value {
    let total = results.len();
    let cross_project_count = if current_project.is_some() {
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
    if let Some(cp) = current_project {
        out.insert("current_project".into(), Value::String(cp.to_string()));
        out.insert(
            "cross_project_count".into(),
            Value::Number(cross_project_count.into()),
        );
        if cross_project_count > 0 {
            out.insert(
                "hint".into(),
                Value::String(cross_project_hint(cross_project_count, total, cp)),
            );
        }
    }
    Value::Object(out)
}

fn cross_project_hint(cross: usize, total: usize, current: &str) -> String {
    if cross == total {
        format!(
            "All {total} results are from other projects (no memories tagged '{current}' matched). Treat as prior-art or general guidance, not direct context for the current project."
        )
    } else {
        format!(
            "{cross} of {total} results are cross-project (is_current_project=false). Use those as prior-art or general guidance, not direct context for '{current}'. Use `memory get <id>` for full content."
        )
    }
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
