use std::sync::{Arc, Mutex};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::schemars;
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::Config;
use crate::db::models::Memory;
use crate::db::queries;
use crate::embedding;
use crate::error::MemoryError;
use crate::project;
use crate::search::{self, SearchOptions, SearchResult};

/// Score multiplier applied to memories tagged with the current project.
const PROJECT_BOOST: f32 = 1.5;
/// Score multiplier applied to memories tagged with the global-scope
/// sentinel project. Intentionally smaller than `PROJECT_BOOST` — see the
/// CLI module for the full tradeoff rationale.
const GLOBAL_BOOST: f32 = 1.25;
/// Reserved `project` ident for global-scoped memories. Kept in sync with
/// `crate::cli::GLOBAL_PROJECT_IDENT`. Re-declared here rather than
/// cross-imported so the MCP surface stays independent of the CLI module's
/// public API shape.
const GLOBAL_PROJECT_IDENT: &str = "__global__";
const DEFAULT_PREVIEW_CHARS: usize = 160;

#[derive(Clone)]
pub struct MemoryServer {
    tool_router: ToolRouter<Self>,
    conn: Arc<Mutex<Connection>>,
    config: Arc<Config>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreArgs {
    /// Memory content text to store
    pub content: String,
    /// Comma-separated tags for categorization
    pub tags: Option<String>,
    /// Project identifier. Defaults to the cwd-derived project ident if omitted.
    pub project: Option<String>,
    /// Agent identifier (e.g. "unreal-plugin-developer")
    pub agent: Option<String>,
    /// Source file path this memory relates to
    pub source_file: Option<String>,
    /// Memory type: user, feedback, project, reference
    pub memory_type: Option<String>,
    /// If true, store without any project tag (skips cwd auto-detection).
    pub no_project: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    /// Search query (natural language or keywords)
    pub query: String,
    /// Maximum number of results to return (default: 10)
    pub limit: Option<usize>,
    /// Project to boost in ranking. Defaults to cwd-derived project ident.
    pub project: Option<String>,
    /// Hard filter: only return memories whose project equals this string.
    pub only: Option<String>,
    /// If true, disable the current-project boost entirely (flat ranking).
    pub no_project_boost: Option<bool>,
    /// Output format: "brief" (default) or "full".
    pub format: Option<String>,
    /// Preview length for brief format (default: 160).
    pub preview_chars: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallArgs {
    /// Filter by project identifier
    pub project: Option<String>,
    /// Filter by agent identifier
    pub agent: Option<String>,
    /// Comma-separated tags to filter by
    pub tags: Option<String>,
    /// Filter by memory type: user, feedback, project, reference
    pub memory_type: Option<String>,
    /// Maximum number of results (default: 10)
    pub limit: Option<usize>,
    /// Output format: "brief" (default) or "full".
    pub format: Option<String>,
    /// Preview length for brief format (default: 160).
    pub preview_chars: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetArgs {
    /// Specific memory ID to delete
    pub id: Option<String>,
    /// Search query to find and delete matching memories
    pub query: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PruneArgs {
    /// Maximum age in days (default: 90)
    pub max_age_days: Option<u64>,
    /// Minimum access count to keep (default: 0)
    pub min_access_count: Option<i64>,
    /// If true, show what would be pruned without deleting
    pub dry_run: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ContextArgs {
    /// Task description to find relevant memories for
    pub description: String,
    /// Maximum number of results (default: 5)
    pub limit: Option<usize>,
    /// Project to boost in ranking. Defaults to cwd-derived project ident.
    pub project: Option<String>,
    /// Hard filter: only return memories whose project equals this string.
    pub only: Option<String>,
    /// If true, disable the current-project boost entirely (flat ranking).
    pub no_project_boost: Option<bool>,
    /// Output format: "brief" (default) or "full".
    pub format: Option<String>,
    /// Preview length for brief format (default: 160).
    pub preview_chars: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetArgs {
    /// Memory IDs to fetch.
    pub ids: Vec<String>,
    /// Output format: "full" (default) or "brief".
    pub format: Option<String>,
    /// Preview length for brief format (default: 160).
    pub preview_chars: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MoveArgs {
    /// Move a single memory by ID.
    pub id: Option<String>,
    /// Move all memories whose current project equals this value. Pass an
    /// empty string ("") to target memories with no project.
    pub from: Option<String>,
    /// New project ident. Pass an empty string ("") to clear the project tag.
    pub to: String,
    /// If true, return the would-be-affected memories without writing.
    pub dry_run: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CopyArgs {
    /// Copy a single memory by ID.
    pub id: Option<String>,
    /// Copy all memories whose current project equals this value. Pass an
    /// empty string ("") to target memories with no project.
    pub from: Option<String>,
    /// New project ident for the copies. Pass an empty string ("") to create
    /// copies with no project tag.
    pub to: String,
    /// If true, return the would-be-affected memories without writing.
    pub dry_run: Option<bool>,
}

#[derive(Copy, Clone)]
enum OutputFormat {
    Brief,
    Full,
}

fn parse_format(s: Option<&str>, default: OutputFormat) -> OutputFormat {
    match s.map(|v| v.to_ascii_lowercase()) {
        Some(ref v) if v == "brief" => OutputFormat::Brief,
        Some(ref v) if v == "full" => OutputFormat::Full,
        _ => default,
    }
}

impl MemoryServer {
    pub fn new(config: Config, conn: Connection) -> Self {
        Self {
            tool_router: Self::tool_router(),
            conn: Arc::new(Mutex::new(conn)),
            config: Arc::new(config),
        }
    }

    fn err_str(e: MemoryError) -> String {
        e.to_string()
    }
}

#[tool_router]
impl MemoryServer {
    /// Store a memory with auto-embedding and BM25 indexing. Use this to save important context,
    /// user preferences, project decisions, feedback, or reference information for future retrieval.
    /// The project tag is auto-derived from the server's working directory unless overridden.
    #[tool(name = "memory_store")]
    fn store(&self, Parameters(args): Parameters<StoreArgs>) -> String {
        let tag_list = args
            .tags
            .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

        // Guard: the MCP surface does not yet expose a `scope` parameter,
        // so the only way to land in the global sentinel from MCP would be
        // to pass `project: "__global__"` explicitly — which we reject.
        // When MCP scope support ships this guard moves to "unless the
        // scope argument was `global`" identical to the CLI path.
        if args.project.as_deref() == Some(GLOBAL_PROJECT_IDENT) {
            return format!(
                "{{\"error\": \"`{GLOBAL_PROJECT_IDENT}` is reserved for global-scoped memories. Use the CLI `memory store --scope global` until MCP exposes a scope argument.\"}}"
            );
        }

        let resolved_project = if args.no_project.unwrap_or(false) {
            None
        } else {
            args.project
                .or_else(|| project::project_ident_from_cwd().ok())
        };

        let mut memory = Memory::new(
            args.content.clone(),
            tag_list,
            resolved_project,
            args.agent,
            args.source_file,
            args.memory_type.or(Some("user".to_string())),
        );

        let emb = match embedding::embed_text(&args.content, &self.config.model_cache_dir) {
            Ok(e) => e,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };
        memory.embedding = Some(emb);

        let conn = self.conn.lock().unwrap();
        if let Err(e) = queries::insert_memory(&conn, &memory) {
            return format!("{{\"error\": \"{}\"}}", Self::err_str(e));
        }

        json!({
            "status": "stored",
            "id": memory.id,
            "project": memory.project,
        })
        .to_string()
    }

    /// Hybrid BM25 + vector search across all memories. Combines keyword matching with semantic
    /// similarity. Boosts current-project memories by default while still surfacing cross-project
    /// results as prior-art. Returns a brief preview per hit by default -- call memory_get for
    /// the full content of hits you want to read.
    #[tool(name = "memory_search")]
    fn search(&self, Parameters(args): Parameters<SearchArgs>) -> String {
        let limit = args.limit.unwrap_or(10);
        let format = parse_format(args.format.as_deref(), OutputFormat::Brief);
        let preview_chars = args.preview_chars.unwrap_or(DEFAULT_PREVIEW_CHARS);

        let cwd_project = project::project_ident_from_cwd().ok();
        let boosts = resolve_boosts(
            args.project.as_deref(),
            cwd_project.as_deref(),
            args.no_project_boost.unwrap_or(false),
        );
        let boost_project_owned = boosts.current_project.map(|s| s.to_string());

        let opts = SearchOptions {
            limit,
            current_project: boosts.current_project,
            boost_factor: boosts.project_boost,
            only_project: args.only.as_deref(),
            global_project: boosts.global_project,
            global_boost_factor: boosts.global_boost,
        };

        let conn = self.conn.lock().unwrap();
        let results =
            match search::hybrid_search(&conn, &args.query, opts, &self.config.model_cache_dir) {
                Ok(r) => r,
                Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
            };

        let out = render_ranked_output(
            &results,
            boost_project_owned.as_deref(),
            format,
            preview_chars,
        );
        serde_json::to_string_pretty(&out).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }

    /// Filter memories by project, agent, tags, or memory type. Use for structured retrieval
    /// when you know the category of memory you're looking for.
    #[tool(name = "memory_recall")]
    fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        let limit = args.limit.unwrap_or(10);
        let format = parse_format(args.format.as_deref(), OutputFormat::Brief);
        let preview_chars = args.preview_chars.unwrap_or(DEFAULT_PREVIEW_CHARS);
        let tag_list: Option<Vec<String>> = args
            .tags
            .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

        let cwd_project = project::project_ident_from_cwd().ok();

        let conn = self.conn.lock().unwrap();
        let memories = match queries::list_memories(
            &conn,
            args.project.as_deref(),
            args.agent.as_deref(),
            tag_list.as_deref(),
            args.memory_type.as_deref(),
            limit,
        ) {
            Ok(m) => m,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };

        let out = render_memory_list(&memories, cwd_project.as_deref(), format, preview_chars);
        serde_json::to_string_pretty(&out).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }

    /// Remove memories by ID or by search query. Use with an ID for precise deletion,
    /// or a query to find and remove matching memories.
    #[tool(name = "memory_forget")]
    fn forget(&self, Parameters(args): Parameters<ForgetArgs>) -> String {
        let conn = self.conn.lock().unwrap();

        if let Some(id) = args.id {
            match queries::delete_memory(&conn, &id) {
                Ok(true) => json!({"status": "deleted", "id": id}).to_string(),
                Ok(false) => json!({"status": "not_found", "id": id}).to_string(),
                Err(e) => format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
            }
        } else if let Some(query) = args.query {
            let opts = SearchOptions::new(5);
            let results =
                match search::hybrid_search(&conn, &query, opts, &self.config.model_cache_dir) {
                    Ok(r) => r,
                    Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
                };

            let mut deleted_ids = Vec::new();
            for r in &results {
                if let Ok(true) = queries::delete_memory(&conn, &r.memory.id) {
                    deleted_ids.push(r.memory.id.clone());
                }
            }
            json!({
                "status": "deleted",
                "count": deleted_ids.len(),
                "ids": deleted_ids,
            })
            .to_string()
        } else {
            r#"{"error": "Either 'id' or 'query' must be provided"}"#.to_string()
        }
    }

    /// Decay stale, low-access memories. Removes memories older than max_age_days with
    /// access_count at or below min_access_count. Use dry_run to preview before deleting.
    #[tool(name = "memory_prune")]
    fn prune(&self, Parameters(args): Parameters<PruneArgs>) -> String {
        let max_age = args.max_age_days.unwrap_or(90);
        let min_access = args.min_access_count.unwrap_or(0);
        let dry_run = args.dry_run.unwrap_or(false);

        let conn = self.conn.lock().unwrap();
        let pruned = match queries::prune_memories(&conn, max_age, min_access, dry_run) {
            Ok(p) => p,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };

        json!({
            "status": if dry_run { "dry_run" } else { "pruned" },
            "count": pruned.len(),
            "memories": pruned.iter().map(|m| json!({
                "id": m.id,
                "content": m.content.chars().take(100).collect::<String>(),
                "access_count": m.access_count,
                "updated_at": m.updated_at,
            })).collect::<Vec<_>>(),
        })
        .to_string()
    }

    /// Return the most relevant memories for a given task description, with a current-project
    /// boost so local context wins ties but cross-project memories can still surface as prior-art.
    /// Use at the start of a task. Returns brief previews by default -- call memory_get for full
    /// content on the hits you want to read.
    #[tool(name = "memory_context")]
    fn context(&self, Parameters(args): Parameters<ContextArgs>) -> String {
        let limit = args.limit.unwrap_or(5);
        let format = parse_format(args.format.as_deref(), OutputFormat::Brief);
        let preview_chars = args.preview_chars.unwrap_or(DEFAULT_PREVIEW_CHARS);

        let cwd_project = project::project_ident_from_cwd().ok();
        let boosts = resolve_boosts(
            args.project.as_deref(),
            cwd_project.as_deref(),
            args.no_project_boost.unwrap_or(false),
        );
        let boost_project_owned = boosts.current_project.map(|s| s.to_string());

        let opts = SearchOptions {
            limit,
            current_project: boosts.current_project,
            boost_factor: boosts.project_boost,
            only_project: args.only.as_deref(),
            global_project: boosts.global_project,
            global_boost_factor: boosts.global_boost,
        };

        let conn = self.conn.lock().unwrap();
        let results = match search::hybrid_search(
            &conn,
            &args.description,
            opts,
            &self.config.model_cache_dir,
        ) {
            Ok(r) => r,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };

        let out = render_ranked_output(
            &results,
            boost_project_owned.as_deref(),
            format,
            preview_chars,
        );
        serde_json::to_string_pretty(&out).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }

    /// Fetch full content for one or more memory IDs. Pair with memory_search or memory_context
    /// using the default brief format for a cheap two-stage retrieval: scan lightweight hits,
    /// then pull the full content of just the ones you want to read.
    #[tool(name = "memory_get")]
    fn get(&self, Parameters(args): Parameters<GetArgs>) -> String {
        let format = parse_format(args.format.as_deref(), OutputFormat::Full);
        let preview_chars = args.preview_chars.unwrap_or(DEFAULT_PREVIEW_CHARS);

        let cwd_project = project::project_ident_from_cwd().ok();

        let conn = self.conn.lock().unwrap();
        let mut fetched: Vec<Memory> = Vec::with_capacity(args.ids.len());
        let mut missing: Vec<String> = Vec::new();
        for id in &args.ids {
            match queries::get_memory_by_id(&conn, id) {
                Ok(m) => fetched.push(m),
                Err(MemoryError::NotFound(_)) => missing.push(id.clone()),
                Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
            }
        }
        let hit_ids: Vec<String> = fetched.iter().map(|m| m.id.clone()).collect();
        if let Err(e) = queries::increment_access(&conn, &hit_ids) {
            return format!("{{\"error\": \"{}\"}}", Self::err_str(e));
        }

        let results: Vec<Value> = fetched
            .iter()
            .map(|m| render_memory(m, cwd_project.as_deref(), format, preview_chars))
            .collect();

        json!({
            "results": results,
            "missing": missing,
        })
        .to_string()
    }

    /// Reassign the project ident on one or more memories. Use this to migrate
    /// memories that were tagged under a legacy project name (e.g.
    /// "trading-platform-sre") to the canonical git-remote ident the cwd
    /// resolver returns now (e.g. "github.com/nitecon/SRE.git"). Pass `id` for
    /// a single memory or `from` to bulk-rename every memory currently tagged
    /// with that project. Empty strings on `from`/`to` target or assign a NULL
    /// project. Set `dry_run: true` to preview without writing.
    #[tool(name = "memory_move")]
    fn move_tool(&self, Parameters(args): Parameters<MoveArgs>) -> String {
        // Mirror the CLI guard: reassigning into the global sentinel via
        // `--to __global__` would bypass the `--scope global` contract.
        if args.to == GLOBAL_PROJECT_IDENT {
            return format!(
                "{{\"error\": \"`{GLOBAL_PROJECT_IDENT}` is reserved for global-scoped memories. Use the CLI `memory store --scope global` for new globals, or re-run with a different `to` value.\"}}"
            );
        }
        let to = empty_to_none_owned(&args.to);
        let dry_run = args.dry_run.unwrap_or(false);
        let conn = self.conn.lock().unwrap();
        match run_move(
            &conn,
            args.id.as_deref(),
            args.from.as_deref(),
            to.as_deref(),
            dry_run,
        ) {
            Ok(v) => serde_json::to_string_pretty(&v)
                .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e)),
            Err(e) => format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        }
    }

    /// Duplicate one or more memories under a new project ident. Preserves
    /// content, tags, agent, source_file, memory_type, and the cached
    /// embedding; a new UUID is minted and timestamps reset. Pass `id` for a
    /// single memory or `from` to bulk-copy every memory currently tagged with
    /// that project. Empty strings on `from`/`to` target or assign a NULL
    /// project. Set `dry_run: true` to preview without writing.
    #[tool(name = "memory_copy")]
    fn copy_tool(&self, Parameters(args): Parameters<CopyArgs>) -> String {
        let to = empty_to_none_owned(&args.to);
        let dry_run = args.dry_run.unwrap_or(false);
        let conn = self.conn.lock().unwrap();
        match run_copy(
            &conn,
            args.id.as_deref(),
            args.from.as_deref(),
            to.as_deref(),
            dry_run,
        ) {
            Ok(v) => serde_json::to_string_pretty(&v)
                .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e)),
            Err(e) => format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        }
    }

    /// List distinct project idents with memory counts. Useful for spotting
    /// alias mismatches (e.g. "trading-platform-sre" vs
    /// "github.com/nitecon/SRE.git") before running memory_move. The current
    /// cwd-derived project is marked with is_current_project: true.
    #[tool(name = "memory_projects")]
    fn projects(&self) -> String {
        let cwd_project = project::project_ident_from_cwd().ok();
        let conn = self.conn.lock().unwrap();
        let rows = match queries::list_projects(&conn) {
            Ok(r) => r,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };
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
        serde_json::to_string_pretty(&Value::Object(out))
            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }
}

fn empty_to_none_owned(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
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
            let from_opt = if from.is_empty() { None } else { Some(from) };
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
            "message": "Either 'id' or 'from' must be provided",
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
            let from_opt = if from.is_empty() { None } else { Some(from) };
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
            "message": "Either 'id' or 'from' must be provided",
        })),
    }
}

/// Mirror of `crate::cli::BoostConfig`. The two modules compute boosts
/// the same way; duplicating the struct keeps each surface self-contained
/// without introducing a cross-module coupling that serves nothing else.
struct BoostConfig<'a> {
    current_project: Option<&'a str>,
    global_project: Option<&'a str>,
    project_boost: f32,
    global_boost: f32,
}

fn resolve_boosts<'a>(
    explicit: Option<&'a str>,
    cwd: Option<&'a str>,
    disable: bool,
) -> BoostConfig<'a> {
    if disable {
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
    current_project: Option<&str>,
    format: OutputFormat,
    preview_chars: usize,
) -> Value {
    let total = results.len();
    let global_scope_count = results.iter().filter(|r| r.is_global).count();
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
    }
    out.insert(
        "global_scope_count".into(),
        Value::Number(global_scope_count.into()),
    );
    if let Some(hint) = results_hint(cross_project_count, global_scope_count, total, current_project)
    {
        out.insert("hint".into(), Value::String(hint));
    }
    Value::Object(out)
}

/// See `crate::cli::results_hint` for the full doc; kept in sync verbatim.
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
                    "{cross} of {total} results are cross-project (is_current_project=false). Use those as prior-art or general guidance, not direct context for '{cp}'. Use memory_get for full content."
                )
            });
        }
    }

    if globals > 0 {
        parts.push(format!(
            "{globals} of {total} results are global-scope preferences (apply across all projects). Treat them as directives, not suggestions."
        ));
    } else if current.is_some() {
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

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "Persistent memory system for AI coding agents. \
                 memory_store saves context (auto-tagged with cwd project), \
                 memory_search and memory_context rank with a current-project boost and \
                 return brief previews by default; fetch full content via memory_get. \
                 memory_recall filters by project/agent/tags/type, \
                 memory_forget deletes by id or query, memory_prune cleans stale memories. \
                 memory_projects lists distinct project idents (spot aliases), \
                 memory_move reassigns the project ident on memories (single id or bulk --from/--to), \
                 memory_copy duplicates memories under a new project ident.",
            )
    }
}
