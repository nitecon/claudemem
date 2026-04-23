use std::sync::{Arc, Mutex};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::schemars;
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use rusqlite::Connection;
use serde::Deserialize;

use crate::config::Config;
use crate::db::models::Memory;
use crate::db::queries::{self, ResolvedId};
use crate::embedding;
use crate::error::MemoryError;
use crate::project;
use crate::render;
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
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetArgs {
    /// Specific memory ID to delete (full UUID or 4+ char short prefix)
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
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetArgs {
    /// Memory IDs to fetch (full UUIDs or 4+ char short prefixes).
    pub ids: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MoveArgs {
    /// Move a single memory by ID (full UUID or 4+ char short prefix).
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
    /// Copy a single memory by ID (full UUID or 4+ char short prefix).
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

impl MemoryServer {
    pub fn new(config: Config, conn: Connection) -> Self {
        Self {
            tool_router: Self::tool_router(),
            conn: Arc::new(Mutex::new(conn)),
            config: Arc::new(config),
        }
    }

    /// Format a `MemoryError` as a single light-XML `<result status="error" .../>`
    /// line so all MCP tool responses share the same shape. Kept local rather
    /// than in the render module because "error during tool call" is an
    /// MCP-specific concept.
    fn err_xml(e: MemoryError) -> String {
        render::render_action_result("error", &[("message", e.to_string())])
    }
}

#[tool_router]
impl MemoryServer {
    /// Store a memory with auto-embedding and BM25 indexing. Use this to save important context,
    /// user preferences, project decisions, feedback, or reference information for future retrieval.
    /// The project tag is auto-derived from the server's working directory unless overridden.
    /// Returns a <result status="stored" .../> light-XML line.
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
            return render::render_action_result(
                "error",
                &[(
                    "message",
                    format!(
                        "`{GLOBAL_PROJECT_IDENT}` is reserved for global-scoped memories. \
                         Use the CLI `memory store --scope global` until MCP exposes a scope argument."
                    ),
                )],
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
            Err(e) => return Self::err_xml(e),
        };
        memory.embedding = Some(emb);

        let conn = self.conn.lock().unwrap();
        if let Err(e) = queries::insert_memory(&conn, &memory) {
            return Self::err_xml(e);
        }

        let mut attrs: Vec<(&str, String)> =
            vec![("id", render::short_id(&memory.id).to_string())];
        if let Some(p) = memory.project.as_deref() {
            attrs.push(("project", p.to_string()));
        }
        render::render_action_result("stored", &attrs)
    }

    /// Hybrid BM25 + vector search across all memories. Combines keyword matching with semantic
    /// similarity. Boosts current-project memories by default while still surfacing cross-project
    /// results as prior-art. Returns a light-XML block with <project_memories>,
    /// <general_knowledge>, and <other_projects> sections.
    #[tool(name = "memory_search")]
    fn search(&self, Parameters(args): Parameters<SearchArgs>) -> String {
        let limit = args.limit.unwrap_or(10);

        let cwd_project = project::project_ident_from_cwd().ok();
        let boosts = resolve_boosts(
            args.project.as_deref(),
            cwd_project.as_deref(),
            args.no_project_boost.unwrap_or(false),
        );

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
                Err(e) => return Self::err_xml(e),
            };

        render_ranked_xml(&results, boosts.current_project)
    }

    /// Filter memories by project, agent, tags, or memory type. Use for structured retrieval
    /// when you know the category of memory you're looking for. Returns a <memories count=".."/>
    /// light-XML block.
    #[tool(name = "memory_recall")]
    fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        let limit = args.limit.unwrap_or(10);
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
            Err(e) => return Self::err_xml(e),
        };

        render::render_memory_list(&memories, cwd_project.as_deref())
    }

    /// Remove memories by ID or by search query. Use with an ID (full UUID or short prefix)
    /// for precise deletion, or a query to find and remove matching memories. Ambiguous short
    /// prefixes return an <ambiguous/> block instead of deleting.
    #[tool(name = "memory_forget")]
    fn forget(&self, Parameters(args): Parameters<ForgetArgs>) -> String {
        let conn = self.conn.lock().unwrap();

        if let Some(id) = args.id {
            match queries::resolve_id_prefix(&conn, &id) {
                Ok(ResolvedId::Exact(full_id)) => match queries::delete_memory(&conn, &full_id) {
                    Ok(true) => render::render_action_result(
                        "forgot",
                        &[("id", render::short_id(&full_id).to_string())],
                    ),
                    Ok(false) => render::render_action_result(
                        "not_found",
                        &[("id", render::short_id(&full_id).to_string())],
                    ),
                    Err(e) => Self::err_xml(e),
                },
                Ok(ResolvedId::Ambiguous(cands)) => render::render_ambiguous(&id, &cands),
                Ok(ResolvedId::NotFound) => {
                    render::render_action_result("not_found", &[("id", id)])
                }
                Err(e) => Self::err_xml(e),
            }
        } else if let Some(query) = args.query {
            let opts = SearchOptions::new(5);
            let results =
                match search::hybrid_search(&conn, &query, opts, &self.config.model_cache_dir) {
                    Ok(r) => r,
                    Err(e) => return Self::err_xml(e),
                };

            if results.is_empty() {
                return render::render_action_result("no_matches", &[]);
            }

            let mut deleted = 0usize;
            for r in &results {
                if let Ok(true) = queries::delete_memory(&conn, &r.memory.id) {
                    deleted += 1;
                }
            }
            render::render_action_result("forgot", &[("count", deleted.to_string())])
        } else {
            render::render_action_result(
                "error",
                &[("message", "Either 'id' or 'query' must be provided".to_string())],
            )
        }
    }

    /// Decay stale, low-access memories. Removes memories older than max_age_days with
    /// access_count at or below min_access_count. Use dry_run to preview before deleting.
    /// Returns a <result status="pruned" count=".."/> line.
    #[tool(name = "memory_prune")]
    fn prune(&self, Parameters(args): Parameters<PruneArgs>) -> String {
        let max_age = args.max_age_days.unwrap_or(90);
        let min_access = args.min_access_count.unwrap_or(0);
        let dry_run = args.dry_run.unwrap_or(false);

        let conn = self.conn.lock().unwrap();
        let pruned = match queries::prune_memories(&conn, max_age, min_access, dry_run) {
            Ok(p) => p,
            Err(e) => return Self::err_xml(e),
        };

        let status = if dry_run { "dry_run" } else { "pruned" };
        render::render_action_result(status, &[("count", pruned.len().to_string())])
    }

    /// Return the most relevant memories for a given task description, with a current-project
    /// boost so local context wins ties but cross-project memories can still surface as prior-art.
    /// Use at the start of a task. Returns a light-XML block grouped into <project_memories>,
    /// <general_knowledge>, and <other_projects> sections.
    #[tool(name = "memory_context")]
    fn context(&self, Parameters(args): Parameters<ContextArgs>) -> String {
        let limit = args.limit.unwrap_or(5);

        let cwd_project = project::project_ident_from_cwd().ok();
        let boosts = resolve_boosts(
            args.project.as_deref(),
            cwd_project.as_deref(),
            args.no_project_boost.unwrap_or(false),
        );

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
            Err(e) => return Self::err_xml(e),
        };

        render_ranked_xml(&results, boosts.current_project)
    }

    /// Fetch full content for one or more memory IDs. Each ID can be a full UUID or a 4+ char
    /// short prefix; ambiguous prefixes return an <ambiguous/> block. Pair with memory_search
    /// or memory_context for a two-stage retrieval: scan previews, then pull full content on
    /// the handful you want to read.
    #[tool(name = "memory_get")]
    fn get(&self, Parameters(args): Parameters<GetArgs>) -> String {
        let conn = self.conn.lock().unwrap();
        let mut fetched: Vec<Memory> = Vec::with_capacity(args.ids.len());
        let mut out_parts: Vec<String> = Vec::with_capacity(args.ids.len());

        for id in &args.ids {
            match queries::resolve_id_prefix(&conn, id) {
                Ok(ResolvedId::Exact(full_id)) => {
                    match queries::get_memory_by_id(&conn, &full_id) {
                        Ok(m) => {
                            out_parts.push(render::render_memory(&m));
                            fetched.push(m);
                        }
                        Err(MemoryError::NotFound(_)) => {
                            out_parts.push(render::render_action_result(
                                "not_found",
                                &[("id", id.clone())],
                            ));
                        }
                        Err(e) => return Self::err_xml(e),
                    }
                }
                Ok(ResolvedId::Ambiguous(cands)) => {
                    out_parts.push(render::render_ambiguous(id, &cands));
                }
                Ok(ResolvedId::NotFound) => {
                    out_parts.push(render::render_action_result(
                        "not_found",
                        &[("id", id.clone())],
                    ));
                }
                Err(e) => return Self::err_xml(e),
            }
        }

        let hit_ids: Vec<String> = fetched.iter().map(|m| m.id.clone()).collect();
        if let Err(e) = queries::increment_access(&conn, &hit_ids) {
            return Self::err_xml(e);
        }

        out_parts.join("\n")
    }

    /// Reassign the project ident on one or more memories. Use this to migrate memories
    /// tagged under a legacy project name to the canonical git-remote ident the cwd resolver
    /// returns now. Pass `id` for a single memory (full UUID or short prefix) or `from` to
    /// bulk-rename. Empty strings on `from`/`to` target or assign a NULL project. Set
    /// `dry_run: true` to preview without writing.
    #[tool(name = "memory_move")]
    fn move_tool(&self, Parameters(args): Parameters<MoveArgs>) -> String {
        // Mirror the CLI guard: reassigning into the global sentinel via
        // `to: "__global__"` would bypass the `--scope global` contract.
        if args.to == GLOBAL_PROJECT_IDENT {
            return render::render_action_result(
                "error",
                &[(
                    "message",
                    format!(
                        "`{GLOBAL_PROJECT_IDENT}` is reserved for global-scoped memories. \
                         Use the CLI `memory store --scope global` for new globals, or \
                         re-run with a different `to` value."
                    ),
                )],
            );
        }
        let to = empty_to_none_owned(&args.to);
        let dry_run = args.dry_run.unwrap_or(false);
        let conn = self.conn.lock().unwrap();

        // Resolve the id (if any) through the prefix resolver so MCP callers
        // can pass short UUIDs just like the CLI.
        let resolved_id = match args.id.as_deref() {
            None => None,
            Some(raw) => match queries::resolve_id_prefix(&conn, raw) {
                Ok(ResolvedId::Exact(full)) => Some(full),
                Ok(ResolvedId::Ambiguous(cands)) => {
                    return render::render_ambiguous(raw, &cands);
                }
                Ok(ResolvedId::NotFound) => {
                    return render::render_action_result(
                        "not_found",
                        &[("id", raw.to_string())],
                    );
                }
                Err(e) => return Self::err_xml(e),
            },
        };

        run_move(
            &conn,
            resolved_id.as_deref(),
            args.from.as_deref(),
            to.as_deref(),
            dry_run,
        )
    }

    /// Duplicate one or more memories under a new project ident. Preserves content, tags,
    /// agent, source_file, memory_type, and the cached embedding; a new UUID is minted and
    /// timestamps reset. Pass `id` for a single memory (full UUID or short prefix) or `from`
    /// to bulk-copy. Empty strings on `from`/`to` target or assign a NULL project. Set
    /// `dry_run: true` to preview without writing.
    #[tool(name = "memory_copy")]
    fn copy_tool(&self, Parameters(args): Parameters<CopyArgs>) -> String {
        let to = empty_to_none_owned(&args.to);
        let dry_run = args.dry_run.unwrap_or(false);
        let conn = self.conn.lock().unwrap();

        let resolved_id = match args.id.as_deref() {
            None => None,
            Some(raw) => match queries::resolve_id_prefix(&conn, raw) {
                Ok(ResolvedId::Exact(full)) => Some(full),
                Ok(ResolvedId::Ambiguous(cands)) => {
                    return render::render_ambiguous(raw, &cands);
                }
                Ok(ResolvedId::NotFound) => {
                    return render::render_action_result(
                        "not_found",
                        &[("id", raw.to_string())],
                    );
                }
                Err(e) => return Self::err_xml(e),
            },
        };

        run_copy(
            &conn,
            resolved_id.as_deref(),
            args.from.as_deref(),
            to.as_deref(),
            dry_run,
        )
    }

    /// List distinct project idents with memory counts. Useful for spotting alias mismatches
    /// before running memory_move. The current cwd-derived project is marked with `*`.
    /// Returns a <projects count=".."/> light-XML block.
    #[tool(name = "memory_projects")]
    fn projects(&self) -> String {
        let cwd_project = project::project_ident_from_cwd().ok();
        let conn = self.conn.lock().unwrap();
        let rows = match queries::list_projects(&conn) {
            Ok(r) => r,
            Err(e) => return Self::err_xml(e),
        };
        render::render_projects(&rows, cwd_project.as_deref())
    }
}

fn empty_to_none_owned(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Execute a move operation and return a light-XML string. Shared between the
/// MCP move tool and the would-be CLI path; duplicated here to keep the MCP
/// module compilable without pulling in cli internals.
fn run_move(
    conn: &Connection,
    id: Option<&str>,
    from: Option<&str>,
    to: Option<&str>,
    dry_run: bool,
) -> String {
    match (id, from) {
        (Some(id), _) => {
            if dry_run {
                match queries::get_memory_by_id(conn, id) {
                    Ok(mem) => {
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
                        render::render_action_result("dry_run", &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            } else {
                match queries::move_memory_by_id(conn, id, to) {
                    Ok(changed) => {
                        let status = if changed { "moved" } else { "not_found" };
                        let mut attrs: Vec<(&str, String)> =
                            vec![("id", render::short_id(id).to_string())];
                        if let Some(t) = to {
                            attrs.push(("to_project", t.to_string()));
                        }
                        render::render_action_result(status, &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            }
        }
        (None, Some(from)) => {
            let from_opt = if from.is_empty() { None } else { Some(from) };
            if dry_run {
                match queries::list_memories_by_project(conn, from_opt) {
                    Ok(mems) => {
                        let mut attrs: Vec<(&str, String)> = vec![
                            ("would_move", mems.len().to_string()),
                            ("from_project", from_opt.unwrap_or("").to_string()),
                        ];
                        if let Some(t) = to {
                            attrs.push(("to_project", t.to_string()));
                        }
                        render::render_action_result("dry_run", &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            } else {
                match queries::move_memories_by_project(conn, from_opt, to) {
                    Ok(count) => {
                        let mut attrs: Vec<(&str, String)> = vec![
                            ("count", count.to_string()),
                            ("from_project", from_opt.unwrap_or("").to_string()),
                        ];
                        if let Some(t) = to {
                            attrs.push(("to_project", t.to_string()));
                        }
                        render::render_action_result("moved", &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            }
        }
        (None, None) => render::render_action_result(
            "error",
            &[("message", "Either 'id' or 'from' must be provided".to_string())],
        ),
    }
}

/// Execute a copy operation and return a light-XML string. See [`run_move`] for
/// the contract; this is its copy-flavored twin.
fn run_copy(
    conn: &Connection,
    id: Option<&str>,
    from: Option<&str>,
    to: Option<&str>,
    dry_run: bool,
) -> String {
    match (id, from) {
        (Some(id), _) => {
            if dry_run {
                match queries::get_memory_by_id(conn, id) {
                    Ok(mem) => {
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
                        render::render_action_result("dry_run", &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            } else {
                match queries::copy_memory_by_id(conn, id, to) {
                    Ok(new_id) => {
                        let mut attrs: Vec<(&str, String)> = vec![
                            ("source_id", render::short_id(id).to_string()),
                            ("new_id", render::short_id(&new_id).to_string()),
                        ];
                        if let Some(t) = to {
                            attrs.push(("to_project", t.to_string()));
                        }
                        render::render_action_result("copied", &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            }
        }
        (None, Some(from)) => {
            let from_opt = if from.is_empty() { None } else { Some(from) };
            if dry_run {
                match queries::list_memories_by_project(conn, from_opt) {
                    Ok(mems) => {
                        let mut attrs: Vec<(&str, String)> = vec![
                            ("would_copy", mems.len().to_string()),
                            ("from_project", from_opt.unwrap_or("").to_string()),
                        ];
                        if let Some(t) = to {
                            attrs.push(("to_project", t.to_string()));
                        }
                        render::render_action_result("dry_run", &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            } else {
                match queries::copy_memories_by_project(conn, from_opt, to) {
                    Ok(new_ids) => {
                        let mut attrs: Vec<(&str, String)> = vec![
                            ("count", new_ids.len().to_string()),
                            ("from_project", from_opt.unwrap_or("").to_string()),
                        ];
                        if let Some(t) = to {
                            attrs.push(("to_project", t.to_string()));
                        }
                        render::render_action_result("copied", &attrs)
                    }
                    Err(e) => MemoryServer::err_xml(e),
                }
            }
        }
        (None, None) => render::render_action_result(
            "error",
            &[("message", "Either 'id' or 'from' must be provided".to_string())],
        ),
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

/// Render a ranked result set as light-XML with the same reflection hint the
/// CLI emits. Factored out so `search` and `context` share the rendering path.
fn render_ranked_xml(results: &[SearchResult], current_project: Option<&str>) -> String {
    let total = results.len();
    let globals = results.iter().filter(|r| r.is_global).count();
    let cross = if current_project.is_some() {
        results.iter().filter(|r| !r.is_current_project).count()
    } else {
        0
    };
    let hint = results_hint(cross, globals, total, current_project);
    let rendered = render::render_search_results(results, current_project, hint.as_deref());
    if rendered.is_empty() {
        "<results count=\"0\"/>".to_string()
    } else {
        rendered
    }
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

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "Persistent memory system for AI coding agents. All tools return light-XML \
                 text (sectioned tags with numbered content lines — no JSON). \
                 memory_store saves context (auto-tagged with cwd project), \
                 memory_search and memory_context rank with a current-project boost and \
                 return grouped <project_memories>/<general_knowledge>/<other_projects> \
                 sections; fetch full content via memory_get. \
                 memory_recall filters by project/agent/tags/type, \
                 memory_forget deletes by id (short prefix supported) or query, \
                 memory_prune cleans stale memories. memory_projects lists distinct project \
                 idents (spot aliases), memory_move reassigns the project ident on memories \
                 (single id or bulk from/to), memory_copy duplicates memories under a new ident.",
            )
    }
}
