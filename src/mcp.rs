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
use crate::db::queries;
use crate::embedding;
use crate::error::MemoryError;
use crate::search;

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
    /// Project identifier for scoping
    pub project: Option<String>,
    /// Agent identifier (e.g. "unreal-plugin-developer")
    pub agent: Option<String>,
    /// Source file path this memory relates to
    pub source_file: Option<String>,
    /// Memory type: user, feedback, project, reference
    pub memory_type: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    /// Search query (natural language or keywords)
    pub query: String,
    /// Maximum number of results to return (default: 10)
    pub limit: Option<usize>,
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
    /// Filter by project identifier
    pub project: Option<String>,
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
    #[tool(name = "memory_store")]
    fn store(&self, Parameters(args): Parameters<StoreArgs>) -> String {
        let tag_list = args
            .tags
            .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

        let mut memory = Memory::new(
            args.content.clone(),
            tag_list,
            args.project,
            args.agent,
            args.source_file,
            args.memory_type.or(Some("user".to_string())),
        );

        let emb = match embedding::embed_text(&args.content, &self.config.model_cache_dir) {
            Ok(e) => e,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };
        memory.embedding = Some(emb);

        // Insert into SQLite (FTS5 triggers handle full-text indexing)
        let conn = self.conn.lock().unwrap();
        if let Err(e) = queries::insert_memory(&conn, &memory) {
            return format!("{{\"error\": \"{}\"}}", Self::err_str(e));
        }

        serde_json::json!({
            "status": "stored",
            "id": memory.id,
        })
        .to_string()
    }

    /// Hybrid BM25 + vector search across all memories. Combines keyword matching with semantic
    /// similarity for best results. Use natural language queries or specific keywords.
    #[tool(name = "memory_search")]
    fn search(&self, Parameters(args): Parameters<SearchArgs>) -> String {
        let limit = args.limit.unwrap_or(10);
        let conn = self.conn.lock().unwrap();

        let results = match search::hybrid_search(
            &conn,
            &args.query,
            limit,
            &self.config.model_cache_dir,
        ) {
            Ok(r) => r,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };

        let output: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "tags": r.memory.tags,
                    "project": r.memory.project,
                    "agent": r.memory.agent,
                    "memory_type": r.memory.memory_type,
                    "score": r.rank_info.score,
                    "bm25_rank": r.rank_info.bm25_rank,
                    "vector_rank": r.rank_info.vector_rank,
                    "access_count": r.memory.access_count,
                })
            })
            .collect();

        serde_json::to_string_pretty(&output).unwrap_or_else(|e| {
            format!("{{\"error\": \"{}\"}}", e)
        })
    }

    /// Filter memories by project, agent, tags, or memory type. Use for structured retrieval
    /// when you know the category of memory you're looking for.
    #[tool(name = "memory_recall")]
    fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        let limit = args.limit.unwrap_or(10);
        let tag_list: Option<Vec<String>> = args
            .tags
            .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

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

        let output: Vec<_> = memories
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "content": m.content,
                    "tags": m.tags,
                    "project": m.project,
                    "agent": m.agent,
                    "memory_type": m.memory_type,
                    "access_count": m.access_count,
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                })
            })
            .collect();

        serde_json::to_string_pretty(&output).unwrap_or_else(|e| {
            format!("{{\"error\": \"{}\"}}", e)
        })
    }

    /// Remove memories by ID or by search query. Use with an ID for precise deletion,
    /// or a query to find and remove matching memories.
    #[tool(name = "memory_forget")]
    fn forget(&self, Parameters(args): Parameters<ForgetArgs>) -> String {
        let conn = self.conn.lock().unwrap();

        if let Some(id) = args.id {
            match queries::delete_memory(&conn, &id) {
                Ok(true) => {
                    serde_json::json!({"status": "deleted", "id": id}).to_string()
                }
                Ok(false) => {
                    serde_json::json!({"status": "not_found", "id": id}).to_string()
                }
                Err(e) => format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
            }
        } else if let Some(query) = args.query {
            let results = match search::hybrid_search(
                &conn,
                &query,
                5,
                &self.config.model_cache_dir,
            ) {
                Ok(r) => r,
                Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
            };

            let mut deleted_ids = Vec::new();
            for r in &results {
                if let Ok(true) = queries::delete_memory(&conn, &r.memory.id) {
                    deleted_ids.push(r.memory.id.clone());
                }
            }
            serde_json::json!({
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

        serde_json::json!({
            "status": if dry_run { "dry_run" } else { "pruned" },
            "count": pruned.len(),
            "memories": pruned.iter().map(|m| serde_json::json!({
                "id": m.id,
                "content": m.content.chars().take(100).collect::<String>(),
                "access_count": m.access_count,
                "updated_at": m.updated_at,
            })).collect::<Vec<_>>(),
        })
        .to_string()
    }

    /// Return the most relevant memories for a given task description. Use at the start of a
    /// task to load relevant context from past conversations and decisions.
    #[tool(name = "memory_context")]
    fn context(&self, Parameters(args): Parameters<ContextArgs>) -> String {
        let limit = args.limit.unwrap_or(5);
        let conn = self.conn.lock().unwrap();

        let results = match search::hybrid_search(
            &conn,
            &args.description,
            limit,
            &self.config.model_cache_dir,
        ) {
            Ok(r) => r,
            Err(e) => return format!("{{\"error\": \"{}\"}}", Self::err_str(e)),
        };

        let output: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "content": r.memory.content,
                    "tags": r.memory.tags,
                    "project": r.memory.project,
                    "agent": r.memory.agent,
                    "memory_type": r.memory.memory_type,
                    "relevance_score": r.rank_info.score,
                })
            })
            .collect();

        serde_json::to_string_pretty(&output).unwrap_or_else(|e| {
            format!("{{\"error\": \"{}\"}}", e)
        })
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "Persistent memory system for Claude Code. \
                 Use memory_store to save important context, \
                 memory_search for hybrid retrieval, \
                 memory_context for task-relevant memories, \
                 memory_recall for filtered retrieval, \
                 memory_forget to remove memories, \
                 memory_prune to clean up stale memories.",
            )
    }
}
