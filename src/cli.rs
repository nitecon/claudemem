use clap::Parser;
use rusqlite::Connection;

use crate::config::Config;
use crate::db::models::Memory;
use crate::db::queries;
use crate::embedding;
use crate::error::MemoryError;
use crate::search;

#[derive(Parser)]
#[command(name = "claude-memory", about = "Persistent hybrid-search memory system for Claude Code")]
pub enum Cli {
    /// Save a memory with auto-embedding and BM25 indexing
    Store {
        /// Memory content text
        content: String,
        /// Comma-separated tags
        #[arg(short, long)]
        tags: Option<String>,
        /// Project identifier
        #[arg(short, long)]
        project: Option<String>,
        /// Agent identifier
        #[arg(short, long)]
        agent: Option<String>,
        /// Source file path
        #[arg(short = 'f', long)]
        source_file: Option<String>,
        /// Memory type: user, feedback, project, reference
        #[arg(short = 'm', long, default_value = "user")]
        memory_type: String,
    },
    /// Hybrid BM25 + vector search
    Search {
        /// Search query
        query: String,
        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,
    },
    /// Filter memories by project/agent/tags
    Recall {
        /// Filter by project
        #[arg(short, long)]
        project: Option<String>,
        /// Filter by agent
        #[arg(short, long)]
        agent: Option<String>,
        /// Comma-separated tags to filter by
        #[arg(short, long)]
        tags: Option<String>,
        /// Filter by memory type
        #[arg(short = 'm', long)]
        memory_type: Option<String>,
        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,
    },
    /// Remove memories by ID or search
    Forget {
        /// Memory ID to remove
        #[arg(short, long)]
        id: Option<String>,
        /// Search query to find and remove memories
        #[arg(short, long)]
        query: Option<String>,
    },
    /// Decay stale/low-access memories
    Prune {
        /// Maximum age in days before pruning
        #[arg(short, long, default_value = "90")]
        max_age_days: u64,
        /// Minimum access count to keep
        #[arg(short = 'c', long, default_value = "0")]
        min_access_count: i64,
        /// Show what would be pruned without deleting
        #[arg(long)]
        dry_run: bool,
    },
    /// Return top-K relevant memories for a task
    Context {
        /// Task description
        description: String,
        /// Number of results
        #[arg(short = 'k', long, default_value = "5")]
        limit: usize,
        /// Filter by project
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Start MCP stdio server
    Serve,
}

pub fn execute(
    cmd: Cli,
    config: Config,
    conn: &Connection,
) -> Result<(), MemoryError> {
    match cmd {
        Cli::Store {
            content,
            tags,
            project,
            agent,
            source_file,
            memory_type,
        } => {
            let tag_list = tags.map(|t| t.split(',').map(|s| s.trim().to_string()).collect());
            let mut memory = Memory::new(
                content.clone(),
                tag_list,
                project,
                agent,
                source_file,
                Some(memory_type),
            );

            // Generate embedding
            let emb = embedding::embed_text(&content, &config.model_cache_dir)?;
            memory.embedding = Some(emb);

            // Store in SQLite (FTS5 triggers handle indexing)
            queries::insert_memory(conn, &memory)?;

            let output = serde_json::json!({
                "status": "stored",
                "id": memory.id,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Search { query, limit } => {
            let results = search::hybrid_search(conn, &query, limit, &config.model_cache_dir)?;

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
                        "created_at": r.memory.created_at,
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Recall {
            project,
            agent,
            tags,
            memory_type,
            limit,
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

            let output: Vec<_> = memories.iter().map(|m| {
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
            }).collect();
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
                let results =
                    search::hybrid_search(conn, &query, 5, &config.model_cache_dir)?;
                if results.is_empty() {
                    println!(r#"{{"status": "no_matches"}}"#);
                } else {
                    let mut deleted_ids = Vec::new();
                    for r in &results {
                        if queries::delete_memory(conn, &r.memory.id)? {
                            deleted_ids.push(r.memory.id.clone());
                        }
                    }
                    let output = serde_json::json!({
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

            let output = serde_json::json!({
                "status": if dry_run { "dry_run" } else { "pruned" },
                "count": pruned.len(),
                "memories": pruned.iter().map(|m| serde_json::json!({
                    "id": m.id,
                    "content": m.content.chars().take(100).collect::<String>(),
                    "access_count": m.access_count,
                    "updated_at": m.updated_at,
                })).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Context {
            description,
            limit,
            project: _project,
        } => {
            // Context is essentially search with task-oriented framing
            let results =
                search::hybrid_search(conn, &description, limit, &config.model_cache_dir)?;

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
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Cli::Serve => {
            unreachable!("Serve is handled in main.rs");
        }
    }
    Ok(())
}
