mod cli;
mod config;
mod db;
mod embedding;
mod error;
mod mcp;
mod search;

use clap::Parser;
use rmcp::ServiceExt;
use tracing_subscriber::EnvFilter;

use crate::cli::Cli;
use crate::config::Config;
use crate::db::open_database;
use crate::search::bm25::open_or_create_index;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli {
        Cli::Serve => {
            // MCP mode: stderr logging only, stdout is JSON-RPC transport
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .init();

            let config = Config::load()?;
            config.ensure_dirs()?;
            let conn = open_database(&config.db_path)?;
            let index = open_or_create_index(&config.tantivy_dir)?;

            let server = mcp::MemoryServer::new(config, conn, index);

            tracing::info!("Starting claude-memory MCP server");

            let service = server
                .serve(rmcp::transport::io::stdio())
                .await?;
            service.waiting().await?;
        }
        other => {
            // CLI mode: stderr for logs, stdout for results
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .with_writer(std::io::stderr)
                .init();

            let config = Config::load()?;
            config.ensure_dirs()?;
            let conn = open_database(&config.db_path)?;
            let index = open_or_create_index(&config.tantivy_dir)?;

            cli::execute(other, config, &conn, &index)?;
        }
    }

    Ok(())
}
