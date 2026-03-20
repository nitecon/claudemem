use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Search index error: {0}")]
    SearchIndex(#[from] tantivy::TantivyError),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Memory not found: {0}")]
    NotFound(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
