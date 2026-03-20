use std::path::PathBuf;

use crate::error::MemoryError;

#[derive(Debug, Clone)]
pub struct Config {
    pub data_dir: PathBuf,
    pub db_path: PathBuf,
    pub model_cache_dir: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self, MemoryError> {
        let data_dir = if let Ok(dir) = std::env::var("CLAUDE_MEMORY_DIR") {
            PathBuf::from(dir)
        } else {
            let home = dirs::home_dir()
                .ok_or_else(|| MemoryError::Config("Could not determine home directory".into()))?;
            home.join(".claude").join("memory")
        };

        Ok(Self {
            db_path: data_dir.join("memory.db"),
            model_cache_dir: data_dir.join("models"),
            data_dir,
        })
    }

    pub fn ensure_dirs(&self) -> Result<(), MemoryError> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.model_cache_dir)?;
        Ok(())
    }
}
