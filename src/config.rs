use std::path::PathBuf;

use crate::error::MemoryError;

#[derive(Debug, Clone)]
pub struct Config {
    pub data_dir: PathBuf,
    pub db_path: PathBuf,
    pub model_cache_dir: PathBuf,
}

impl Config {
    /// Resolve the data directory with the following priority:
    ///
    /// 1. `AGENT_MEMORY_DIR` environment variable (explicit override)
    /// 2. `~/.agentic/` if `~/.agentic/memory.db` already exists (user-local)
    /// 3. `/opt/agentic/` as the global default (Linux/macOS)
    ///    `%USERPROFILE%\.agentic\` on Windows
    pub fn load() -> Result<Self, MemoryError> {
        let data_dir = if let Ok(dir) = std::env::var("AGENT_MEMORY_DIR") {
            PathBuf::from(dir)
        } else if let Some(user_dir) = Self::user_local_dir() {
            if user_dir.join("memory.db").exists() {
                user_dir
            } else {
                Self::global_dir().unwrap_or(user_dir)
            }
        } else {
            Self::global_dir()
                .ok_or_else(|| MemoryError::Config("Could not determine data directory".into()))?
        };

        Ok(Self {
            db_path: data_dir.join("memory.db"),
            model_cache_dir: data_dir.join("models"),
            data_dir,
        })
    }

    /// User-local directory: ~/.agentic/
    fn user_local_dir() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".agentic"))
    }

    /// Global directory: /opt/agentic/ on Unix, %USERPROFILE%\.agentic\ on Windows
    fn global_dir() -> Option<PathBuf> {
        if cfg!(windows) {
            dirs::home_dir().map(|h| h.join(".agentic"))
        } else {
            Some(PathBuf::from("/opt/agentic"))
        }
    }

    pub fn ensure_dirs(&self) -> Result<(), MemoryError> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.model_cache_dir)?;
        Ok(())
    }
}
