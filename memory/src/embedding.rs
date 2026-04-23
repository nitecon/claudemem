use std::path::Path;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::MemoryError;

static EMBEDDING_MODEL: Mutex<Option<TextEmbedding>> = Mutex::new(None);

pub fn get_or_init_model(cache_dir: &Path) -> Result<(), MemoryError> {
    let mut guard = EMBEDDING_MODEL.lock().unwrap();
    if guard.is_none() {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_cache_dir(cache_dir.to_path_buf())
                .with_show_download_progress(true),
        )
        .map_err(|e| MemoryError::Embedding(e.to_string()))?;
        *guard = Some(model);
    }
    Ok(())
}

pub fn embed_text(text: &str, cache_dir: &Path) -> Result<Vec<f32>, MemoryError> {
    get_or_init_model(cache_dir)?;
    let guard = EMBEDDING_MODEL.lock().unwrap();
    let model = guard.as_ref().unwrap();
    let embeddings = model
        .embed(vec![text], None)
        .map_err(|e| MemoryError::Embedding(e.to_string()))?;
    embeddings
        .into_iter()
        .next()
        .ok_or_else(|| MemoryError::Embedding("No embedding returned".into()))
}
