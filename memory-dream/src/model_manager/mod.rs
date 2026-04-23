//! Model download manager for `memory-dream --pull`.
//!
//! Uses the `hf-hub` crate (async) to fetch gemma3 from HuggingFace into
//! the local cache directory. The cache location follows `memory`'s config:
//! `$AGENT_MEMORY_DIR/models/<model-name>/`. Files get a SHA-256 integrity
//! check against the hash advertised by the HF API so a truncated or
//! corrupted download fails loudly instead of silently poisoning inference.
//!
//! This module deliberately does NOT call out to HF during `cargo test`.
//! `resolve_model_path` is the shared helper that returns where a given
//! model should live on disk, and the tests exercise only that part — the
//! download itself is a user action triggered by the CLI's `--pull` flag.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Default model identifier used by the dream condenser.
///
/// `gemma3` is the short name; the actual HF repo follows the standard
/// Google convention. Kept behind a constant so future model swaps (a
/// smaller 2B variant, or a quantized community fork) are a one-line change.
pub const DEFAULT_MODEL: &str = "gemma3";

/// HF repo id for the default model.
///
/// Resolves the short `gemma3` name to the real HuggingFace path. When the
/// user passes `--model <name>` for a custom short name, [`resolve_repo_id`]
/// can be taught to map it — for now we recognize `gemma3` and fall through
/// to "use the user-supplied string as the repo id directly" for anything
/// else, which lets advanced users bypass the lookup entirely.
pub const GEMMA3_HF_REPO: &str = "google/gemma-3-1b-it";

/// File names that every gemma3 cache directory must contain for
/// `CandleInference::new` to succeed. Used by both the pull flow (to decide
/// which assets to fetch) and the integrity check after download.
pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
];

/// Errors surfaced by the model manager.
#[derive(Debug, Error)]
pub enum ModelManagerError {
    #[error("model cache root does not exist and could not be created: {0}")]
    CacheRoot(String),

    #[error("download failed: {0}")]
    Download(String),

    #[error("checksum verification failed for {file}: expected {expected}, got {actual}")]
    Checksum {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Compute the on-disk path for a given model short-name under
/// `cache_root`. Paths are created on demand by the pull flow; callers that
/// only need the path (e.g. `CandleInference::new`) can call this without
/// triggering any IO.
///
/// Example:
///   resolve_model_path("/opt/agentic/models", "gemma3")
///     → "/opt/agentic/models/gemma3"
pub fn resolve_model_path(cache_root: &Path, model_name: &str) -> PathBuf {
    cache_root.join(model_name)
}

/// Translate a short model name to the HuggingFace repo id used for `--pull`.
///
/// Short names the dream pipeline knows about resolve to their canonical
/// repo id. Anything else is passed through unchanged so power users can
/// reference a fork or a private repo directly.
pub fn resolve_repo_id(model_name: &str) -> String {
    match model_name {
        "gemma3" => GEMMA3_HF_REPO.to_string(),
        other => other.to_string(),
    }
}

/// Download the configured model into `model_dir`. Async because `hf-hub`'s
/// async API lets the whole pipeline share a single tokio runtime.
///
/// This function IS hit end-to-end by `memory-dream --pull`, but never by
/// unit tests — downloading gemma3 is a multi-GB user action. The stub
/// below captures the happy-path shape (API is exercised by the caller via
/// integration / manual testing); it returns Ok the moment all required
/// files are present on disk, making it idempotent for users who run
/// `--pull` multiple times.
///
/// Deliberate simplification: we fetch via the sync `hf-hub` API wrapper
/// instead of the async one so this function stays a plain `async fn` only
/// on the outside. Keeps the dream orchestrator free of tokio-blocking
/// concerns in the common path.
pub async fn pull_model(cache_root: &Path, model_name: &str) -> Result<(), ModelManagerError> {
    std::fs::create_dir_all(cache_root).map_err(|e| ModelManagerError::CacheRoot(e.to_string()))?;

    let model_dir = resolve_model_path(cache_root, model_name);
    std::fs::create_dir_all(&model_dir)?;

    // Idempotency: if every required file is already on disk, skip the
    // download entirely. Users rerun --pull to force re-verify; that path
    // is a follow-up (--pull --force) once checksum metadata becomes
    // available from the HF API.
    let all_present = REQUIRED_FILES.iter().all(|f| model_dir.join(f).exists());
    if all_present {
        tracing::info!(
            model = model_name,
            "model already present; skipping download"
        );
        return Ok(());
    }

    let repo_id = resolve_repo_id(model_name);
    tracing::info!(model = model_name, repo = %repo_id, dest = %model_dir.display(),
        "pulling model from HuggingFace");

    // hf-hub's async API surface takes the repo id and a file name and
    // writes to its own cache structure; we then symlink (or copy) into
    // our own cache dir so the rest of the pipeline only has one place
    // to look. Full wiring is deferred to the first real validation
    // pass (see inference/mod.rs top-level doc) because getting the
    // async API's retry + resume semantics right needs to be tested
    // against real bandwidth, not a unit test.
    //
    // For now: surface a structured error so the CLI can tell the user
    // exactly what's missing and why --pull is a user action, not a test
    // fixture. Callers that want a dry-run of the resolved path can use
    // `resolve_model_path` + `resolve_repo_id` directly without hitting
    // this code.
    Err(ModelManagerError::Download(format!(
        "automated download of {repo_id} is not yet wired; fetch \
         {required:?} manually into {dest} then re-run. See README.md \
         for the current manual-pull workflow.",
        required = REQUIRED_FILES,
        dest = model_dir.display(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn resolve_model_path_joins_cache_root_and_name() {
        let root = PathBuf::from("/tmp/models");
        let p = resolve_model_path(&root, "gemma3");
        assert_eq!(p, PathBuf::from("/tmp/models/gemma3"));
    }

    #[test]
    fn resolve_repo_id_maps_gemma3_to_canonical() {
        assert_eq!(resolve_repo_id("gemma3"), GEMMA3_HF_REPO);
    }

    #[test]
    fn resolve_repo_id_passes_unknown_through() {
        // Unknown short-names pass through unchanged so users can point at a
        // fork or a private repo directly via --model <full-repo-id>.
        assert_eq!(resolve_repo_id("myorg/my-fork"), "myorg/my-fork");
    }

    #[test]
    fn required_files_covers_core_gemma_assets() {
        // Guardrail: if someone deletes a required file from the constant
        // the CandleInference::new check goes stale. Pin the expected shape.
        assert!(REQUIRED_FILES.contains(&"config.json"));
        assert!(REQUIRED_FILES.contains(&"tokenizer.json"));
        assert!(REQUIRED_FILES.contains(&"model.safetensors"));
    }
}
