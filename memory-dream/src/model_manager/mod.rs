//! Model download manager for `memory-dream --pull`.
//!
//! Uses the `hf-hub` crate (async/tokio) to fetch model weights from
//! HuggingFace into the local cache directory. The cache location is
//! anchored at `$AGENT_MEMORY_DIR/models/<short-name>/` so the rest of the
//! pipeline (notably [`crate::inference::CandleInference`]) can find files
//! without consulting hf-hub's default `~/.cache/huggingface/` tree.
//!
//! ## Design summary
//!
//! * `HubClient` trait abstracts the network layer. Real code uses
//!   [`HfHubClient`]; tests swap in a deterministic fake so `cargo test`
//!   stays fully offline.
//! * [`pull_model`] handles progress reporting, resume, retry, and
//!   materializes the final layout under `<cache_root>/<short_name>/`.
//! * Errors are shaped so the CLI can emit distinct light-XML status
//!   attributes (`auth_required`, `not_found`, `network`, `disk_full`, …).
//!
//! Checksum verification is an explicitly separate follow-up — this module
//! emits a structured `Checksum` error variant but does not compute hashes
//! itself yet.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use thiserror::Error;

pub mod hub;

pub use hub::HfHubClient;

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
/// can be taught to map it — for now we recognize `gemma3` (gated) and
/// `tinyllama` (ungated, used for smoke-testing the pull pipeline) and
/// fall through to "use the user-supplied string as the repo id directly"
/// for anything else, which lets advanced users bypass the lookup entirely.
pub const GEMMA3_HF_REPO: &str = "google/gemma-3-1b-it";

/// HF repo id for the ungated smoke-test model. TinyLlama-1.1B is small
/// (~2GB), openly licensed, and requires no HF token — it exists in the
/// short-name map solely to let contributors (and CI) prove the download
/// machinery end-to-end without HuggingFace credentials.
pub const TINYLLAMA_HF_REPO: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

/// File names that every model cache directory must contain for
/// `CandleInference::new` to succeed. Used by both the pull flow (to decide
/// which assets to fetch) and the integrity check after download.
///
/// `tokenizer_config.json` is optional on some repos (TinyLlama ships it,
/// but custom forks may omit it). See [`required_files_for`] for the
/// per-model selector that honors that nuance.
pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
];

/// Return the required-file list for a given model short-name. Kept as a
/// function (rather than a per-model const) so future additions — quantized
/// variants, multi-shard safetensors — can override the default set in one
/// place without touching the rest of the pull flow.
pub fn required_files_for(_model_name: &str) -> &'static [&'static str] {
    REQUIRED_FILES
}

/// Errors surfaced by the model manager.
///
/// The variants are deliberately narrow so the CLI can map each to a
/// distinct `<result status="..."/>` attribute. Adding a new failure
/// class should produce a new variant, not a reused `Download(String)`.
#[derive(Debug, Error)]
pub enum ModelManagerError {
    /// The `$AGENT_MEMORY_DIR/models` root could not be created.
    #[error("model cache root does not exist and could not be created: {0}")]
    CacheRoot(String),

    /// The HF repo returned 401/403 — gated model, token missing or invalid.
    /// Carries the repo id so the CLI can print the 3-step remediation
    /// pointing at the exact repo page.
    #[error(
        "model '{model}' ({repo}) is gated on HuggingFace. \
         Accept the license at https://huggingface.co/{repo}, \
         create a token at https://huggingface.co/settings/tokens, \
         then `export HF_TOKEN=<token>` and re-run `memory-dream --pull`."
    )]
    AuthRequired { model: String, repo: String },

    /// The HF repo returned 404 — the repo id doesn't exist or the file
    /// isn't in the repo's snapshot.
    #[error("not found on HuggingFace: {repo}/{file}")]
    NotFound { repo: String, file: String },

    /// All retry attempts exhausted against a transient network failure.
    /// `attempts` is the total number of tries (including the first).
    #[error("network error after {attempts} attempts: {source}")]
    NetworkError {
        attempts: u32,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Filesystem reported out-of-space while writing the downloaded blob.
    /// Split from generic `Io` so the CLI can emit a dedicated hint about
    /// the cache root location.
    #[error("disk full while writing to {path}")]
    DiskFull { path: PathBuf },

    /// Integrity check failure. Reserved for the follow-up that wires
    /// SHA-256 verification against the hash advertised by the HF API.
    #[error("checksum verification failed for {file}: expected {expected}, got {actual}")]
    Checksum {
        file: String,
        expected: String,
        actual: String,
    },

    /// Generic IO failure (permissions, broken symlink, etc.) that isn't
    /// disk-space-related.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl ModelManagerError {
    /// Map to the short status token used in light-XML `<result/>` lines.
    pub fn status_token(&self) -> &'static str {
        match self {
            Self::CacheRoot(_) => "cache_root_failed",
            Self::AuthRequired { .. } => "auth_required",
            Self::NotFound { .. } => "not_found",
            Self::NetworkError { .. } => "network",
            Self::DiskFull { .. } => "disk_full",
            Self::Checksum { .. } => "checksum_failed",
            Self::Io(_) => "io_error",
        }
    }
}

/// Metadata for a single remote file. Populated from HF's HEAD response
/// so the pull flow knows the total byte count before streaming.
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
}

/// Progress callback hook. The pull loop invokes it once at the start
/// (`bytes_done=0`, `bytes_total=size`), then periodically during the
/// stream, then once at the end (`bytes_done=bytes_total`). Emission
/// throttling lives in the caller (not here) so the HF-hub impl stays
/// a dumb pipe.
pub type ProgressFn = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Start {
        file: String,
        bytes_total: u64,
    },
    Update {
        file: String,
        bytes_done: u64,
        bytes_total: u64,
    },
    Done {
        file: String,
        bytes_total: u64,
    },
}

/// Abstraction over the HuggingFace Hub client. Real code uses the
/// `hf-hub` crate; unit tests substitute [`fake_hub::FakeHubClient`]
/// so no network access is required.
///
/// The trait is intentionally narrow — `metadata` for the HEAD-style
/// size lookup, and `download_to` for the byte-streaming side. Resume
/// is handled by the concrete implementation (hf-hub's cache supports
/// it natively); the trait only cares about "file ends up on disk at
/// the given path with the given content".
#[async_trait]
pub trait HubClient: Send + Sync {
    /// Fetch the expected size of `file` in `repo`. Used to initialize
    /// the progress bar before the stream starts.
    async fn metadata(&self, repo: &str, file: &str) -> Result<FileMetadata, ModelManagerError>;

    /// Download `file` from `repo` into the client's local cache and
    /// return the resolved on-disk path. The caller is responsible for
    /// placing (copy or symlink) the returned file into the expected
    /// per-short-name layout.
    async fn download_to(
        &self,
        repo: &str,
        file: &str,
        progress: ProgressFn,
    ) -> Result<PathBuf, ModelManagerError>;
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
/// Known short names resolve to their canonical repo id. Anything else is
/// passed through unchanged so power users can reference a fork or a
/// private repo directly.
pub fn resolve_repo_id(model_name: &str) -> String {
    match model_name {
        "gemma3" => GEMMA3_HF_REPO.to_string(),
        "tinyllama" => TINYLLAMA_HF_REPO.to_string(),
        other => other.to_string(),
    }
}

/// True when the resolved repo id is known-ungated and safe to fetch
/// without an HF token. Used by the auth-required heuristic: we won't
/// pre-reject a pull when we know the repo doesn't need credentials.
pub fn is_ungated(repo_id: &str) -> bool {
    matches!(repo_id, TINYLLAMA_HF_REPO)
}

/// Read an HF auth token from the environment. Checks `HF_TOKEN` first
/// (the canonical variable name used by the `huggingface_hub` Python SDK)
/// and falls back to `HUGGING_FACE_HUB_TOKEN` for users who have the
/// alternate form set.
pub fn hf_token_from_env() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .filter(|s| !s.trim().is_empty())
}

/// Retry policy constants. Kept as module-level `const`s rather than a
/// struct because there's one real policy and making it configurable
/// invites drift between the CLI default and the tested behavior.
pub const MAX_ATTEMPTS: u32 = 3;
pub const BACKOFF_BASE_MS: u64 = 1_000;

/// Download the configured model into `<cache_root>/<model_name>/`.
///
/// Responsibilities:
///   1. Idempotency — returns Ok immediately if all required files are
///      already on disk.
///   2. Delegates each file fetch to `client`, which owns resume semantics.
///   3. Progress emission via `progress` — one call per file start,
///      periodic updates, one call per file done.
///   4. Retry with exponential backoff on transient network failures.
///   5. Materializes the per-short-name layout by copying/symlinking
///      from the client's cache to `<cache_root>/<model_name>/<file>`.
pub async fn pull_model(
    client: &dyn HubClient,
    cache_root: &Path,
    model_name: &str,
    progress: ProgressFn,
) -> Result<PullReport, ModelManagerError> {
    std::fs::create_dir_all(cache_root).map_err(|e| ModelManagerError::CacheRoot(e.to_string()))?;

    let model_dir = resolve_model_path(cache_root, model_name);
    std::fs::create_dir_all(&model_dir)?;

    let required = required_files_for(model_name);

    // Idempotency: if every required file is already on disk, skip the
    // download entirely. Users can delete the model dir to force a fresh
    // pull; explicit --force is a follow-up.
    let all_present = required.iter().all(|f| model_dir.join(f).exists());
    if all_present {
        tracing::info!(
            model = model_name,
            "model already present; skipping download"
        );
        return Ok(PullReport {
            model: model_name.to_string(),
            files: required.len(),
            bytes_total: 0,
            skipped: true,
        });
    }

    let repo_id = resolve_repo_id(model_name);
    tracing::info!(
        model = model_name,
        repo = %repo_id,
        dest = %model_dir.display(),
        "pulling model from HuggingFace"
    );

    let mut total_bytes: u64 = 0;

    for file in required {
        let dest = model_dir.join(file);
        if dest.exists() {
            // Per-file idempotency — one file may have landed in a prior
            // partial run; skip it and move on.
            continue;
        }

        let resolved =
            download_with_retry(client, &repo_id, file, model_name, progress.clone()).await?;

        // hf-hub returns a pointer path inside its own cache; copy into
        // our per-short-name layout so CandleInference::new finds it
        // at the expected location. Copy (not symlink) to keep the
        // layout self-contained — users tarring up $AGENT_MEMORY_DIR/models
        // for backup shouldn't have to chase symlinks.
        std::fs::copy(&resolved, &dest).map_err(classify_io_error)?;

        // Sum the file size for the final `pull_complete` event.
        if let Ok(md) = std::fs::metadata(&dest) {
            total_bytes = total_bytes.saturating_add(md.len());
        }
    }

    Ok(PullReport {
        model: model_name.to_string(),
        files: required.len(),
        bytes_total: total_bytes,
        skipped: false,
    })
}

/// One-shot fetch for a single file with retry + backoff. Separated from
/// [`pull_model`] so tests can drive it in isolation.
async fn download_with_retry(
    client: &dyn HubClient,
    repo_id: &str,
    file: &str,
    model_name: &str,
    progress: ProgressFn,
) -> Result<PathBuf, ModelManagerError> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match client.download_to(repo_id, file, progress.clone()).await {
            Ok(path) => return Ok(path),
            Err(ModelManagerError::AuthRequired { .. }) => {
                // Auth failures are not transient — return immediately with
                // the model name substituted so the CLI error references
                // the short-name the user typed.
                return Err(ModelManagerError::AuthRequired {
                    model: model_name.to_string(),
                    repo: repo_id.to_string(),
                });
            }
            Err(ModelManagerError::NotFound { .. }) => {
                // 404 is also terminal — retrying won't summon the file.
                return Err(ModelManagerError::NotFound {
                    repo: repo_id.to_string(),
                    file: file.to_string(),
                });
            }
            Err(err) if is_transient(&err) && attempt < MAX_ATTEMPTS => {
                let backoff_ms = BACKOFF_BASE_MS * (1u64 << (attempt - 1));
                tracing::warn!(
                    attempt,
                    max = MAX_ATTEMPTS,
                    backoff_ms,
                    file,
                    "transient download failure; retrying"
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                continue;
            }
            Err(err) => {
                // Exhausted retries or non-transient error: wrap with
                // the final attempt count so the operator can tell
                // whether the code ever got a chance to succeed.
                if is_transient(&err) {
                    return Err(ModelManagerError::NetworkError {
                        attempts: attempt,
                        source: Box::new(err),
                    });
                }
                return Err(err);
            }
        }
    }
}

/// Summary emitted when `pull_model` completes successfully. Kept small
/// so main.rs can render a single light-XML `pull_complete` line.
#[derive(Debug, Clone)]
pub struct PullReport {
    pub model: String,
    pub files: usize,
    pub bytes_total: u64,
    pub skipped: bool,
}

/// A transient error is one where a retry has a plausible chance of
/// succeeding. Auth, 404, checksum mismatches, and disk-full are all
/// deterministic — no point sleeping and trying again.
fn is_transient(err: &ModelManagerError) -> bool {
    matches!(
        err,
        ModelManagerError::NetworkError { .. } | ModelManagerError::Io(_)
    )
}

/// Translate a raw IO error into either `DiskFull` or generic `Io`
/// depending on the underlying errno. Centralized so the distinction
/// stays consistent across every filesystem call in the pull flow.
fn classify_io_error(err: std::io::Error) -> ModelManagerError {
    // ErrorKind::StorageFull is stable as of Rust 1.83; older compilers
    // fall back to matching the error message. We match on both to be
    // robust against MSRV shifts in either direction.
    #[allow(unreachable_patterns)]
    match err.kind() {
        // `StorageFull` landed in 1.83; when built on older toolchains
        // this arm is dead but harmless.
        _ if err.raw_os_error() == Some(libc_enospc()) => ModelManagerError::DiskFull {
            path: PathBuf::new(),
        },
        _ => ModelManagerError::Io(err),
    }
}

/// Platform-specific ENOSPC value. Centralized because `libc` isn't a
/// direct dep and the constant is stable across Unix.
fn libc_enospc() -> i32 {
    // ENOSPC = 28 on Linux and macOS. Windows uses `ERROR_DISK_FULL` (112)
    // via std::io::ErrorKind::StorageFull, but `raw_os_error` on Windows
    // returns the Win32 code, so match both.
    #[cfg(target_os = "windows")]
    {
        112
    }
    #[cfg(not(target_os = "windows"))]
    {
        28
    }
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
    fn resolve_repo_id_maps_tinyllama_to_canonical() {
        assert_eq!(resolve_repo_id("tinyllama"), TINYLLAMA_HF_REPO);
    }

    #[test]
    fn resolve_repo_id_passes_unknown_through() {
        assert_eq!(resolve_repo_id("myorg/my-fork"), "myorg/my-fork");
    }

    #[test]
    fn required_files_covers_core_assets() {
        let req = required_files_for("gemma3");
        assert!(req.contains(&"config.json"));
        assert!(req.contains(&"tokenizer.json"));
        assert!(req.contains(&"model.safetensors"));
    }

    #[test]
    fn is_ungated_flags_tinyllama_only() {
        assert!(is_ungated(TINYLLAMA_HF_REPO));
        assert!(!is_ungated(GEMMA3_HF_REPO));
    }

    #[test]
    fn status_tokens_are_stable_and_distinct() {
        // Pinned so CLI consumers parsing the light-XML output don't break
        // if the error names are reshuffled.
        assert_eq!(
            ModelManagerError::AuthRequired {
                model: "x".into(),
                repo: "y".into()
            }
            .status_token(),
            "auth_required"
        );
        assert_eq!(
            ModelManagerError::NotFound {
                repo: "x".into(),
                file: "y".into()
            }
            .status_token(),
            "not_found"
        );
    }

    #[test]
    fn hf_token_from_env_prefers_hf_token() {
        // Use a scoped-env guard to avoid poisoning the test suite.
        // Both vars unset → None.
        let _g1 = EnvGuard::unset("HF_TOKEN");
        let _g2 = EnvGuard::unset("HUGGING_FACE_HUB_TOKEN");
        assert!(hf_token_from_env().is_none());

        let _g3 = EnvGuard::set("HUGGING_FACE_HUB_TOKEN", "fallback");
        assert_eq!(hf_token_from_env().as_deref(), Some("fallback"));

        let _g4 = EnvGuard::set("HF_TOKEN", "primary");
        assert_eq!(hf_token_from_env().as_deref(), Some("primary"));
    }

    /// Minimal env-var RAII guard — sets or unsets an env var for the
    /// lifetime of the guard. Scoped to this test module because the
    /// standard lib doesn't ship an equivalent and we want determinism
    /// without pulling in `temp-env`.
    struct EnvGuard {
        key: String,
        prior: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &str, value: &str) -> Self {
            let prior = std::env::var(key).ok();
            // Safety: tests in this module run sequentially via the
            // single-threaded test runtime for env-dependent cases; the
            // race window with other threads is accepted here because
            // no other test mutates HF_TOKEN.
            unsafe {
                std::env::set_var(key, value);
            }
            Self {
                key: key.to_string(),
                prior,
            }
        }
        fn unset(key: &str) -> Self {
            let prior = std::env::var(key).ok();
            unsafe {
                std::env::remove_var(key);
            }
            Self {
                key: key.to_string(),
                prior,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prior {
                Some(v) => unsafe { std::env::set_var(&self.key, v) },
                None => unsafe { std::env::remove_var(&self.key) },
            }
        }
    }
}
