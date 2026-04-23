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
//!   attributes (`auth_required`, `not_found`, `network`, `disk_full`,
//!   `checksum_failed`, …).
//!
//! Each downloaded file is SHA-256-verified against the LFS-pointer hash
//! advertised in the HF `x-linked-etag` header. Files without an
//! advertised SHA (small text assets that only carry a git-blob etag)
//! emit a `<result status="checksum_skipped"/>` event and the pull
//! continues — there's nothing sensible to verify against.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(test)]
pub mod fake_hub;
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

/// HF repo id for the CI smoke model. SmolLM-135M-Instruct is ~300MB,
/// Llama-architecture, ungated, and runs on a laptop CPU in under a minute
/// end-to-end — the right size for routine verification of the inference
/// pipeline. TinyLlama is kept around for higher-quality smoke tests where
/// the extra 2GB of download budget is acceptable.
pub const SMOLLM_HF_REPO: &str = "HuggingFaceTB/SmolLM-135M-Instruct";

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

    /// SHA-256 of the downloaded file didn't match the hash advertised
    /// by HuggingFace's LFS pointer. `pull_model` deletes the corrupt
    /// file before surfacing this so the next `--pull` re-downloads
    /// cleanly.
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
/// so the pull flow knows the total byte count and, when available, the
/// SHA-256 hash (for LFS-tracked files).
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    /// SHA-256 hash of the file content when HuggingFace advertised one,
    /// otherwise `None`. LFS pointers always carry a SHA-256 — the hub
    /// surfaces it via `x-linked-etag`. Plain text files (config.json,
    /// tokenizer.json on some repos) often have only a weak git-blob
    /// etag, which isn't a useful integrity check, so we ignore those.
    ///
    /// Callers use [`extract_expected_sha256`] to validate whatever the
    /// hub returned actually looks like a SHA-256 hex digest before
    /// running the comparison.
    pub expected_sha256: Option<String>,
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
    /// Emitted once per file after the SHA-256 comparison succeeds.
    /// Carries the hash so the operator can audit it against the HF repo
    /// page if ever suspicious.
    ChecksumOk {
        file: String,
        sha256: String,
    },
    /// Emitted when HF didn't advertise a SHA-256 etag for this file
    /// (usually small JSON/text assets that only have a weak git etag).
    /// The pull continues — there's nothing sensible to verify against.
    ChecksumSkipped {
        file: String,
        reason: String,
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
        "smollm" => SMOLLM_HF_REPO.to_string(),
        other => other.to_string(),
    }
}

/// True when the resolved repo id is known-ungated and safe to fetch
/// without an HF token. Used by the auth-required heuristic: we won't
/// pre-reject a pull when we know the repo doesn't need credentials.
pub fn is_ungated(repo_id: &str) -> bool {
    matches!(repo_id, TINYLLAMA_HF_REPO | SMOLLM_HF_REPO)
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

/// Validate that `s` looks like a SHA-256 hex digest (exactly 64 lowercase
/// hex chars). Returns the lowercased string when valid, `None` otherwise.
///
/// Used by both the HF header parser (where an etag might legitimately be a
/// git-blob hash or a weak ETag) and the fake-client test harness. Keeping
/// the check central means one place decides "is this a usable checksum",
/// instead of scattering the 64-char/hex invariant across modules.
pub fn extract_expected_sha256(s: &str) -> Option<String> {
    let lower = s.trim().to_lowercase();
    if lower.len() == 64 && lower.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(lower)
    } else {
        None
    }
}

/// Compute the SHA-256 of the file at `path` and return it as a lowercase
/// hex string. Reads in 64 KiB chunks so multi-GB safetensors files don't
/// blow up memory usage.
pub fn sha256_of_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Run the SHA-256 check against `file_path` using the expected hash
/// advertised in `metadata`. Emits one checksum event via `progress` and
/// returns Ok on both `verified` and `skipped` outcomes — the dream flow
/// only halts when the hashes disagree. On mismatch the corrupt file is
/// deleted (best-effort) so the next pull can re-fetch cleanly.
fn verify_file_checksum(
    file_path: &Path,
    file_name: &str,
    metadata: &FileMetadata,
    progress: &ProgressFn,
) -> Result<(), ModelManagerError> {
    let Some(expected) = metadata
        .expected_sha256
        .as_deref()
        .and_then(extract_expected_sha256)
    else {
        (progress)(ProgressEvent::ChecksumSkipped {
            file: file_name.to_string(),
            reason: "no SHA-256 etag advertised by HuggingFace".to_string(),
        });
        return Ok(());
    };

    let actual = sha256_of_file(file_path).map_err(ModelManagerError::Io)?;
    if actual == expected {
        (progress)(ProgressEvent::ChecksumOk {
            file: file_name.to_string(),
            sha256: actual,
        });
        Ok(())
    } else {
        // Delete the corrupt file so the next --pull re-downloads cleanly.
        // Best-effort — if removal fails we still surface the checksum
        // error to the caller, who owns the retry decision.
        if let Err(e) = std::fs::remove_file(file_path) {
            tracing::warn!(
                file = file_name,
                error = %e,
                "failed to delete corrupt file after checksum mismatch"
            );
        }
        Err(ModelManagerError::Checksum {
            file: file_name.to_string(),
            expected,
            actual,
        })
    }
}

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

    // Idempotency is decided per-file inside the loop below. Files that
    // already exist are SHA-256-verified against the advertised hash
    // (when HF provides one); a mismatch triggers a re-download. This
    // costs one HEAD per file on re-runs but guarantees a tampered or
    // truncated blob self-heals on the next `--pull`. Users who want an
    // unconditional full re-download can delete the model dir.
    let all_present_pre = required.iter().all(|f| model_dir.join(f).exists());

    let repo_id = resolve_repo_id(model_name);
    tracing::info!(
        model = model_name,
        repo = %repo_id,
        dest = %model_dir.display(),
        "pulling model from HuggingFace"
    );

    let mut total_bytes: u64 = 0;
    let mut any_downloaded = false;

    for file in required {
        let dest = model_dir.join(file);

        // Fetch metadata up-front so we have the advertised SHA-256 (when
        // HF provides one) for verification — both against an existing
        // on-disk file (idempotency check) and against a fresh download.
        // Errors here are non-terminal: we treat "no metadata" as "no
        // checksum" and continue without verification rather than halting
        // on transient HEAD failures.
        let metadata = match client.metadata(&repo_id, file).await {
            Ok(md) => md,
            Err(e) => {
                tracing::warn!(file, error = %e, "metadata probe failed; \
                    continuing without checksum verification");
                FileMetadata {
                    size: 0,
                    expected_sha256: None,
                }
            }
        };

        if dest.exists() {
            // Per-file idempotency — an earlier partial pull may have
            // landed one file. Verify it against the advertised hash
            // before trusting it; a mismatch (tampered or truncated
            // blob) re-enters the download path below.
            match verify_file_checksum(&dest, file, &metadata, &progress) {
                Ok(()) => {
                    // Sum it into the final byte count so the report
                    // reflects the on-disk layout accurately.
                    if let Ok(md) = std::fs::metadata(&dest) {
                        total_bytes = total_bytes.saturating_add(md.len());
                    }
                    continue;
                }
                Err(ModelManagerError::Checksum { .. }) => {
                    // verify_file_checksum already deleted the bad file;
                    // fall through to re-download.
                    tracing::warn!(
                        file,
                        "existing file failed checksum verification; re-downloading"
                    );
                }
                Err(e) => return Err(e),
            }
        }

        let resolved =
            download_with_retry(client, &repo_id, file, model_name, progress.clone()).await?;

        // hf-hub returns a pointer path inside its own cache; copy into
        // our per-short-name layout so CandleInference::new finds it
        // at the expected location. Copy (not symlink) to keep the
        // layout self-contained — users tarring up $AGENT_MEMORY_DIR/models
        // for backup shouldn't have to chase symlinks.
        std::fs::copy(&resolved, &dest).map_err(classify_io_error)?;
        any_downloaded = true;

        // Sum the file size for the final `pull_complete` event.
        if let Ok(md) = std::fs::metadata(&dest) {
            total_bytes = total_bytes.saturating_add(md.len());
        }

        // Checksum verification. On mismatch the file at `dest` is deleted
        // and the error propagates up — caller sees `status=checksum_failed`.
        // Skipped (no etag) is a progress event, not an error.
        verify_file_checksum(&dest, file, &metadata, &progress)?;
    }

    // `skipped` is true only when every required file was already on
    // disk AND verified cleanly — i.e. zero bytes flowed over the wire
    // this invocation. Partial re-downloads (e.g. a tampered safetensors
    // that had to be re-fetched) count as a non-skipped pull.
    let skipped = all_present_pre && !any_downloaded;

    Ok(PullReport {
        model: model_name.to_string(),
        files: required.len(),
        bytes_total: total_bytes,
        skipped,
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
    use crate::model_manager::fake_hub::FakeHubClient;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn null_progress() -> ProgressFn {
        Arc::new(|_ev| {})
    }

    fn capturing_progress() -> (ProgressFn, Arc<Mutex<Vec<ProgressEvent>>>) {
        let buf: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let buf_cb = buf.clone();
        let cb: ProgressFn = Arc::new(move |ev| {
            buf_cb.lock().unwrap().push(ev);
        });
        (cb, buf)
    }

    fn tmp_cache() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

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
    fn resolve_repo_id_maps_smollm_to_canonical() {
        assert_eq!(resolve_repo_id("smollm"), SMOLLM_HF_REPO);
    }

    #[test]
    fn is_ungated_flags_smollm() {
        // SmolLM is the laptop-CPU CI smoke model — must be ungated so
        // fresh contributors can run the e2e pipeline without HF_TOKEN.
        assert!(is_ungated(SMOLLM_HF_REPO));
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

    #[tokio::test]
    async fn pull_model_happy_path_downloads_all_files() {
        let cache = tmp_cache();
        let mut files = HashMap::new();
        for name in REQUIRED_FILES {
            files.insert(name.to_string(), vec![42u8; 16]);
        }
        let client = FakeHubClient::new(files);

        let (cb, events) = capturing_progress();
        let rep = pull_model(&client, cache.path(), "gemma3", cb)
            .await
            .unwrap();
        assert_eq!(rep.files, REQUIRED_FILES.len());
        assert!(!rep.skipped);
        for name in REQUIRED_FILES {
            assert!(cache.path().join("gemma3").join(name).exists());
        }
        // Progress emits Start/Done at minimum per file.
        let ev = events.lock().unwrap();
        let starts = ev
            .iter()
            .filter(|e| matches!(e, ProgressEvent::Start { .. }))
            .count();
        let dones = ev
            .iter()
            .filter(|e| matches!(e, ProgressEvent::Done { .. }))
            .count();
        assert_eq!(starts, REQUIRED_FILES.len());
        assert_eq!(dones, REQUIRED_FILES.len());
    }

    #[tokio::test]
    async fn pull_model_idempotent_when_files_present() {
        let cache = tmp_cache();
        let model_dir = cache.path().join("gemma3");
        std::fs::create_dir_all(&model_dir).unwrap();
        for f in REQUIRED_FILES {
            std::fs::write(model_dir.join(f), b"stub").unwrap();
        }
        // Client with NO files configured — if pull_model were to hit it,
        // the call would fail. Idempotent short-circuit means we never do.
        let client = FakeHubClient::new(HashMap::new());
        let rep = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap();
        assert!(rep.skipped);
    }

    #[tokio::test]
    async fn pull_model_surfaces_auth_required_with_model_short_name() {
        let cache = tmp_cache();
        let client = FakeHubClient::new(HashMap::new()).with_auth_required();
        let err = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap_err();
        match err {
            ModelManagerError::AuthRequired { model, repo } => {
                // Substitution: CLI short-name is threaded through so the
                // error message points the user at the repo they tried to
                // pull, not at the file-name of the first missing asset.
                assert_eq!(model, "gemma3");
                assert_eq!(repo, GEMMA3_HF_REPO);
            }
            other => panic!("expected AuthRequired, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pull_model_surfaces_not_found() {
        let cache = tmp_cache();
        let client = FakeHubClient::new(HashMap::new()).with_not_found();
        let err = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap_err();
        assert!(matches!(err, ModelManagerError::NotFound { .. }));
    }

    #[tokio::test]
    async fn pull_model_retries_transient_then_exhausts() {
        let cache = tmp_cache();
        // Every attempt fails transiently — retry loop bails after
        // MAX_ATTEMPTS and surfaces NetworkError with attempt count.
        let client = FakeHubClient::new(HashMap::new()).with_always_transient();
        let err = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap_err();
        match err {
            ModelManagerError::NetworkError { attempts, .. } => {
                assert_eq!(attempts, MAX_ATTEMPTS);
            }
            other => panic!("expected NetworkError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pull_model_retries_transient_then_recovers() {
        let cache = tmp_cache();
        let mut files = HashMap::new();
        for name in REQUIRED_FILES {
            files.insert(name.to_string(), vec![7u8; 8]);
        }
        // First attempt transient, then success. Counts are per-file.
        let client = FakeHubClient::new(files).with_transient_first_attempts(1);

        let rep = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap();
        assert_eq!(rep.files, REQUIRED_FILES.len());
    }

    #[tokio::test]
    async fn pull_model_resumes_when_one_file_already_on_disk() {
        // Simulates an interrupted pull — pre-seed one of the required
        // files in the target dir, then ensure the fake client is only
        // asked for the remaining three.
        let cache = tmp_cache();
        let model_dir = cache.path().join("gemma3");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"already-here").unwrap();

        let mut files = HashMap::new();
        for name in REQUIRED_FILES {
            if *name != "config.json" {
                files.insert(name.to_string(), vec![1u8; 4]);
            }
        }
        let client = FakeHubClient::new(files);

        let rep = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap();
        assert_eq!(rep.files, REQUIRED_FILES.len());
        assert_eq!(
            client.call_count(),
            REQUIRED_FILES.len() - 1,
            "pre-existing file must not be re-downloaded"
        );
        // Pre-seeded content untouched.
        let body = std::fs::read(model_dir.join("config.json")).unwrap();
        assert_eq!(body, b"already-here");
    }

    // -----------------------------------------------------------------
    // Checksum verification
    // -----------------------------------------------------------------

    /// Helper: compute the expected SHA-256 of a byte buffer so tests can
    /// wire the fake client with the "correct" hash.
    fn hash_bytes(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        hex::encode(h.finalize())
    }

    #[test]
    fn extract_expected_sha256_accepts_canonical_hex_digest() {
        let good = "a".repeat(64);
        assert_eq!(extract_expected_sha256(&good), Some(good.clone()));
    }

    #[test]
    fn extract_expected_sha256_rejects_wrong_length() {
        // Too short.
        assert_eq!(extract_expected_sha256(&"a".repeat(63)), None);
        // Too long.
        assert_eq!(extract_expected_sha256(&"a".repeat(65)), None);
    }

    #[test]
    fn extract_expected_sha256_rejects_non_hex() {
        // 64 chars but contains a non-hex character.
        let bad = format!("{}{}", "a".repeat(63), "Z");
        assert_eq!(extract_expected_sha256(&bad), None);
    }

    #[test]
    fn extract_expected_sha256_lowercases_uppercase_input() {
        let upper = "A".repeat(64);
        assert_eq!(extract_expected_sha256(&upper), Some("a".repeat(64)));
    }

    #[test]
    fn sha256_of_file_hashes_content_deterministically() {
        let dir = tmp_cache();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let actual = sha256_of_file(&path).unwrap();
        // Pinned reference from `printf "hello world" | sha256sum`.
        assert_eq!(
            actual,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[tokio::test]
    async fn pull_model_checksum_happy_path_matches_and_emits_ok() {
        let cache = tmp_cache();
        let mut files = HashMap::new();
        let mut client = FakeHubClient::new({
            let mut m = HashMap::new();
            for name in REQUIRED_FILES {
                let body = vec![42u8; 16];
                m.insert(name.to_string(), body.clone());
                files.insert(name.to_string(), body);
            }
            m
        });
        // Attach the correct etag for every file so verification succeeds.
        for (name, body) in &files {
            client = client.with_etag(name, Some(hash_bytes(body)));
        }

        let (cb, events) = capturing_progress();
        let rep = pull_model(&client, cache.path(), "gemma3", cb)
            .await
            .unwrap();
        assert_eq!(rep.files, REQUIRED_FILES.len());

        let ev = events.lock().unwrap();
        let ok_count = ev
            .iter()
            .filter(|e| matches!(e, ProgressEvent::ChecksumOk { .. }))
            .count();
        assert_eq!(
            ok_count,
            REQUIRED_FILES.len(),
            "every downloaded file must emit ChecksumOk"
        );
    }

    #[tokio::test]
    async fn pull_model_checksum_mismatch_deletes_file_and_errors() {
        let cache = tmp_cache();
        let mut files = HashMap::new();
        for name in REQUIRED_FILES {
            files.insert(name.to_string(), vec![1u8; 8]);
        }
        let client = FakeHubClient::new(files).with_etag(
            "config.json",
            // Wrong hash — 64 hex chars so it passes the
            // extract_expected_sha256 validation but won't match the real
            // SHA-256 of the file contents.
            Some("a".repeat(64)),
        );

        let err = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap_err();
        match err {
            ModelManagerError::Checksum { file, .. } => {
                assert_eq!(file, "config.json");
                // Corrupt file deleted so the next pull starts clean.
                assert!(
                    !cache.path().join("gemma3").join("config.json").exists(),
                    "mismatched file must be removed from disk"
                );
            }
            other => panic!("expected Checksum, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pull_model_checksum_skipped_when_no_etag_advertised() {
        let cache = tmp_cache();
        let mut files = HashMap::new();
        for name in REQUIRED_FILES {
            files.insert(name.to_string(), vec![5u8; 8]);
        }
        // Default FakeHubClient returns None for every etag → pull should
        // succeed and emit ChecksumSkipped per file.
        let client = FakeHubClient::new(files);

        let (cb, events) = capturing_progress();
        let rep = pull_model(&client, cache.path(), "gemma3", cb)
            .await
            .unwrap();
        assert_eq!(rep.files, REQUIRED_FILES.len());

        let ev = events.lock().unwrap();
        let skipped = ev
            .iter()
            .filter(|e| matches!(e, ProgressEvent::ChecksumSkipped { .. }))
            .count();
        assert_eq!(skipped, REQUIRED_FILES.len());
        // No `ChecksumOk` events fired because there was nothing to check.
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, ProgressEvent::ChecksumOk { .. }))
                .count(),
            0
        );
    }

    #[tokio::test]
    async fn pull_model_redownloads_tampered_existing_file() {
        // Pre-seed the model dir with a file whose contents don't match
        // the advertised SHA-256 — simulates a truncation or tampering
        // between runs. The pull must re-fetch the authentic bytes and
        // the final on-disk content must be the authentic body.
        let cache = tmp_cache();
        let model_dir = cache.path().join("gemma3");
        std::fs::create_dir_all(&model_dir).unwrap();

        let authentic = vec![0u8; 32];
        let tampered = vec![0xFFu8; 32];
        std::fs::write(model_dir.join("config.json"), &tampered).unwrap();

        let mut files = HashMap::new();
        for name in REQUIRED_FILES {
            files.insert(name.to_string(), authentic.clone());
        }
        // Attach the correct hash for config.json so verification on the
        // existing tampered file fails; other files have no etag, so
        // they pass through as checksum_skipped.
        let client =
            FakeHubClient::new(files).with_etag("config.json", Some(hash_bytes(&authentic)));

        let rep = pull_model(&client, cache.path(), "gemma3", null_progress())
            .await
            .unwrap();
        assert!(
            !rep.skipped,
            "pull must NOT be marked skipped when a file was re-downloaded"
        );
        // The tampered file was replaced with the authentic bytes from
        // the fake client.
        let on_disk = std::fs::read(model_dir.join("config.json")).unwrap();
        assert_eq!(on_disk, authentic, "tampered file must be replaced");
    }

    #[tokio::test]
    async fn pull_model_checksum_skipped_when_etag_is_not_a_hash() {
        // Some HF assets surface weak git-blob etags like `W/"abc"`. Our
        // header parser tries to strip those but occasionally a string
        // that doesn't match our 64-hex rule lands in `expected_sha256`.
        // Defense in depth: the module-level `extract_expected_sha256`
        // filter must reject anything that isn't a proper SHA.
        let cache = tmp_cache();
        let mut files = HashMap::new();
        for name in REQUIRED_FILES {
            files.insert(name.to_string(), vec![9u8; 8]);
        }
        let mut client = FakeHubClient::new(files);
        // 8-char git hash — not a SHA-256. Must be treated as "no etag".
        client = client.with_etag("config.json", Some("deadbeef".to_string()));

        let (cb, events) = capturing_progress();
        let rep = pull_model(&client, cache.path(), "gemma3", cb)
            .await
            .unwrap();
        assert_eq!(rep.files, REQUIRED_FILES.len());

        // The malformed-etag file produced a ChecksumSkipped with a hint
        // about the missing-SHA condition in the reason.
        let ev = events.lock().unwrap();
        let skipped_for_config: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                ProgressEvent::ChecksumSkipped { file, .. } if file == "config.json" => {
                    Some(file.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(skipped_for_config.len(), 1);
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
