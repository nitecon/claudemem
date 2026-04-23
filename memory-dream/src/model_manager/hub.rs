//! [`HubClient`] implementation backed by the `hf-hub` crate (tokio flavor).
//!
//! This module is the only place in the codebase that touches
//! `hf_hub::api::tokio::*`. Keeping the dependency surface narrow lets the
//! rest of `model_manager` evolve (retry policy, progress renderers,
//! additional short-name mappings) without needing to re-read how hf-hub
//! models its own types.
//!
//! Behavior notes:
//! * The hf-hub client is configured with `with_cache_dir(<cache_root>)`
//!   so its intermediate blob/snapshot tree lands inside our own cache
//!   (`$AGENT_MEMORY_DIR/models/.hf/`). Files are then copied into the
//!   per-short-name layout by the caller in [`super::pull_model`].
//! * `with_progress(false)` silences hf-hub's built-in indicatif bar —
//!   the CLI surfaces its own light-XML progress lines.
//! * Auth errors (401/403) are detected by string-matching the
//!   `hf_hub::api::tokio::ApiError::RequestError` payload. The status
//!   code isn't exposed as a typed field on that variant, but the
//!   `reqwest::Error` `Display` includes it verbatim.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, Progress};

use super::{FileMetadata, HubClient, ModelManagerError, ProgressEvent, ProgressFn};

/// Sub-directory under `cache_root` used for hf-hub's internal cache
/// layout. Prefixed with a dot so it doesn't collide with the
/// per-short-name directories (`gemma3/`, `tinyllama/`, ...) the caller
/// materializes alongside it.
const HF_CACHE_SUBDIR: &str = ".hf";

/// Real hf-hub-backed client.
///
/// Constructed once per `memory-dream --pull` invocation. The underlying
/// [`Api`] is cheap to clone and internally refcounted, so taking `&self`
/// for every download is fine.
pub struct HfHubClient {
    api: Api,
}

impl HfHubClient {
    /// Build a new client rooted at `cache_root`.
    ///
    /// * `cache_root` — typically `$AGENT_MEMORY_DIR/models`. The hf-hub
    ///   internal cache lives at `<cache_root>/.hf/`.
    /// * `token` — optional HF access token. When `None`, hf-hub falls
    ///   back to whatever the `HF_HOME/token` file contains, which is
    ///   usually not set on servers; gated repos will 401 in that case
    ///   and [`super::pull_model`] surfaces `AuthRequired`.
    pub fn new(cache_root: &Path, token: Option<String>) -> Result<Self, ModelManagerError> {
        let hf_cache = cache_root.join(HF_CACHE_SUBDIR);
        std::fs::create_dir_all(&hf_cache)?;

        let mut builder = ApiBuilder::new()
            .with_cache_dir(hf_cache)
            .with_progress(false);
        if let Some(tok) = token {
            builder = builder.with_token(Some(tok));
        }
        let api = builder.build().map_err(map_api_error)?;
        Ok(Self { api })
    }
}

#[async_trait]
impl HubClient for HfHubClient {
    async fn metadata(&self, repo: &str, file: &str) -> Result<FileMetadata, ModelManagerError> {
        // hf-hub doesn't expose a cheap HEAD directly on ApiRepo, but the
        // download path fetches metadata first as part of its normal
        // flow; we surface size via a dedicated HEAD using the raw
        // reqwest client so the progress bar can initialize with the
        // true total before bytes start flowing.
        //
        // Trade-off: this costs one extra HTTP round-trip per file vs.
        // letting hf-hub handle everything in one go. That's cheap (~5ms)
        // against multi-minute downloads.
        //
        // IMPORTANT: The first hop (huggingface.co) is the ONLY one that
        // surfaces `x-linked-etag` — the true SHA-256 of the LFS pointer.
        // The S3/xet mirror past the redirect only exposes its own
        // object-addressed hash (xet content id), which is not the
        // safetensors SHA-256 and would cause every checksum check to
        // fail. So we build a dedicated no-redirect client here and read
        // the etag headers off the 302 response directly.
        let url = self.api.model(repo.to_string()).url(file);
        let no_redirect_client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| ModelManagerError::NetworkError {
                attempts: 1,
                source: Box::new(e),
            })?;
        let first_hop = no_redirect_client
            .get(&url)
            .header(reqwest::header::RANGE, "bytes=0-0")
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, repo, file))?;

        if !first_hop.status().is_success() && !first_hop.status().is_redirection() {
            return Err(classify_status(first_hop.status(), repo, file));
        }

        // Capture the LFS-pointer SHA from the first hop BEFORE following
        // the redirect. `x-linked-etag` is set by the hub on LFS-tracked
        // files (.safetensors, .bin, .gguf, ...) and contains the
        // canonical SHA-256. The plain `etag` field on small assets is a
        // git-blob hash — useless for integrity — so we prefer the linked
        // form and fall back to the regular one; the caller validates the
        // format before trusting it as a checksum.
        let expected_sha256 = extract_sha_from_headers(first_hop.headers());

        // Follow the redirect manually so we can read Content-Range on
        // the final hop for progress-bar initialization. The redirect
        // target (S3/xet) returns Content-Range; the first hop returns
        // just a Location header.
        let resp = if first_hop.status().is_redirection() {
            let loc = first_hop
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string();
            if loc.is_empty() {
                return Err(ModelManagerError::NetworkError {
                    attempts: 1,
                    source: "HEAD redirect missing Location header".into(),
                });
            }
            self.api
                .client()
                .get(&loc)
                .header(reqwest::header::RANGE, "bytes=0-0")
                .send()
                .await
                .map_err(|e| classify_reqwest_error(e, repo, file))?
        } else {
            first_hop
        };

        let size = resp
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.rsplit('/').next())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        Ok(FileMetadata {
            size,
            expected_sha256,
        })
    }

    async fn download_to(
        &self,
        repo: &str,
        file: &str,
        progress: ProgressFn,
    ) -> Result<PathBuf, ModelManagerError> {
        // Adapt our ProgressFn-shaped callback to hf-hub's Progress trait.
        // hf-hub drives init/update/finish per file; we forward each to
        // a translated ProgressEvent. Update-throttling (don't flood
        // stdout on every chunk) happens at the CLI layer — here we
        // emit every event hf-hub gives us so tests can assert shape.
        let adapter = ProgressAdapter::new(file.to_string(), progress);
        let api_repo = self.api.model(repo.to_string());
        let path = api_repo
            .download_with_progress(file, adapter)
            .await
            .map_err(|e| map_api_error_contextual(e, repo, file))?;
        Ok(path)
    }
}

/// Map a raw [`ApiError`] into our typed taxonomy.
///
/// Used when building the Api (no repo/file context available) — the
/// contextual variant for per-download errors lives in
/// [`map_api_error_contextual`].
fn map_api_error(e: ApiError) -> ModelManagerError {
    ModelManagerError::NetworkError {
        attempts: 1,
        source: Box::new(e),
    }
}

/// Map an `ApiError` thrown from a specific repo/file download into a
/// typed [`ModelManagerError`]. This is where 401/403/404 get
/// distinguished from transient network errors.
fn map_api_error_contextual(e: ApiError, repo: &str, file: &str) -> ModelManagerError {
    // Detect HTTP status by inspecting the error's Display. reqwest's
    // Display always includes "HTTP status <code>" when `.status()` was
    // set, so the string match is stable.
    let msg = e.to_string();
    if msg.contains("401") || msg.contains("Unauthorized") {
        return ModelManagerError::AuthRequired {
            model: repo.to_string(),
            repo: repo.to_string(),
        };
    }
    if msg.contains("403") || msg.contains("Forbidden") {
        return ModelManagerError::AuthRequired {
            model: repo.to_string(),
            repo: repo.to_string(),
        };
    }
    if msg.contains("404") || msg.contains("Not Found") {
        return ModelManagerError::NotFound {
            repo: repo.to_string(),
            file: file.to_string(),
        };
    }
    // Anything else is treated as transient so the retry loop gets a
    // chance. Non-recoverable cases (auth, 404) are already branched
    // above; genuine permanent failures will exhaust retries and land
    // under NetworkError{attempts=MAX}.
    ModelManagerError::NetworkError {
        attempts: 1,
        source: Box::new(e),
    }
}

/// Convert a raw reqwest error (from our HEAD request) into a typed
/// `ModelManagerError`. Symmetric with `map_api_error_contextual`.
fn classify_reqwest_error(e: reqwest::Error, repo: &str, file: &str) -> ModelManagerError {
    if let Some(status) = e.status() {
        return classify_status(status, repo, file);
    }
    ModelManagerError::NetworkError {
        attempts: 1,
        source: Box::new(e),
    }
}

/// Pull a SHA-256 out of the HF response headers. Prefers `x-linked-etag`
/// (the LFS-pointer sha) over plain `etag` (git-blob sha). Strips
/// surrounding quotes and the `W/` weak prefix before returning.
///
/// Returns None when neither header is present or when the value doesn't
/// look like a 64-char hex string after cleanup — validation by the caller
/// via [`super::extract_expected_sha256`] is defense in depth.
fn extract_sha_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    const LINKED: &str = "x-linked-etag";
    const PLAIN: &str = "etag";
    let raw = headers
        .get(LINKED)
        .or_else(|| headers.get(PLAIN))
        .and_then(|v| v.to_str().ok())?;
    let cleaned = raw
        .trim()
        .trim_start_matches("W/")
        .trim_start_matches("w/")
        .trim_matches('"')
        .to_string();
    super::extract_expected_sha256(&cleaned).map(|s| s.to_string())
}

/// Classify a reqwest `StatusCode` into the typed error taxonomy.
fn classify_status(status: reqwest::StatusCode, repo: &str, file: &str) -> ModelManagerError {
    match status.as_u16() {
        401 | 403 => ModelManagerError::AuthRequired {
            model: repo.to_string(),
            repo: repo.to_string(),
        },
        404 => ModelManagerError::NotFound {
            repo: repo.to_string(),
            file: file.to_string(),
        },
        code => ModelManagerError::NetworkError {
            attempts: 1,
            source: format!("HTTP {code} from {repo}/{file}").into(),
        },
    }
}

/// Adapter between hf-hub's [`Progress`] trait and our `ProgressFn`
/// callback. The adapter is cloned across chunk workers (hf-hub drives
/// concurrent chunked downloads), so `Clone` is mandatory — we share
/// state via `Arc<Mutex<..>>` to keep the running byte total accurate.
#[derive(Clone)]
struct ProgressAdapter {
    file: String,
    progress: ProgressFn,
    state: Arc<Mutex<AdapterState>>,
}

#[derive(Default)]
struct AdapterState {
    total: u64,
    done: u64,
}

impl ProgressAdapter {
    fn new(file: String, progress: ProgressFn) -> Self {
        Self {
            file,
            progress,
            state: Arc::new(Mutex::new(AdapterState::default())),
        }
    }
}

impl Progress for ProgressAdapter {
    async fn init(&mut self, size: usize, filename: &str) {
        {
            let mut s = self.state.lock().unwrap();
            s.total = size as u64;
            s.done = 0;
        }
        (self.progress)(ProgressEvent::Start {
            file: filename.to_string(),
            bytes_total: size as u64,
        });
    }

    async fn update(&mut self, size: usize) {
        let (done, total) = {
            let mut s = self.state.lock().unwrap();
            s.done = s.done.saturating_add(size as u64);
            (s.done, s.total)
        };
        (self.progress)(ProgressEvent::Update {
            file: self.file.clone(),
            bytes_done: done,
            bytes_total: total,
        });
    }

    async fn finish(&mut self) {
        let total = self.state.lock().unwrap().total;
        (self.progress)(ProgressEvent::Done {
            file: self.file.clone(),
            bytes_total: total,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn new_creates_hf_cache_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let _client = HfHubClient::new(tmp.path(), None).expect("build client");
        assert!(tmp.path().join(HF_CACHE_SUBDIR).exists());
    }

    #[test]
    fn new_accepts_token() {
        let tmp = tempfile::tempdir().unwrap();
        let _client =
            HfHubClient::new(tmp.path(), Some("hf_dummytoken".to_string())).expect("build client");
        // No easy way to assert the token was applied short of issuing a
        // request; the happy-path construction is the signal we care about.
    }

    #[tokio::test]
    async fn progress_adapter_forwards_start_update_done() {
        let events: Arc<std::sync::Mutex<Vec<ProgressEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let ev_cb = events.clone();
        let cb: ProgressFn = Arc::new(move |ev| ev_cb.lock().unwrap().push(ev));
        let mut adapter = ProgressAdapter::new("model.safetensors".into(), cb);

        adapter.init(1_000, "model.safetensors").await;
        adapter.update(400).await;
        adapter.update(600).await;
        adapter.finish().await;

        let got = events.lock().unwrap();
        assert!(matches!(
            &got[0],
            ProgressEvent::Start {
                bytes_total: 1000,
                ..
            }
        ));
        assert!(matches!(
            &got[1],
            ProgressEvent::Update {
                bytes_done: 400,
                bytes_total: 1000,
                ..
            }
        ));
        assert!(matches!(
            &got[2],
            ProgressEvent::Update {
                bytes_done: 1000,
                bytes_total: 1000,
                ..
            }
        ));
        assert!(matches!(
            &got[3],
            ProgressEvent::Done {
                bytes_total: 1000,
                ..
            }
        ));
    }
}
