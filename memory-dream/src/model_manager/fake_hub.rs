//! Test-only fake [`HubClient`] that reads bytes from an in-memory map
//! instead of hitting the network.
//!
//! Designed to keep `cargo test` fully offline while still exercising
//! every branch of [`super::pull_model`]: happy path, auth failure, 404,
//! transient retries (with recovery), retry exhaustion, and partial
//! resume (the pull flow skips already-present files at the filesystem
//! level, so "resume" here just means the fake client isn't called for
//! a file that already exists on disk).
//!
//! The fake writes its output to a `tempfile::TempDir` owned by the
//! client so each test has an isolated staging area — the caller of
//! [`super::pull_model`] then `std::fs::copy`s from that staging path
//! into the real cache layout, matching what [`super::hub::HfHubClient`]
//! does with hf-hub's snapshot directory.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};

use async_trait::async_trait;

use super::{FileMetadata, HubClient, ModelManagerError, ProgressEvent, ProgressFn};

/// Deterministic fake HF client. Construct with a map of `filename →
/// body`; every requested file must be present or the client surfaces
/// a configurable error.
pub struct FakeHubClient {
    files: HashMap<String, Vec<u8>>,
    mode: Mode,
    /// Staging directory — the fake writes downloaded bodies here so the
    /// caller can copy them into its own layout. Kept alive for the
    /// lifetime of the client.
    staging: tempfile::TempDir,
    /// Tracks how many times `download_to` has been invoked. Used by
    /// tests that assert "resume skipped N files".
    call_counter: Arc<AtomicU32>,
    /// Per-file attempt counters. Used by the transient-then-succeed
    /// mode — for each file, the first `first_n_fail` calls return a
    /// transient NetworkError and the (N+1)th call succeeds.
    attempt_counters: Arc<Mutex<HashMap<String, u32>>>,
    /// Per-file advertised SHA-256 hash. `None` means "no etag" (the hub
    /// didn't return a usable hash for this file — skip verification).
    /// `Some(hash)` means the hub advertised this hash; pull_model will
    /// compare it to what it computes on the downloaded bytes.
    etags: HashMap<String, Option<String>>,
}

#[derive(Clone, Debug)]
enum Mode {
    /// Always succeed (or return NotFound if the file isn't in the map).
    Normal,
    /// Return AuthRequired for every call.
    AuthRequired,
    /// Return NotFound for every call, regardless of map contents.
    NotFound,
    /// Return NetworkError on every call — simulates a persistent
    /// transient fault so the retry loop exhausts its attempts.
    AlwaysTransient,
    /// Return NetworkError for the first `n` calls on each file, then
    /// succeed. Models "flaky network, recovers after a beat".
    TransientThenOk { first_n_fail: u32 },
}

impl FakeHubClient {
    pub fn new(files: HashMap<String, Vec<u8>>) -> Self {
        Self {
            files,
            mode: Mode::Normal,
            staging: tempfile::tempdir().expect("staging tempdir"),
            call_counter: Arc::new(AtomicU32::new(0)),
            attempt_counters: Arc::new(Mutex::new(HashMap::new())),
            etags: HashMap::new(),
        }
    }

    /// Attach a SHA-256 hex string (or `None` for "no etag available") to
    /// `file`. Tests use this to drive the checksum verification paths:
    ///   * happy path — set `Some(correct_hash)`.
    ///   * mismatch  — set `Some(wrong_hash)`; pull should delete + error.
    ///   * skipped   — set `None` (default); pull should emit skipped
    ///     event and succeed.
    pub fn with_etag(mut self, file: &str, etag: Option<String>) -> Self {
        self.etags.insert(file.to_string(), etag);
        self
    }

    pub fn with_auth_required(mut self) -> Self {
        self.mode = Mode::AuthRequired;
        self
    }

    pub fn with_not_found(mut self) -> Self {
        self.mode = Mode::NotFound;
        self
    }

    pub fn with_always_transient(mut self) -> Self {
        self.mode = Mode::AlwaysTransient;
        self
    }

    pub fn with_transient_first_attempts(mut self, first_n_fail: u32) -> Self {
        self.mode = Mode::TransientThenOk { first_n_fail };
        self
    }

    /// Total calls made to `download_to` across all files. Used by the
    /// resume test to assert pre-existing files aren't re-fetched.
    pub fn call_count(&self) -> usize {
        self.call_counter.load(Ordering::SeqCst) as usize
    }

    /// Materialize a file body into the staging dir and return the
    /// path. Separated from `download_to` so the success path is a
    /// straight line at the call site.
    fn materialize(&self, file: &str) -> Result<PathBuf, ModelManagerError> {
        let body = self
            .files
            .get(file)
            .ok_or_else(|| ModelManagerError::NotFound {
                repo: "fake".to_string(),
                file: file.to_string(),
            })?;
        let out = self.staging.path().join(file);
        std::fs::write(&out, body)?;
        Ok(out)
    }
}

#[async_trait]
impl HubClient for FakeHubClient {
    async fn metadata(&self, _repo: &str, file: &str) -> Result<FileMetadata, ModelManagerError> {
        let size = self.files.get(file).map(|b| b.len() as u64).unwrap_or(0);
        let expected_sha256 = self.etags.get(file).cloned().unwrap_or(None);
        Ok(FileMetadata {
            size,
            expected_sha256,
        })
    }

    async fn download_to(
        &self,
        _repo: &str,
        file: &str,
        progress: ProgressFn,
    ) -> Result<PathBuf, ModelManagerError> {
        self.call_counter.fetch_add(1, Ordering::SeqCst);

        match self.mode.clone() {
            Mode::AuthRequired => Err(ModelManagerError::AuthRequired {
                model: "fake".to_string(),
                repo: "fake".to_string(),
            }),
            Mode::NotFound => Err(ModelManagerError::NotFound {
                repo: "fake".to_string(),
                file: file.to_string(),
            }),
            Mode::AlwaysTransient => Err(ModelManagerError::NetworkError {
                attempts: 1,
                source: "fake transient".into(),
            }),
            Mode::TransientThenOk { first_n_fail } => {
                let attempts_for_file = {
                    let mut map = self.attempt_counters.lock().unwrap();
                    let c = map.entry(file.to_string()).or_insert(0);
                    *c += 1;
                    *c
                };
                if attempts_for_file <= first_n_fail {
                    return Err(ModelManagerError::NetworkError {
                        attempts: attempts_for_file,
                        source: "fake transient (recovering)".into(),
                    });
                }
                emit_progress(
                    &progress,
                    file,
                    self.files.get(file).map(|b| b.len()).unwrap_or(0),
                );
                self.materialize(file)
            }
            Mode::Normal => {
                emit_progress(
                    &progress,
                    file,
                    self.files.get(file).map(|b| b.len()).unwrap_or(0),
                );
                self.materialize(file)
            }
        }
    }
}

/// Emit Start/Update/Done in quick succession so tests that assert the
/// progress sequence see the same shape as the real hf-hub adapter.
fn emit_progress(progress: &ProgressFn, file: &str, size: usize) {
    let total = size as u64;
    progress(ProgressEvent::Start {
        file: file.to_string(),
        bytes_total: total,
    });
    progress(ProgressEvent::Update {
        file: file.to_string(),
        bytes_done: total,
        bytes_total: total,
    });
    progress(ProgressEvent::Done {
        file: file.to_string(),
        bytes_total: total,
    });
}
