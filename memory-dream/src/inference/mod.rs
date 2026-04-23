//! Inference backend abstraction + a candle-backed implementation.
//!
//! The dream pipeline depends on exactly one capability: given a prompt,
//! return a completion string. That narrow surface is modeled by the
//! [`Inference`] trait, which lets tests swap in a deterministic stub
//! without ever loading a multi-GB checkpoint.
//!
//! Impls in this module tree:
//!
//! * [`FixedInference`] — returns a canned string for every call; the dream
//!   unit tests rely on it to exercise the full condense → parse → dedup
//!   pipeline without ever touching candle or a tokenizer.
//! * [`NoopInference`] — always returns [`InferenceError::ModelMissing`];
//!   the CLI drops to this when the candle backend can't initialize so the
//!   dream orchestrator still runs a dedup-only pass.
//! * [`CandleInference`] — real in-process inference. Loads a tokenizer and
//!   model weights from disk, runs a KV-cache-aware sampling loop, and
//!   returns the decoded completion. Lives in [`candle_backend`] so the
//!   heavy dependency surface is scoped to one file.
//! * [`HeadlessInference`] — spawns an external CLI (e.g. `claude -p`) as
//!   an inference source. Kept separate so a user without a local model
//!   can still condense memories through whatever CLI they already have.

use std::path::{Path, PathBuf};

use thiserror::Error;

pub mod candle_backend;
pub mod device;
pub mod headless;

pub use candle_backend::CandleInference;
pub use device::{resolve_device, DevicePreference};
pub use headless::HeadlessInference;

/// Errors surfaced by any [`Inference`] implementation.
///
/// Grouped so the dream orchestrator can distinguish:
///   * "transient — retry later" (`GenerationFailed`, `Io`)
///   * "misconfigured — user must act" (`ModelMissing`, `LoadFailed`,
///     `ArchUnsupported`, `DeviceUnavailable`)
///   * "response is unusable — fall back to raw" (`Refusal`, `ParseError`)
#[derive(Debug, Error)]
pub enum InferenceError {
    /// The configured model directory does not contain the expected files.
    /// Surfaced by `CandleInference::new` when a user runs `memory-dream`
    /// before `memory-dream --pull`.
    #[error("model not found at {path}: {detail}")]
    ModelMissing { path: PathBuf, detail: String },

    /// Loading one of the model artifacts failed. Distinct from
    /// `ModelMissing` — the file exists but is malformed, truncated, a
    /// wrong version, or a safetensors file with a shape mismatch against
    /// the declared config. Carries the path so the CLI can tell the user
    /// which asset to re-pull.
    #[error("failed to load {file}: {reason}")]
    LoadFailed { file: PathBuf, reason: String },

    /// The config's `model_type` is recognized by HuggingFace but not yet
    /// wired into our dispatch table. Surfacing this cleanly lets us land
    /// architectures one at a time without forking the CLI.
    #[error("unsupported model architecture '{0}'; supported: gemma3, llama (TinyLlama, SmolLM)")]
    ArchUnsupported(String),

    /// The caller asked for a device (Metal/CUDA) that isn't usable on this
    /// host. Only raised when the user *explicitly* selected the device —
    /// the `Auto` preference silently falls back to CPU instead.
    #[error("device '{requested}' unavailable: {detail}")]
    DeviceUnavailable { requested: String, detail: String },

    /// The model produced a response but it could not be parsed into the
    /// structured shape dream expects (e.g. the JSON envelope was malformed).
    /// The dream pipeline falls back to keeping the raw memory when it sees
    /// this variant.
    #[error("failed to parse model response: {0}")]
    ParseError(String),

    /// The model refused the task (safety filter, instruction override, etc).
    /// Detected heuristically by the condense module; dream keeps the raw
    /// memory when this surfaces.
    #[error("model refused: {0}")]
    Refusal(String),

    /// A runtime failure during the candle forward pass or sampling loop.
    /// Backstop for any error thrown by candle/tokenizers mid-generation.
    #[error("generation failed: {0}")]
    GenerationFailed(String),

    /// Unexpected underlying IO / tokenizer / candle failure outside of the
    /// generation hot path (e.g. subprocess spawn for HeadlessInference).
    #[error("inference backend error: {0}")]
    Io(String),
}

/// The minimal contract a dream inference backend must satisfy.
///
/// Synchronous on purpose: the dream orchestrator is a one-shot CLI that
/// processes memories sequentially, and async would only complicate the
/// transaction-per-memory flow without buying any throughput (candle's
/// generation loop is already GPU-bound).
pub trait Inference: Send + Sync {
    /// Run the model against `prompt` and return up to `max_tokens` output
    /// tokens as a decoded string. Implementations must respect `max_tokens`
    /// as an upper bound — the dream pipeline relies on it to cap response
    /// length and avoid runaway generation.
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, InferenceError>;
}

/// Test double — returns a canned response for every call.
///
/// Lets the dream unit tests exercise the full condense → parse → dedup
/// pipeline without loading any real weights. Construct with the string
/// the stub should return; every `generate` call returns a clone of it.
pub struct FixedInference {
    response: String,
}

impl FixedInference {
    /// Create a stub that returns `response` for every `generate` call.
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

impl Inference for FixedInference {
    fn generate(&self, _prompt: &str, _max_tokens: u32) -> Result<String, InferenceError> {
        Ok(self.response.clone())
    }
}

/// Always returns [`InferenceError::ModelMissing`] — used by the CLI when
/// the candle backend couldn't initialize (e.g. no model pulled yet) so the
/// dream orchestrator can still run a dedup-only fallback pass. Every
/// `generate` call surfaces the same error, which the condense module
/// converts into "keep the raw content" per its fallback contract.
#[derive(Debug)]
pub struct NoopInference {
    reason: String,
}

impl NoopInference {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl Inference for NoopInference {
    fn generate(&self, _prompt: &str, _max_tokens: u32) -> Result<String, InferenceError> {
        Err(InferenceError::ModelMissing {
            path: Path::new("").to_path_buf(),
            detail: self.reason.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_inference_returns_canned_response() {
        let i = FixedInference::new("hello world");
        let out = i.generate("ignored prompt", 32).unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn noop_inference_returns_model_missing() {
        let i = NoopInference::new("no model on disk");
        let err = i.generate("x", 1).unwrap_err();
        match err {
            InferenceError::ModelMissing { detail, .. } => {
                assert!(detail.contains("no model"));
            }
            other => panic!("expected ModelMissing, got {other:?}"),
        }
    }

    #[test]
    fn candle_inference_errors_on_missing_model_dir() {
        // Smoke: construction against a non-existent path surfaces a clean
        // ModelMissing (no candle load is attempted). Detailed load-failure
        // tests live in `candle_backend::tests`.
        let err =
            CandleInference::new("/tmp/definitely-not-a-real-path-xyz", DevicePreference::Cpu)
                .unwrap_err();
        match err {
            InferenceError::ModelMissing { .. } => {}
            other => panic!("expected ModelMissing, got {other:?}"),
        }
    }
}
