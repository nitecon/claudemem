//! Inference backend abstraction + a candle-backed implementation.
//!
//! The dream pipeline depends on exactly one capability: given a prompt,
//! return a completion string. That narrow surface is modeled by the
//! [`Inference`] trait, which lets tests swap in a deterministic stub
//! without ever loading a multi-GB gemma3 checkpoint.
//!
//! Two impls live in this file:
//!
//! * [`FixedInference`] — returns a canned string for every call, used by the
//!   dream unit tests. No dependencies on candle or model files.
//! * [`CandleInference`] — wraps `candle-core` + `candle-transformers`. Loads
//!   gemma3 weights from the local model cache; fails with a structured error
//!   when the cache is empty (the user must run `memory-dream --pull` first).
//!
//! The candle path is deliberately *thin*. Anything that needs a real inference
//! pass is a user action; unit tests never invoke it. The surface exists so
//! the rest of the pipeline (prompt construction, response parsing, dedup)
//! compiles and exercises against the trait without touching network or GPU.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Errors surfaced by any [`Inference`] implementation.
///
/// Grouped so the dream orchestrator can distinguish "transient — retry
/// later" (`Io`) from "misconfigured — user must act" (`ModelMissing`) from
/// "response is unusable — fall back to raw" (`Refusal`, `ParseError`).
#[derive(Debug, Error)]
pub enum InferenceError {
    /// The configured model directory does not contain the expected files.
    /// Surfaced by `CandleInference::new` when a user runs `memory-dream`
    /// before `memory-dream --pull`.
    #[error("model not found at {path}: {detail}")]
    ModelMissing { path: PathBuf, detail: String },

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

    /// Unexpected underlying IO / tokenizer / candle failure.
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

/// Real candle-backed inference for gemma3.
///
/// Construction loads the model from `model_dir`. The actual generation
/// implementation is intentionally stubbed out below — loading gemma3 and
/// running a full sampling loop requires multi-GB weights the user must
/// pull explicitly, and wiring the candle sampling loop here without a
/// tested checkpoint would be a guess. The stub surface exists so the rest
/// of the pipeline compiles and the CLI can expose `--dry-run` today; a
/// follow-up user-driven pass lands the real sampling loop once gemma3
/// weights have been downloaded and validated end-to-end.
#[derive(Debug)]
pub struct CandleInference {
    #[allow(dead_code)]
    model_dir: PathBuf,
}

impl CandleInference {
    /// Load gemma3 from `model_dir`. Returns `ModelMissing` when the
    /// expected files (config.json, tokenizer.json, safetensors shards)
    /// are absent so the caller can nudge the user toward `--pull`.
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config = model_dir.join("config.json");
        let tokenizer = model_dir.join("tokenizer.json");
        if !config.exists() || !tokenizer.exists() {
            return Err(InferenceError::ModelMissing {
                path: model_dir.clone(),
                detail: format!(
                    "expected config.json and tokenizer.json; got config={} tokenizer={}",
                    config.exists(),
                    tokenizer.exists()
                ),
            });
        }
        // TODO(dream): wire up the candle gemma3 load + sampling loop once
        // a real checkpoint has been pulled and validated by the user. For
        // now we accept the directory shape and defer actual model loading
        // so `memory-dream --dry-run` can operate without ever touching
        // candle's multi-GB allocation path.
        Ok(Self { model_dir })
    }
}

impl Inference for CandleInference {
    fn generate(&self, _prompt: &str, _max_tokens: u32) -> Result<String, InferenceError> {
        // Honest stub: the real sampling loop is a follow-up user action
        // (requires gemma3 weights present and a validated generation path).
        // Surfacing this as ModelMissing keeps the orchestrator's fallback
        // behavior correct — the pass falls back to dedup-only on error
        // per the `memory-dream --dry-run` documented behavior.
        Err(InferenceError::ModelMissing {
            path: self.model_dir.clone(),
            detail: "candle generation path not wired; run --pull and an \
                     end-to-end validation pass before enabling condensation"
                .to_string(),
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
    fn candle_inference_errors_on_missing_model_dir() {
        let err = CandleInference::new("/tmp/definitely-not-a-real-path-xyz").unwrap_err();
        match err {
            InferenceError::ModelMissing { .. } => {}
            other => panic!("expected ModelMissing, got {other:?}"),
        }
    }
}
