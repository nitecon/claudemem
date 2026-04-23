//! Real candle-backed inference with architecture dispatch.
//!
//! The heavy lifting for the dream local backend. Given a model directory
//! produced by `memory-dream --pull`, this module:
//!
//! 1. Reads `config.json` once to discover the architecture (`model_type`,
//!    falling back to `architectures[0]`). See [`detect_architecture`].
//! 2. Dispatches to the matching candle `Model` / `Llama` implementation
//!    (gemma3 or llama). Unsupported architectures surface
//!    [`InferenceError::ArchUnsupported`] so landing new ones is a small
//!    localized change rather than a rewrite.
//! 3. Builds a tokenizer from `tokenizer.json`.
//! 4. Memory-maps `model.safetensors` into a [`VarBuilder`] and instantiates
//!    the model on the resolved device.
//! 5. On each [`Inference::generate`] call, runs a token-at-a-time sampling
//!    loop with KV-cache, greedy decoding (temperature = 0), and an EOS /
//!    JSON-envelope stop condition.
//!
//! The sampling loop is deliberately greedy: the dream condensation prompt
//! asks for a single-key JSON object, and determinism is essential —
//! non-greedy sampling introduces drift that breaks the parse contract.
//!
//! ## Why a `Backend` enum rather than `Box<dyn>`?
//!
//! The two supported architectures don't share a trait in candle (`gemma3`
//! and `llama` expose different `forward` signatures — llama needs a mutable
//! `Cache`, gemma3 holds its own state via `&mut self`). Wrapping them in a
//! local `enum` keeps the dispatch explicit and lets the `Send + Sync` bound
//! on [`Inference`] compose cleanly via an interior `Mutex`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::{
    gemma3 as gemma3_mod,
    llama::{self as llama_mod, Cache as LlamaCache, LlamaConfig},
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::{resolve_device, DevicePreference, Inference, InferenceError};

/// Required files on disk before a model can load. Kept here (not in
/// `model_manager`) because the inference side owns the "what must be
/// present to actually run" contract; the pull flow uses its own list
/// via `model_manager::required_files_for`.
const REQUIRED_LOCAL_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

/// Stop conditions for the generation loop.
///
/// * `MaxTokens` — produced `max_tokens` and stopped.
/// * `Eos`      — model emitted its end-of-sequence token.
/// * `JsonClose` — a heuristic for the condensation prompt: as soon as a
///   closing `}` arrives at the outermost JSON depth we stop, since the
///   dream envelope is a single-key `{"condensed": "..."}` object and
///   anything past the close is garbage that would only slow us down.
#[derive(Debug, Clone, Copy)]
enum StopReason {
    MaxTokens,
    Eos,
    JsonClose,
}

/// Sampling seed. Greedy decoding doesn't actually consume the RNG but
/// `LogitsProcessor::from_sampling` wants a seed for construction.
const SAMPLING_SEED: u64 = 0;

/// Light-weight `config.json` shape we parse just to detect the
/// architecture. Full config deserialization happens in the arch-specific
/// branch afterward.
#[derive(Debug, Deserialize)]
struct ArchDetectConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
}

/// Stable architecture identifier dispatched by the loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Architecture {
    Gemma3,
    Llama,
}

/// Peek at `config.json` and return the dispatched architecture.
///
/// Order of precedence mirrors what HF transformers does:
///   1. `model_type` (the canonical field).
///   2. `architectures[0]`, lowercased and stripped of the trailing
///      `ForCausalLM` suffix (`LlamaForCausalLM` → `llama`).
///
/// Surfaces [`InferenceError::ArchUnsupported`] for anything outside the
/// current {gemma3, llama} set so future architectures land as new dispatch
/// arms rather than silent misroutes.
pub(crate) fn detect_architecture(config_path: &Path) -> Result<Architecture, InferenceError> {
    let body = std::fs::read_to_string(config_path).map_err(|e| InferenceError::LoadFailed {
        file: config_path.to_path_buf(),
        reason: format!("read config.json: {e}"),
    })?;
    let c: ArchDetectConfig =
        serde_json::from_str(&body).map_err(|e| InferenceError::LoadFailed {
            file: config_path.to_path_buf(),
            reason: format!("parse config.json: {e}"),
        })?;

    let raw = c
        .model_type
        .clone()
        .or_else(|| c.architectures.first().cloned())
        .ok_or_else(|| {
            InferenceError::LoadFailed {
                file: config_path.to_path_buf(),
                reason: "config.json has neither `model_type` nor `architectures[]`"
                    .to_string(),
            }
        })?;

    // Normalize: lowercase, strip trailing "forcausallm" if present.
    let normalized = raw
        .to_lowercase()
        .trim_end_matches("forcausallm")
        .to_string();

    match normalized.as_str() {
        "gemma3" => Ok(Architecture::Gemma3),
        "llama" => Ok(Architecture::Llama),
        other => Err(InferenceError::ArchUnsupported(other.to_string())),
    }
}

/// Architecture-specific model state. Owning the model + its KV cache here
/// (rather than behind a `Box<dyn>`) keeps the dispatch explicit — the
/// candle crates don't offer a shared generation trait, so we'd have to
/// invent one anyway.
///
/// The llama variant caches the parsed `Config` so we can re-initialize the
/// `Cache` for every generation — carrying a stale KV cache across calls
/// would bias subsequent prompts with tokens from a previous one.
enum Backend {
    Gemma3 {
        model: gemma3_mod::Model,
        eos_token_id: Option<u32>,
    },
    Llama {
        model: llama_mod::Llama,
        config: llama_mod::Config,
        eos_token_id: Option<u32>,
    },
}

/// Real candle-backed inference.
///
/// Constructed once per process — weights stay mmapped for the lifetime of
/// the instance. `generate` is `&self` per the [`Inference`] contract, so
/// the mutable model state (KV cache, logits processor) lives behind an
/// interior [`Mutex`]. The dream orchestrator is sequential so there's no
/// contention.
pub struct CandleInference {
    model_dir: PathBuf,
    tokenizer: Tokenizer,
    device: Device,
    state: Mutex<GenState>,
    bos_token_id: Option<u32>,
    arch_label: &'static str,
}

/// Mutable state guarded by the [`CandleInference::state`] mutex.
struct GenState {
    backend: Backend,
    logits_processor: LogitsProcessor,
}

impl std::fmt::Debug for CandleInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleInference")
            .field("model_dir", &self.model_dir)
            .field("arch", &self.arch_label)
            .field("device", &format!("{:?}", self.device))
            .finish()
    }
}

impl CandleInference {
    /// Load a model from `model_dir` onto the device resolved from `pref`.
    ///
    /// Fast checks first (directory present, required files there) so the
    /// user sees a `ModelMissing` error before we pay the cost of mmapping
    /// a multi-GB safetensors file. Architecture detection happens next;
    /// the heavy loader is only invoked after we know which one to run.
    pub fn new(
        model_dir: impl AsRef<Path>,
        pref: DevicePreference,
    ) -> Result<Self, InferenceError> {
        let model_dir = model_dir.as_ref().to_path_buf();
        if !model_dir.is_dir() {
            return Err(InferenceError::ModelMissing {
                path: model_dir.clone(),
                detail: "model directory does not exist".to_string(),
            });
        }

        for f in REQUIRED_LOCAL_FILES {
            if !model_dir.join(f).exists() {
                return Err(InferenceError::ModelMissing {
                    path: model_dir.clone(),
                    detail: format!("missing required file '{f}'"),
                });
            }
        }

        let device = resolve_device(pref)?;
        let arch = detect_architecture(&model_dir.join("config.json"))?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| InferenceError::LoadFailed {
                file: tokenizer_path.clone(),
                reason: format!("tokenizer load: {e}"),
            })?;

        let weights_path = model_dir.join("model.safetensors");
        // Safety: mmap is inherently unsafe because the file can mutate
        // under the mapping. The dream flow treats the cache directory as
        // read-only — no concurrent writer. The perf win (avoiding a full
        // multi-GB read into RAM) is worth the contract.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
        }
        .map_err(|e| InferenceError::LoadFailed {
            file: weights_path.clone(),
            reason: format!("load safetensors: {e}"),
        })?;

        let config_path = model_dir.join("config.json");
        let (backend, bos_token_id, arch_label) = match arch {
            Architecture::Gemma3 => load_gemma3(&config_path, vb, &device)?,
            Architecture::Llama => load_llama(&config_path, vb, &device)?,
        };

        // Greedy decoding. Temperature=0.0 means `LogitsProcessor` takes the
        // `ArgMax` branch and doesn't consume its RNG — deterministic across
        // runs, which the JSON envelope parse contract relies on.
        let logits_processor = LogitsProcessor::from_sampling(SAMPLING_SEED, Sampling::ArgMax);

        Ok(Self {
            model_dir,
            tokenizer,
            device,
            bos_token_id,
            arch_label,
            state: Mutex::new(GenState {
                backend,
                logits_processor,
            }),
        })
    }

    /// Access the underlying device (used by tests to assert device wiring).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Architecture short-name — `"gemma3"` or `"llama"`. Used by the CLI
    /// for `<test_result>` output labeling.
    pub fn architecture(&self) -> &'static str {
        self.arch_label
    }
}

/// Gemma3 loader: deserialize `config.json` into the arch-specific `Config`,
/// instantiate the model, read the EOS token id from the tokenizer config
/// (falls back to the config's own hint if present).
fn load_gemma3(
    config_path: &Path,
    vb: VarBuilder,
    _device: &Device,
) -> Result<(Backend, Option<u32>, &'static str), InferenceError> {
    let body =
        std::fs::read_to_string(config_path).map_err(|e| InferenceError::LoadFailed {
            file: config_path.to_path_buf(),
            reason: format!("read gemma3 config: {e}"),
        })?;
    let cfg: gemma3_mod::Config =
        serde_json::from_str(&body).map_err(|e| InferenceError::LoadFailed {
            file: config_path.to_path_buf(),
            reason: format!("parse gemma3 config: {e}"),
        })?;

    let model = gemma3_mod::Model::new(false, &cfg, vb).map_err(|e| {
        InferenceError::LoadFailed {
            file: config_path.to_path_buf(),
            reason: format!("gemma3 model load: {e}"),
        }
    })?;

    let eos_token_id = lookup_eos_from_config(config_path).ok().flatten();
    let bos_token_id = lookup_bos_from_config(config_path).ok().flatten();
    Ok((
        Backend::Gemma3 {
            model,
            eos_token_id,
        },
        bos_token_id,
        "gemma3",
    ))
}

/// Llama loader: the config is stored as `LlamaConfig` on disk (the HF
/// format) and then converted to the runtime `Config` before model load.
/// Both TinyLlama and SmolLM live on this path.
fn load_llama(
    config_path: &Path,
    vb: VarBuilder,
    device: &Device,
) -> Result<(Backend, Option<u32>, &'static str), InferenceError> {
    let body =
        std::fs::read_to_string(config_path).map_err(|e| InferenceError::LoadFailed {
            file: config_path.to_path_buf(),
            reason: format!("read llama config: {e}"),
        })?;
    let llama_config: LlamaConfig =
        serde_json::from_str(&body).map_err(|e| InferenceError::LoadFailed {
            file: config_path.to_path_buf(),
            reason: format!("parse llama config: {e}"),
        })?;

    let cfg = llama_config.into_config(false);
    let bos_token_id = cfg.bos_token_id;
    let eos_token_id = match &cfg.eos_token_id {
        Some(llama_mod::LlamaEosToks::Single(id)) => Some(*id),
        Some(llama_mod::LlamaEosToks::Multiple(ids)) => ids.first().copied(),
        None => None,
    };
    // Smoke-load the cache once during construction so malformed configs
    // surface early. The real per-call cache is re-created inside
    // `generate` so no KV state leaks between prompts.
    LlamaCache::new(true, DType::F32, &cfg, device).map_err(|e| InferenceError::LoadFailed {
        file: config_path.to_path_buf(),
        reason: format!("llama cache init: {e}"),
    })?;
    let model = llama_mod::Llama::load(vb, &cfg).map_err(|e| InferenceError::LoadFailed {
        file: config_path.to_path_buf(),
        reason: format!("llama model load: {e}"),
    })?;
    Ok((
        Backend::Llama {
            model,
            config: cfg,
            eos_token_id,
        },
        bos_token_id,
        "llama",
    ))
}

/// Partial shape of `config.json` for EOS/BOS lookups — some configs store
/// the ids here even though they're primarily in `tokenizer_config.json`.
/// We try config first (most reliable across forks) and the caller falls
/// back to the arch-config hints when this returns None.
#[derive(Debug, Deserialize)]
struct TokenIdsProbe {
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    bos_token_id: Option<serde_json::Value>,
}

fn lookup_eos_from_config(config_path: &Path) -> Result<Option<u32>, InferenceError> {
    let body = std::fs::read_to_string(config_path).map_err(|e| InferenceError::LoadFailed {
        file: config_path.to_path_buf(),
        reason: format!("read config: {e}"),
    })?;
    let probe: TokenIdsProbe = serde_json::from_str(&body).unwrap_or(TokenIdsProbe {
        eos_token_id: None,
        bos_token_id: None,
    });
    Ok(extract_u32(probe.eos_token_id))
}

fn lookup_bos_from_config(config_path: &Path) -> Result<Option<u32>, InferenceError> {
    let body = std::fs::read_to_string(config_path).map_err(|e| InferenceError::LoadFailed {
        file: config_path.to_path_buf(),
        reason: format!("read config: {e}"),
    })?;
    let probe: TokenIdsProbe = serde_json::from_str(&body).unwrap_or(TokenIdsProbe {
        eos_token_id: None,
        bos_token_id: None,
    });
    Ok(extract_u32(probe.bos_token_id))
}

/// EOS/BOS ids can be a single number or an array of numbers. Extract the
/// first usable u32 so either shape loads cleanly.
fn extract_u32(v: Option<serde_json::Value>) -> Option<u32> {
    match v? {
        serde_json::Value::Number(n) => n.as_u64().map(|v| v as u32),
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .find_map(|x| x.as_u64().map(|v| v as u32)),
        _ => None,
    }
}

impl Inference for CandleInference {
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, InferenceError> {
        // Tokenize. `add_special_tokens = true` lets the tokenizer add the
        // BOS token if the tokenizer_config asks for it; we also have a
        // manual prepend path below for tokenizers that don't.
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| InferenceError::GenerationFailed(format!("tokenize prompt: {e}")))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Some tokenizers (notably Llama-based ones without a post-processor
        // configured) don't auto-prepend BOS. Guarantee it's there exactly
        // once — candle's llama cache assumes a leading BOS for correct
        // position-encoding bookkeeping.
        if let Some(bos) = self.bos_token_id {
            if tokens.first().copied() != Some(bos) {
                tokens.insert(0, bos);
            }
        }

        let mut state = self
            .state
            .lock()
            .map_err(|e| InferenceError::GenerationFailed(format!("state lock poisoned: {e}")))?;

        // Fresh KV cache per generation so prompts are independent. Gemma3
        // holds its state via `&mut self` on the model, so we pass
        // `pos_offset` only; the llama branch owns its Cache here.
        let mut llama_cache = if let Backend::Llama { config, .. } = &state.backend {
            Some(
                LlamaCache::new(true, DType::F32, config, &self.device).map_err(|e| {
                    InferenceError::GenerationFailed(format!("llama cache alloc: {e}"))
                })?,
            )
        } else {
            None
        };

        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens as usize);
        let mut output_text = String::new();
        let mut json_depth: i32 = 0;
        let mut json_opened = false;
        let mut stop_reason = StopReason::MaxTokens;

        // Prefill: run the whole prompt through the model once to populate
        // the KV cache (for llama) or set up the internal state (for gemma3).
        // Then loop one token at a time.
        let mut ctx_tokens = tokens.clone();
        let mut pos_offset: usize = 0;

        for step in 0..max_tokens {
            let input = Tensor::new(ctx_tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| InferenceError::GenerationFailed(format!("build input: {e}")))?;

            let logits = forward(
                &mut state.backend,
                &input,
                pos_offset,
                llama_cache.as_mut(),
            )?;

            // After normalization below, logits are (vocab,). The LogitsProcessor
            // API wants a 1-D tensor of per-token scores.
            let logits = logits
                .to_dtype(DType::F32)
                .map_err(|e| InferenceError::GenerationFailed(format!("cast logits: {e}")))?;

            let next_id = state
                .logits_processor
                .sample(&logits)
                .map_err(|e| InferenceError::GenerationFailed(format!("sample: {e}")))?;

            generated_ids.push(next_id);

            if let Some(eos) = eos_token_id(&state.backend) {
                if next_id == eos {
                    stop_reason = StopReason::Eos;
                    break;
                }
            }

            // Incremental decode so we can update our JSON-depth counter
            // and bail as soon as the envelope closes.
            let piece = self
                .tokenizer
                .decode(&[next_id], false)
                .map_err(|e| InferenceError::GenerationFailed(format!("decode token: {e}")))?;
            output_text.push_str(&piece);
            for ch in piece.chars() {
                match ch {
                    '{' => {
                        json_depth += 1;
                        json_opened = true;
                    }
                    '}' => {
                        json_depth -= 1;
                        if json_opened && json_depth <= 0 {
                            stop_reason = StopReason::JsonClose;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if matches!(stop_reason, StopReason::JsonClose) {
                break;
            }

            // Next iteration: feed only the last token and advance the
            // position offset by the current prefill size. Llama's cache
            // picks up from pos_offset; gemma3 maintains its own internal
            // counter but still needs `seqlen_offset`.
            pos_offset += ctx_tokens.len();
            ctx_tokens = vec![next_id];

            if step + 1 == max_tokens {
                stop_reason = StopReason::MaxTokens;
            }
        }

        tracing::debug!(
            stop = ?stop_reason,
            tokens = generated_ids.len(),
            "candle generation finished"
        );

        Ok(output_text)
    }
}

/// Invoke the architecture-specific forward pass and normalize the output
/// to a 1-D `(vocab,)` tensor for [`LogitsProcessor::sample`].
///
/// Gemma3 returns `(batch=1, seq=1, vocab)` (it already slices off the
/// final timestep internally). Llama returns `(batch=1, vocab)` because its
/// forward also slices to the last timestep before the lm_head. Both
/// collapse cleanly to `(vocab,)` after squeezing the leading dims.
fn forward(
    backend: &mut Backend,
    input: &Tensor,
    pos_offset: usize,
    llama_cache: Option<&mut LlamaCache>,
) -> Result<Tensor, InferenceError> {
    match backend {
        Backend::Gemma3 { model, .. } => {
            let logits = model
                .forward(input, pos_offset)
                .map_err(|e| InferenceError::GenerationFailed(format!("gemma3 forward: {e}")))?;
            // Shape: (1, 1, vocab) -> (vocab,).
            logits
                .squeeze(0)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| InferenceError::GenerationFailed(format!("gemma3 squeeze: {e}")))
        }
        Backend::Llama { model, .. } => {
            let cache = llama_cache.ok_or_else(|| {
                InferenceError::GenerationFailed("llama backend missing cache".into())
            })?;
            let logits = model
                .forward(input, pos_offset, cache)
                .map_err(|e| InferenceError::GenerationFailed(format!("llama forward: {e}")))?;
            // Shape: (1, vocab) -> (vocab,).
            logits
                .squeeze(0)
                .map_err(|e| InferenceError::GenerationFailed(format!("llama squeeze: {e}")))
        }
    }
}

fn eos_token_id(backend: &Backend) -> Option<u32> {
    match backend {
        Backend::Gemma3 { eos_token_id, .. } => *eos_token_id,
        Backend::Llama { eos_token_id, .. } => *eos_token_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_file(dir: &Path, name: &str, body: &str) {
        fs::write(dir.join(name), body).expect("write test file");
    }

    #[test]
    fn detect_arch_reads_model_type() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "config.json", r#"{"model_type":"llama"}"#);
        let a = detect_architecture(&tmp.path().join("config.json")).unwrap();
        assert_eq!(a, Architecture::Llama);
    }

    #[test]
    fn detect_arch_falls_back_to_architectures_array() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "config.json",
            r#"{"architectures":["LlamaForCausalLM"]}"#,
        );
        let a = detect_architecture(&tmp.path().join("config.json")).unwrap();
        assert_eq!(a, Architecture::Llama);
    }

    #[test]
    fn detect_arch_handles_gemma3() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "config.json", r#"{"model_type":"gemma3"}"#);
        let a = detect_architecture(&tmp.path().join("config.json")).unwrap();
        assert_eq!(a, Architecture::Gemma3);
    }

    #[test]
    fn detect_arch_rejects_unknown() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "config.json", r#"{"model_type":"mistral"}"#);
        let err = detect_architecture(&tmp.path().join("config.json")).unwrap_err();
        match err {
            InferenceError::ArchUnsupported(a) => assert_eq!(a, "mistral"),
            other => panic!("expected ArchUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn detect_arch_rejects_missing_config() {
        let tmp = TempDir::new().unwrap();
        let err = detect_architecture(&tmp.path().join("config.json")).unwrap_err();
        match err {
            InferenceError::LoadFailed { .. } => {}
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }

    #[test]
    fn detect_arch_rejects_config_without_hints() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "config.json", r#"{"hidden_size":1}"#);
        let err = detect_architecture(&tmp.path().join("config.json")).unwrap_err();
        match err {
            InferenceError::LoadFailed { reason, .. } => {
                assert!(
                    reason.contains("model_type") || reason.contains("architectures"),
                    "reason should mention the missing fields: {reason}"
                );
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }

    #[test]
    fn new_errors_on_nonexistent_dir() {
        let err = CandleInference::new("/tmp/definitely-no-such-dir-xyz", DevicePreference::Cpu)
            .unwrap_err();
        match err {
            InferenceError::ModelMissing { .. } => {}
            other => panic!("expected ModelMissing, got {other:?}"),
        }
    }

    #[test]
    fn new_errors_on_missing_required_file() {
        let tmp = TempDir::new().unwrap();
        // Only create config.json — tokenizer.json and model.safetensors are missing.
        write_file(tmp.path(), "config.json", r#"{"model_type":"llama"}"#);
        let err = CandleInference::new(tmp.path(), DevicePreference::Cpu).unwrap_err();
        match err {
            InferenceError::ModelMissing { detail, .. } => {
                assert!(detail.contains("tokenizer.json") || detail.contains("model.safetensors"));
            }
            other => panic!("expected ModelMissing, got {other:?}"),
        }
    }

    #[test]
    fn new_errors_on_unsupported_arch_before_loading_weights() {
        // All required files present, but the arch is unsupported — we
        // should bail with ArchUnsupported rather than trying to mmap a
        // bogus safetensors file.
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "config.json", r#"{"model_type":"qwen2"}"#);
        write_file(tmp.path(), "tokenizer.json", "{}");
        write_file(tmp.path(), "model.safetensors", "");
        let err = CandleInference::new(tmp.path(), DevicePreference::Cpu).unwrap_err();
        match err {
            InferenceError::ArchUnsupported(a) => assert_eq!(a, "qwen2"),
            other => panic!("expected ArchUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn new_surfaces_load_failed_for_malformed_tokenizer() {
        // Config is well-formed and arch is supported, but tokenizer.json is
        // invalid JSON. The loader should surface LoadFailed pointing at the
        // bad file.
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "config.json", r#"{"model_type":"llama"}"#);
        write_file(tmp.path(), "tokenizer.json", "this is not a tokenizer");
        write_file(tmp.path(), "model.safetensors", "");
        let err = CandleInference::new(tmp.path(), DevicePreference::Cpu).unwrap_err();
        match err {
            InferenceError::LoadFailed { file, .. } => {
                assert!(file.ends_with("tokenizer.json"));
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }

    #[test]
    fn extract_u32_handles_scalar_and_array() {
        use serde_json::json;
        assert_eq!(extract_u32(Some(json!(42))), Some(42));
        assert_eq!(extract_u32(Some(json!([99, 100]))), Some(99));
        assert_eq!(extract_u32(Some(json!("string"))), None);
        assert_eq!(extract_u32(None), None);
    }
}
