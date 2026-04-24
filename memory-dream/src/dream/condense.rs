//! Per-memory condensation with a strict three-way response contract.
//!
//! Replaces the Release 2.3 plain-bullets parser. The v1.4.4 prompt
//! ([`crate::dream::prompt::CONDENSE_PROMPT_TEMPLATE`]) instructs the
//! model to respond with EXACTLY ONE of:
//!
//!   1. The single word `skip` — no change needed.
//!   2. The single word `forget` — noise; delete the memory.
//!   3. A rewritten condensed body — headline + bullets, strictly
//!      shorter than the input.
//!
//! This module parses that response and returns a [`Decision`] that the
//! orchestrator maps onto DB side effects. Deliberately small — all the
//! "what to do next" logic lives in [`crate::dream::mod`].
//!
//! # Parser semantics
//!
//! * The raw response is trimmed of leading/trailing whitespace before
//!   any matching. Empty → [`CondenseError::ParseFailed`].
//! * Case-insensitive literal match for `skip` / `forget` on the first
//!   line, but any trailing content after the literal marker is rejected
//!   as malformed (the model broke the "exactly one of these forms"
//!   contract and we refuse to guess intent).
//! * Anything else is treated as a rewritten body. It must be strictly
//!   shorter than the input and must not match a refusal marker.

use thiserror::Error;

use agent_memory::db::models::Memory;

use crate::dream::prompt::{build_condense_prompt, CondensePromptInputs, MAX_OUTPUT_TOKENS};
use crate::inference::{Inference, InferenceError};

/// Sentinel project ident that flags a memory as user-wide / global.
/// Mirrors the value used by the `memory` crate's CLI surface.
const GLOBAL_PROJECT_IDENT: &str = "__global__";

/// Parsed per-memory decision from a model response.
///
/// `Rewrite` carries the validated body so the orchestrator can persist
/// it directly without re-parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Skip,
    Forget,
    Rewrite { text: String },
}

/// Everything that can go wrong in the condensation pipeline.
#[derive(Debug, Error)]
pub enum CondenseError {
    #[error("inference backend failed: {0}")]
    InferenceFailed(#[from] InferenceError),

    #[error("failed to parse model response: {0}")]
    ParseFailed(String),

    #[error(
        "condensed text was not shorter than input (raw={raw_len}, condensed={condensed_len})"
    )]
    TooLong {
        raw_len: usize,
        condensed_len: usize,
    },

    #[error("model refused the task: {0}")]
    Refused(String),
}

/// Refusal patterns matched case-insensitively on the raw model output
/// before the parse stage. False positives cost a condensation; false
/// negatives would let a refusal sneak into the `content` column, so
/// the bias is toward aggressive matching.
const REFUSAL_MARKERS: &[&str] = &[
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i'm sorry, but",
    "i am sorry, but",
    "as an ai",
    "as a language model",
];

/// Strip optional markdown code fences some models wrap their output in.
/// The prompt forbids them but defense-in-depth is cheap.
fn strip_code_fence(s: &str) -> &str {
    let trimmed = s.trim();
    if let Some(body) = trimmed
        .strip_prefix("```text")
        .or_else(|| trimmed.strip_prefix("```markdown"))
        .or_else(|| trimmed.strip_prefix("```"))
    {
        body.trim_end_matches("```").trim()
    } else {
        trimmed
    }
}

/// Check for refusal markers in the raw response before the parse stage.
fn detect_refusal(raw: &str) -> Option<&'static str> {
    let lower = raw.to_lowercase();
    for marker in REFUSAL_MARKERS {
        if lower.contains(marker) {
            return Some(*marker);
        }
    }
    None
}

/// Parse a model response against the three-way contract.
///
/// * `raw_response` is the model's stdout, pre-trim.
/// * `raw_input` is the memory's current content (used to enforce the
///   strictly-shorter invariant on rewrites).
///
/// Returns a [`Decision`] on success.
pub fn parse_response(raw_response: &str, raw_input: &str) -> Result<Decision, CondenseError> {
    let body = strip_code_fence(raw_response).trim();
    if body.is_empty() {
        return Err(CondenseError::ParseFailed(
            "empty response from model".to_string(),
        ));
    }

    // Literal-word short-circuits. Strict — a `skip` or `forget` reply
    // followed by explanatory text violates the "exactly one form"
    // contract. Accepting it would let the model silently substitute its
    // own interpretation for our parser's.
    let lower = body.to_lowercase();
    if lower == "skip" {
        return Ok(Decision::Skip);
    }
    if lower == "forget" {
        return Ok(Decision::Forget);
    }

    // Detect the "keyword + extra text" shape explicitly so the error
    // message is actionable (vs. a generic "too long").
    if let Some(rest) = lower.strip_prefix("skip") {
        if rest.starts_with(|c: char| c.is_whitespace()) {
            return Err(CondenseError::ParseFailed(
                "response starts with `skip` but contains extra text".to_string(),
            ));
        }
    }
    if let Some(rest) = lower.strip_prefix("forget") {
        if rest.starts_with(|c: char| c.is_whitespace()) {
            return Err(CondenseError::ParseFailed(
                "response starts with `forget` but contains extra text".to_string(),
            ));
        }
    }

    // Refusal detection runs before the length gate so a "I cannot" reply
    // that happens to be shorter than the input doesn't sneak in.
    if let Some(marker) = detect_refusal(body) {
        return Err(CondenseError::Refused(format!(
            "response contains refusal marker '{marker}'"
        )));
    }

    // Strictly-shorter invariant. The prompt is explicit about this; the
    // orchestrator enforces it so a stubborn model can't grow the corpus.
    let condensed_len = body.chars().count();
    let raw_len = raw_input.trim().chars().count();
    if condensed_len >= raw_len {
        return Err(CondenseError::TooLong {
            raw_len,
            condensed_len,
        });
    }

    Ok(Decision::Rewrite {
        text: body.to_string(),
    })
}

/// Run a full per-memory condensation pass.
///
/// Builds the prompt from the memory's metadata + content, invokes the
/// backend once, and parses the response. The caller ([`crate::dream::mod`])
/// maps the resulting [`Decision`] onto DB side effects.
pub fn run_per_memory(
    inference: &dyn Inference,
    source: &Memory,
) -> Result<Decision, CondenseError> {
    // `project` already stores the literal `__global__` sentinel for
    // global-scope memories, so no translation is needed before passing
    // it through as the prompt's scope field. We reference the constant
    // explicitly so the intent is greppable from here.
    let _ = GLOBAL_PROJECT_IDENT;
    let tags_joined = source.tags.as_ref().map(|ts| ts.join(","));
    let inputs = CondensePromptInputs {
        memory_type: source.memory_type.as_deref(),
        project_or_global: source.project.as_deref(),
        tags: tags_joined.as_deref(),
        content: &source.content,
    };
    let prompt = build_condense_prompt(&inputs);
    let response = inference.generate(&prompt, MAX_OUTPUT_TOKENS)?;
    parse_response(&response, &source.content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FixedInference;

    fn mk_memory(content: &str) -> Memory {
        Memory::new(
            content.to_string(),
            Some(vec!["tag1".to_string()]),
            Some("test".to_string()),
            None,
            None,
            Some("user".to_string()),
        )
    }

    #[test]
    fn literal_skip_returns_skip_decision() {
        let out = parse_response("skip", "some input").unwrap();
        assert_eq!(out, Decision::Skip);
    }

    #[test]
    fn literal_skip_is_case_insensitive() {
        let out = parse_response("SKIP", "some input").unwrap();
        assert_eq!(out, Decision::Skip);
    }

    #[test]
    fn skip_with_trailing_newline_is_accepted() {
        // `claude -p` appends a newline by default; the pre-trim in the
        // parser must strip it so `skip\n` reads as the literal word.
        let out = parse_response("skip\n", "some input").unwrap();
        assert_eq!(out, Decision::Skip);
    }

    #[test]
    fn skip_then_explanation_is_rejected_as_malformed() {
        let err = parse_response("skip because the memory is fine", "some input").unwrap_err();
        match err {
            CondenseError::ParseFailed(msg) => assert!(msg.contains("extra text")),
            other => panic!("expected ParseFailed, got {other:?}"),
        }
    }

    #[test]
    fn literal_forget_returns_forget_decision() {
        let out = parse_response("forget", "long noisy input").unwrap();
        assert_eq!(out, Decision::Forget);
    }

    #[test]
    fn forget_case_insensitive() {
        let out = parse_response("Forget\n", "long noisy input").unwrap();
        assert_eq!(out, Decision::Forget);
    }

    #[test]
    fn forget_then_explanation_is_rejected_as_malformed() {
        let err = parse_response("forget - CI noise", "some long input").unwrap_err();
        match err {
            CondenseError::ParseFailed(msg) => assert!(msg.contains("extra text")),
            other => panic!("expected ParseFailed, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_shorter_than_input_returns_rewrite() {
        let raw =
            "On 2026-04-23 the user decided to keep the legacy index for one more release cycle.";
        let resp = "2026-04-23: keep legacy index\n- Scope: follow-up release";
        assert!(
            resp.chars().count() < raw.chars().count(),
            "test precondition: rewrite must be shorter than raw input"
        );
        let out = parse_response(resp, raw).unwrap();
        match out {
            Decision::Rewrite { text } => {
                assert!(text.starts_with("2026-04-23"));
                assert!(text.contains("- Scope"));
            }
            other => panic!("expected Rewrite, got {other:?}"),
        }
    }

    #[test]
    fn too_long_rewrite_is_rejected() {
        let raw = "short";
        let resp = "this is a much longer response than the raw input";
        let err = parse_response(resp, raw).unwrap_err();
        match err {
            CondenseError::TooLong {
                raw_len,
                condensed_len,
            } => {
                assert_eq!(raw_len, raw.chars().count());
                assert!(condensed_len > raw_len);
            }
            other => panic!("expected TooLong, got {other:?}"),
        }
    }

    #[test]
    fn empty_response_is_parse_failed() {
        let err = parse_response("   \n\n  ", "x").unwrap_err();
        match err {
            CondenseError::ParseFailed(msg) => assert!(msg.contains("empty")),
            other => panic!("expected ParseFailed, got {other:?}"),
        }
    }

    #[test]
    fn refusal_marker_is_detected_before_length_check() {
        let raw = "Some verbose user story we want condensed into a single bullet line.";
        // Short enough to pass the length gate if we didn't check refusals first.
        let resp = "I cannot help.";
        let err = parse_response(resp, raw).unwrap_err();
        assert!(matches!(err, CondenseError::Refused(_)));
    }

    #[test]
    fn code_fence_wrapped_rewrite_is_unwrapped() {
        let raw = "long input with several words so the fenced reply is shorter";
        let resp = "```\nshort rewrite\n- fact\n```";
        let out = parse_response(resp, raw).unwrap();
        match out {
            Decision::Rewrite { text } => assert_eq!(text, "short rewrite\n- fact"),
            other => panic!("expected Rewrite, got {other:?}"),
        }
    }

    #[test]
    fn run_per_memory_wires_inputs_through_prompt() {
        // FixedInference ignores the prompt and returns "skip" — we just
        // verify the outer plumbing compiles and runs.
        let mem = mk_memory("hello world");
        let inf = FixedInference::new("skip");
        let d = run_per_memory(&inf, &mem).unwrap();
        assert_eq!(d, Decision::Skip);
    }

    #[test]
    fn run_per_memory_uses_global_ident_in_prompt_scope() {
        // When the memory's project is `__global__`, the prompt-scope
        // field must reflect that verbatim. A FixedInference echo would
        // let us inspect the prompt; since FixedInference doesn't, we
        // instead rely on the scope_label mapping being exercised by
        // build_condense_prompt in prompt::tests. This test just
        // guards the wiring.
        let mut mem = mk_memory("x");
        mem.project = Some(GLOBAL_PROJECT_IDENT.to_string());
        let inf = FixedInference::new("skip");
        let d = run_per_memory(&inf, &mem).unwrap();
        assert_eq!(d, Decision::Skip);
    }
}
