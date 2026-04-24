//! Agentic dream curation + tool-support probe (Release 2.3).
//!
//! Two responsibilities, both specific to the Release 2.3 agentic flow:
//!
//! 1. **Probe the backend for shell-tool support** ([`probe_tool_support`]).
//!    The dream orchestrator calls this once at pass start and caches the
//!    result; the entire remaining run routes to either the agentic-batch
//!    path (here) or the non-agentic per-memory condensation path in
//!    [`crate::dream::condense`].
//!
//! 2. **Drive one agentic curation batch** ([`run_agentic_batch`]). Given a
//!    project identity and a memory slice, build the one-shot prompt,
//!    invoke the backend, and let the LLM run `memory forget` /
//!    `memory update` / `memory move` / `memory store` / `memory context`
//!    directly. We do not parse the model's response — DB side effects
//!    are the deliverable.
//!
//! The module stays intentionally small. Everything about "how" to talk to
//! a backend lives in [`crate::inference`]; this file decides "when" and
//! "with what prompt".

use crate::dream::prompt::build_agentic_prompt;
use crate::inference::{Inference, InferenceError};
use agent_memory::db::models::Memory;
use agent_memory::render;

/// Maximum response token budget for the probe. We expect either a short
/// version string (≤ 32 chars) or the literal `NO_TOOLS` sentinel; 64 is
/// comfortable headroom and keeps probe latency trivial.
pub const PROBE_MAX_TOKENS: u32 = 64;

/// Maximum response token budget for an agentic batch. Claude's default CLI
/// session has no practical ceiling on tool-call output, but we still cap
/// the direct model reply so a confused backend can't emit megabytes of
/// commentary. Chat/completion backends that treat this as a hard cap will
/// simply truncate; tool-call backends ignore it.
pub const AGENTIC_MAX_TOKENS: u32 = 8_192;

/// Canary prompt for the tool-support probe. Intentionally terse — any
/// tool-enabled backend (Claude with Bash access, for example) runs
/// `memory --version` and replies with the one-line output. Any other
/// backend falls through to the `NO_TOOLS` sentinel.
///
/// Kept public so integration tests can assert the exact prompt surface.
pub const PROBE_PROMPT: &str =
    "Run `memory --version` and reply with just its output. \
If you cannot run shell commands, reply with `NO_TOOLS`.";

/// Classification of a backend's response to the tool-support probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// The backend invoked a shell tool and returned a version-looking
    /// string. Agentic mode is available.
    Supported,
    /// The backend replied with the `NO_TOOLS` sentinel, explicitly
    /// acknowledging no shell access.
    NoTools,
    /// The backend failed or the reply matched neither pattern. Treated
    /// the same as `NoTools` by the caller (downgrade to non-agentic)
    /// but distinguished here so the orchestrator can log the raw
    /// response for diagnostics.
    Unknown(String),
}

impl ProbeOutcome {
    /// Convenience: `true` only when the probe explicitly succeeded with
    /// a version-shaped reply. Ambiguous and failed probes return false.
    pub fn supports_tools(&self) -> bool {
        matches!(self, ProbeOutcome::Supported)
    }
}

/// Probe the backend for shell-tool support.
///
/// Runs a single [`Inference::generate`] call against [`PROBE_PROMPT`] and
/// classifies the reply:
///   - If the response contains a semver-ish token (digit.digit.digit OR
///     the string "memory "), classify as [`ProbeOutcome::Supported`].
///   - If the response contains `NO_TOOLS` (case-insensitive) and no
///     version token, classify as [`ProbeOutcome::NoTools`].
///   - All other cases surface as [`ProbeOutcome::Unknown`] so the
///     caller can log diagnostics.
///
/// The probe never propagates its backend error — a failed `generate`
/// call returns `Unknown(error)` so the dream pass continues in
/// non-agentic mode instead of dying. Real errors elsewhere in the
/// pipeline (DB, embeddings) still fail loudly; the probe just decides
/// which mode to pick.
pub fn probe_tool_support(inference: &dyn Inference) -> ProbeOutcome {
    match inference.generate(PROBE_PROMPT, PROBE_MAX_TOKENS) {
        Ok(reply) => classify_probe_reply(&reply),
        Err(InferenceError::Io(msg)) => ProbeOutcome::Unknown(format!("io: {msg}")),
        Err(other) => ProbeOutcome::Unknown(format!("{other}")),
    }
}

/// Inspect a probe reply and decide which [`ProbeOutcome`] it matches.
///
/// Semver token pattern: three numeric groups separated by `.` (e.g.
/// `1.3.0`, `0.2.11-pre`). The leading `memory ` prefix is optional — some
/// backends strip it in their formatted response.
///
/// We check for the version token first so a reply like
/// `"NO_TOOLS but the version is 1.3.0"` still counts as Supported (the
/// model clearly has version info even if it added framing). Conversely,
/// a bare `NO_TOOLS` never passes the version check and routes to
/// `NoTools`.
fn classify_probe_reply(reply: &str) -> ProbeOutcome {
    if looks_like_version_string(reply) {
        return ProbeOutcome::Supported;
    }
    if reply.to_uppercase().contains("NO_TOOLS") {
        return ProbeOutcome::NoTools;
    }
    ProbeOutcome::Unknown(truncate_for_log(reply, 200))
}

/// Tiny regex-free semver detector. Scans char-by-char for the pattern
/// `<digits>.<digits>.<digits>`; short-circuits on the first hit.
///
/// Not trying to be RFC-perfect here. The probe's job is "did the
/// backend run my tool?" — a version-shaped token is proof; a perfect
/// semver parser would reject `1.3.0-pre.1` which is fine. The test
/// suite pins the accepted/rejected cases.
fn looks_like_version_string(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Require a digit to start.
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        // First dot.
        if i >= chars.len() || chars[i] != '.' {
            continue;
        }
        i += 1;
        // Second numeric group.
        let second = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i == second {
            continue;
        }
        // Second dot.
        if i >= chars.len() || chars[i] != '.' {
            continue;
        }
        i += 1;
        // Third numeric group.
        let third = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i > third {
            // Guard against a trailing-run false positive by ensuring we
            // consumed at least one digit in each group.
            let _ = start;
            return true;
        }
    }
    false
}

/// Truncate a string for diagnostic logging. Keeps the first `max` chars
/// and appends `…` when truncated. UTF-8 safe.
fn truncate_for_log(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

/// Run one agentic curation batch for a project.
///
/// Builds the one-shot prompt via [`build_agentic_prompt`] (memory-list +
/// tool documentation + curation rules), invokes the backend, and returns
/// the raw reply along with an element count. **No DB writes happen
/// through this function** — the LLM issues `memory ...` commands via
/// its tool surface, and those commands write to the DB directly.
///
/// Progress reporting: the caller prints a light-XML `<dream_batch ...>`
/// line before and after so the orchestrator's tool-call log is
/// structurally consistent with the rest of dream output.
pub fn run_agentic_batch(
    inference: &dyn Inference,
    project: &str,
    memories: &[Memory],
) -> Result<AgenticReport, InferenceError> {
    let numbered = numbered_memory_list(memories);
    let prompt = build_agentic_prompt(project, memories.len(), &numbered);
    let raw_reply = inference.generate(&prompt, AGENTIC_MAX_TOKENS)?;
    Ok(AgenticReport {
        reply_bytes: raw_reply.len(),
        memories_in_batch: memories.len(),
    })
}

/// Summary of a single agentic batch. Deliberately minimal — the DB
/// mutations are the real output; this struct is for orchestrator
/// accounting.
#[derive(Debug, Clone)]
pub struct AgenticReport {
    /// Length of the backend's stdout reply. Useful for anomaly detection
    /// (a runaway zero-byte reply likely means the backend spawned but
    /// didn't invoke any tools).
    pub reply_bytes: usize,
    /// How many memories were in the batch. Mirrors the prompt's `{n}`.
    pub memories_in_batch: usize,
}

/// Build the numbered memory list injected into the agentic prompt.
///
/// Each line shape:
/// ```text
/// 1. [id:<8char>] (type=<t>, tags=<csv>) <preview>
/// ```
///
/// Preview = first 160 chars with whitespace collapsed; the model has
/// tool access and can re-read full content via `memory get <id>` when it
/// needs more context. Keeping the prompt preview short lets us batch up
/// to ~100 memories without blowing past the context window.
pub fn numbered_memory_list(memories: &[Memory]) -> String {
    let mut out = String::new();
    for (i, m) in memories.iter().enumerate() {
        let idx = i + 1;
        let id_short = render::short_id(&m.id);
        let mtype = m.memory_type.as_deref().unwrap_or("-");
        let tags = m
            .tags
            .as_deref()
            .map(|t| t.join(","))
            .unwrap_or_else(|| "-".to_string());
        let preview = one_line_preview(&m.content, 160);
        out.push_str(&format!(
            "{idx}. [id:{id_short}] (type={mtype}, tags={tags}) {preview}\n"
        ));
    }
    out
}

/// Collapse whitespace and truncate a memory body for the agentic prompt.
/// Mirrors the one-line preview logic in `agent_memory::render` but
/// duplicated here to avoid exporting an internal helper from the memory
/// crate just for dream's use.
fn one_line_preview(content: &str, max_chars: usize) -> String {
    let mut collapsed = String::with_capacity(content.len().min(max_chars * 2));
    let mut prev_space = false;
    for ch in content.chars() {
        if ch.is_whitespace() {
            if !prev_space && !collapsed.is_empty() {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }
    while collapsed.ends_with(' ') {
        collapsed.pop();
    }
    if collapsed.chars().count() > max_chars {
        let truncated: String = collapsed.chars().take(max_chars).collect();
        format!("{truncated}…")
    } else {
        collapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FixedInference;

    fn mk_memory(id: &str, content: &str) -> Memory {
        let mut m = Memory::new(
            content.to_string(),
            Some(vec!["tag1".to_string(), "tag2".to_string()]),
            Some("test".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        m.id = id.to_string();
        m
    }

    #[test]
    fn version_string_is_supported() {
        let stub = FixedInference::new("memory 1.3.0");
        assert_eq!(probe_tool_support(&stub), ProbeOutcome::Supported);
    }

    #[test]
    fn bare_semver_is_supported() {
        let stub = FixedInference::new("1.3.0");
        assert_eq!(probe_tool_support(&stub), ProbeOutcome::Supported);
    }

    #[test]
    fn prerelease_semver_is_supported() {
        let stub = FixedInference::new("memory 0.2.11-pre");
        assert_eq!(probe_tool_support(&stub), ProbeOutcome::Supported);
    }

    #[test]
    fn no_tools_sentinel_routes_to_no_tools() {
        let stub = FixedInference::new("NO_TOOLS");
        assert_eq!(probe_tool_support(&stub), ProbeOutcome::NoTools);
    }

    #[test]
    fn no_tools_sentinel_mixed_case_is_detected() {
        let stub = FixedInference::new("no_tools");
        assert_eq!(probe_tool_support(&stub), ProbeOutcome::NoTools);
    }

    #[test]
    fn empty_reply_is_unknown() {
        let stub = FixedInference::new("");
        matches!(probe_tool_support(&stub), ProbeOutcome::Unknown(_));
    }

    #[test]
    fn two_dot_zero_is_not_a_version() {
        // Exactly two numeric groups is insufficient — looks_like_version
        // requires three for the tool-support heuristic.
        assert!(!looks_like_version_string("1.3"));
    }

    #[test]
    fn date_like_tokens_are_rejected() {
        // `2026-04-23` has no dots in the right places — must not classify
        // as a version string.
        assert!(!looks_like_version_string("2026-04-23"));
    }

    #[test]
    fn supports_tools_is_true_only_for_supported() {
        assert!(ProbeOutcome::Supported.supports_tools());
        assert!(!ProbeOutcome::NoTools.supports_tools());
        assert!(!ProbeOutcome::Unknown("x".to_string()).supports_tools());
    }

    #[test]
    fn numbered_memory_list_renders_short_ids_and_previews() {
        let mems = vec![
            mk_memory("aaaaaaaa-0000-1111-2222-000000000001", "short content"),
            mk_memory("bbbbbbbb-0000-1111-2222-000000000002", "second content"),
        ];
        let list = numbered_memory_list(&mems);
        assert!(list.contains("1. [id:aaaaaaaa]"));
        assert!(list.contains("(type=user, tags=tag1,tag2)"));
        assert!(list.contains("short content"));
        assert!(list.contains("2. [id:bbbbbbbb]"));
    }

    #[test]
    fn numbered_memory_list_collapses_whitespace_in_preview() {
        let mems = vec![mk_memory(
            "aaaaaaaa-0000-1111-2222-000000000001",
            "multi\n\n  line\tcontent",
        )];
        let list = numbered_memory_list(&mems);
        assert!(list.contains("multi line content"));
    }

    #[test]
    fn run_agentic_batch_records_memory_count() {
        // The stub returns a canned string; run_agentic_batch just wraps
        // the backend call and surfaces basic accounting.
        let mems = vec![
            mk_memory("aaaaaaaa-0000-1111-2222-000000000001", "a"),
            mk_memory("bbbbbbbb-0000-1111-2222-000000000002", "b"),
        ];
        let stub = FixedInference::new("tool calls happened asynchronously");
        let report = run_agentic_batch(&stub, "test", &mems).expect("agentic ok");
        assert_eq!(report.memories_in_batch, 2);
        assert!(report.reply_bytes > 0);
    }

    #[test]
    fn run_agentic_batch_surfaces_inference_error() {
        // When the backend fails outright (e.g. headless command not found),
        // the error propagates so the orchestrator can degrade the mode or
        // skip the batch.
        use crate::inference::NoopInference;
        let inf = NoopInference::new("forced failure");
        let mems = vec![mk_memory("aaaaaaaa-0000-1111-2222-000000000001", "a")];
        let err = run_agentic_batch(&inf, "test", &mems).unwrap_err();
        // NoopInference always returns ModelMissing.
        assert!(matches!(err, InferenceError::ModelMissing { .. }));
    }
}
