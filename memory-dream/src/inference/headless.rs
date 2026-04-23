//! Headless inference backend — spawns an external CLI to produce completions.
//!
//! Rationale: users who already have `claude -p`, `gemini -p`, or any
//! stdin-readable LLM CLI installed can route dream condensation through
//! it instead of pulling multi-GB weights for the local candle backend.
//! Keeps the dream feature valuable on machines without a GPU or with
//! restricted disk.
//!
//! # Safety contract
//!
//! **Memory content never reaches a shell.** The command template is
//! tokenized *once* via `shlex::split` at [`HeadlessInference::new`] time.
//! For every `generate` call we:
//!
//! 1. Clone the pre-tokenized argv.
//! 2. Replace every occurrence of the literal string `{prompt}` in each
//!    token with the actual prompt (via `String::replace` — `{prompt}` is
//!    a literal marker, *not* a format spec, so `{` / `}` inside the
//!    memory content is safe).
//! 3. Spawn via [`std::process::Command::new`]`(argv[0]).args(&argv[1..])`.
//!    No `/bin/sh`, no `cmd.exe`, no word-splitting, no glob expansion.
//!
//! This is the single hardest invariant in the module — the verification
//! suite ships a shell-injection probe that seeds a memory containing
//! `"; rm -rf /tmp/DANGER; echo "` and asserts the filesystem is untouched.
//!
//! # Timeouts
//!
//! Timeouts use polling (`try_wait`) with a 50ms tick. `wait_timeout` was
//! considered but adds an extra dep for a 15-line behavior that the stdlib
//! handles cleanly. On timeout we send SIGKILL (the stdlib's `Child::kill`)
//! — best-effort; if the child ignores it the polling loop exits the function
//! with a `Timeout` error regardless of child state so the dream pass
//! continues.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::{Inference, InferenceError};

/// Literal placeholder substituted with the prompt at call time.
///
/// `{prompt}` is chosen so `config set headless.command "claude -p '{prompt}'"`
/// reads naturally. The placeholder is substituted via `String::replace`, not
/// any format-spec mechanism, so prompts containing `{` or `}` pass through
/// unharmed.
pub const PROMPT_PLACEHOLDER: &str = "{prompt}";

/// External-CLI inference backend.
///
/// Holds the pre-tokenized argv so `generate` is a cheap clone + replace +
/// spawn rather than re-parsing the template every call.
#[derive(Debug, Clone)]
pub struct HeadlessInference {
    /// Shell-tokenized argv. `argv[0]` is the binary; the rest are arguments.
    /// One or more tokens may contain the literal `{prompt}` marker; we
    /// replace each hit at call time.
    argv: Vec<String>,
    /// Wall-clock bound on a single invocation. `None` = no timeout.
    timeout: Option<Duration>,
}

impl HeadlessInference {
    /// Build a headless backend from a shell-style command template.
    ///
    /// Tokenization errors (unbalanced quotes, invalid escape) surface as
    /// [`InferenceError::Io`] with an explanatory message — settings-load
    /// should catch these first, but this constructor also accepts templates
    /// passed via `--command` at the CLI so the safety net belongs here too.
    ///
    /// `timeout_ms = 0` disables the timeout. Anything else is treated as a
    /// wall-clock bound on the subprocess.
    pub fn new(template: &str, timeout_ms: u64) -> Result<Self, InferenceError> {
        let argv = shlex::split(template).ok_or_else(|| {
            InferenceError::Io(format!(
                "headless command template {template:?} could not be tokenized \
                 (unbalanced quotes or invalid escape)"
            ))
        })?;
        if argv.is_empty() {
            return Err(InferenceError::Io(format!(
                "headless command template {template:?} tokenizes to zero arguments"
            )));
        }
        let timeout = if timeout_ms == 0 {
            None
        } else {
            Some(Duration::from_millis(timeout_ms))
        };
        Ok(Self { argv, timeout })
    }

    /// Substitute `{prompt}` in every token. Exposed for tests; callers use
    /// [`Self::generate`] which does the full spawn.
    fn substitute(&self, prompt: &str) -> Vec<String> {
        self.argv
            .iter()
            .map(|t| t.replace(PROMPT_PLACEHOLDER, prompt))
            .collect()
    }
}

impl Inference for HeadlessInference {
    fn generate(&self, prompt: &str, _max_tokens: u32) -> Result<String, InferenceError> {
        let argv = self.substitute(prompt);
        // argv is guaranteed non-empty by `new` — no `expect` needed on
        // `argv[0]`, but being defensive costs nothing.
        let program = argv
            .first()
            .ok_or_else(|| InferenceError::Io("empty argv after substitution".into()))?
            .clone();
        let args: Vec<&str> = argv.iter().skip(1).map(String::as_str).collect();

        let mut child = Command::new(&program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => InferenceError::Io(format!(
                    "headless command '{program}' not found on PATH: {e}"
                )),
                _ => InferenceError::Io(format!("spawn {program}: {e}")),
            })?;

        let (status, stdout, stderr_tail) = if let Some(timeout) = self.timeout {
            wait_with_timeout(&mut child, timeout)?
        } else {
            wait_blocking(&mut child)?
        };

        if !status.success() {
            let code = status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string());
            return Err(InferenceError::Io(format!(
                "headless command '{program}' exited non-zero (code={code}): {stderr_tail}"
            )));
        }

        Ok(stdout)
    }
}

/// Blocking wait — collect stdout/stderr and the exit status when no
/// timeout is configured. Factored out so the timeout path can reuse the
/// same stdout/stderr collection logic.
fn wait_blocking(
    child: &mut std::process::Child,
) -> Result<(std::process::ExitStatus, String, String), InferenceError> {
    // `wait_with_output` consumes the child; we can't re-use it. The
    // equivalent for a `&mut Child` is to take the pipes, drain them, then
    // `wait()`. Order matters: drain stdout/stderr *before* `wait` so a
    // child writing more than the pipe-buffer worth of output doesn't
    // deadlock on its own stdout.
    let mut out = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_string(&mut out)
            .map_err(|e| InferenceError::Io(format!("read stdout: {e}")))?;
    }
    let mut err = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_string(&mut err)
            .map_err(|e| InferenceError::Io(format!("read stderr: {e}")))?;
    }
    let status = child
        .wait()
        .map_err(|e| InferenceError::Io(format!("wait: {e}")))?;
    Ok((status, out, tail(&err, 256)))
}

/// Poll for completion up to `timeout`, killing the child on deadline.
///
/// Poll interval is 50ms — fast enough that a snappy CLI (Claude, echo)
/// returns within a tick or two, slow enough that 100s of dream rows don't
/// saturate a core on the polling itself.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, String, String), InferenceError> {
    let deadline = Instant::now() + timeout;
    let poll_interval = Duration::from_millis(50);

    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                // Child exited — fall through to the blocking drain, which
                // now returns immediately since `try_wait` confirmed exit.
                return wait_blocking(child);
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    // Best-effort kill. Ignore the result — if the child is
                    // already reaped via a race with try_wait, or if kill
                    // fails because it's a zombie, we still surface the
                    // timeout to the caller and let the dream pass continue.
                    let _ = child.kill();
                    // Reap so the process table doesn't accumulate zombies
                    // across a long dream pass. Ignore errors for the
                    // already-exited case.
                    let _ = child.wait();
                    return Err(InferenceError::Io(format!(
                        "headless command timed out after {}ms",
                        timeout.as_millis()
                    )));
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => return Err(InferenceError::Io(format!("try_wait: {e}"))),
        }
    }
}

/// Keep the last `max` bytes of a string, UTF-8-safe. Used to cap stderr
/// noise in error messages — full stderr can be megabytes on a runaway CLI.
fn tail(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let start = s.len() - max;
        // Walk forward to the nearest char boundary so we don't slice a
        // UTF-8 sequence in half.
        let boundary = (start..s.len()).find(|i| s.is_char_boundary(*i)).unwrap_or(start);
        format!("…{}", &s[boundary..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_backend(template: &str) -> HeadlessInference {
        HeadlessInference::new(template, 10_000).expect("construct")
    }

    #[test]
    fn new_rejects_empty_template() {
        let err = HeadlessInference::new("", 0).unwrap_err();
        assert!(matches!(err, InferenceError::Io(_)));
    }

    #[test]
    fn new_rejects_unbalanced_quotes() {
        let err = HeadlessInference::new("claude -p 'unbalanced", 0).unwrap_err();
        match err {
            InferenceError::Io(msg) => assert!(msg.contains("tokenized")),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn substitute_replaces_every_placeholder() {
        let h = echo_backend("echo prefix {prompt} suffix {prompt}");
        let argv = h.substitute("HELLO");
        assert_eq!(
            argv,
            vec!["echo", "prefix", "HELLO", "suffix", "HELLO"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn substitute_passes_through_when_no_placeholder() {
        let h = echo_backend("echo fixed argument");
        let argv = h.substitute("ignored");
        assert_eq!(
            argv,
            vec!["echo", "fixed", "argument"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn happy_path_echo_returns_stdout() {
        // `/bin/echo` is universally available on macOS/Linux CI.
        let h = echo_backend("echo {prompt}");
        let out = h.generate("hello world", 32).expect("generate ok");
        assert_eq!(out.trim(), "hello world");
    }

    #[test]
    fn command_not_found_surfaces_io_error() {
        let h = echo_backend("definitely-not-a-real-binary-xyz {prompt}");
        let err = h.generate("x", 32).unwrap_err();
        match err {
            InferenceError::Io(msg) => assert!(
                msg.contains("not found") || msg.contains("spawn"),
                "expected NotFound in message, got: {msg}"
            ),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn non_zero_exit_surfaces_io_error_with_stderr() {
        // `/bin/false` exits 1 with no output. Fits perfectly for the
        // non-zero-exit path without requiring a shell.
        let h = echo_backend("false");
        let err = h.generate("x", 32).unwrap_err();
        match err {
            InferenceError::Io(msg) => assert!(msg.contains("exited non-zero")),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn timeout_kills_long_running_command() {
        // Sleep 10s; timeout at 200ms. Must return Timeout well before 10s.
        let h = HeadlessInference::new("sleep 10", 200).unwrap();
        let start = Instant::now();
        let err = h.generate("x", 32).unwrap_err();
        let elapsed = start.elapsed();
        match err {
            InferenceError::Io(msg) => assert!(msg.contains("timed out")),
            other => panic!("expected Io(timed out), got {other:?}"),
        }
        // Upper bound: well below the 10s sleep. 3s is generous for a CI
        // scheduler + cleanup races but still proves the kill works.
        assert!(
            elapsed < Duration::from_secs(3),
            "expected timeout <3s, took {elapsed:?}"
        );
    }

    /// Shell-injection probe — the core safety test. If the spawn path
    /// ever routes through a shell, this memory content would create
    /// `/tmp/agent_memory_headless_DANGER_<pid>` as a side-effect. The
    /// argv-based spawn must echo the string verbatim and touch nothing.
    ///
    /// Uses a PID-suffixed path so parallel test runs don't step on each
    /// other's sentinel files.
    #[test]
    fn shell_injection_probe_never_creates_sentinel_file() {
        let pid = std::process::id();
        let sentinel = std::env::temp_dir().join(format!("agent_memory_headless_DANGER_{pid}"));
        // Clean up any stale sentinel from a prior failed run.
        let _ = std::fs::remove_file(&sentinel);

        let malicious = format!("\"; touch {} ; echo \"", sentinel.display());
        let h = echo_backend("echo {prompt}");
        let out = h.generate(&malicious, 32).expect("echo must succeed");

        // Output must contain the literal payload — proof that no shell
        // expansion stripped the metacharacters.
        assert!(
            out.contains(&malicious),
            "expected literal payload in stdout; got: {out}"
        );
        // Hard invariant: the sentinel file must NOT exist.
        assert!(
            !sentinel.exists(),
            "shell injection succeeded: {sentinel:?} was created"
        );
    }

    #[test]
    fn tail_returns_string_unchanged_when_short() {
        assert_eq!(tail("short", 100), "short");
    }

    #[test]
    fn tail_truncates_to_char_boundary() {
        let long: String = "x".repeat(500);
        let t = tail(&long, 10);
        // Leading '…' plus 10 chars of 'x'.
        assert!(t.starts_with('…'));
        assert_eq!(t.chars().filter(|c| *c == 'x').count(), 10);
    }
}
