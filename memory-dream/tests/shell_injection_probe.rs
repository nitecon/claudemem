//! End-to-end shell-injection probe for `memory-dream test --backend headless`.
//!
//! This test is the single hardest safety invariant Release 2.2 ships. If the
//! spawn path in `HeadlessInference` ever regresses — e.g. someone switches
//! from `Command::new(tokens[0]).args(...)` to `Command::new("sh").arg("-c")`
//! — a maliciously-crafted memory could execute arbitrary shell commands.
//! This test seeds such a memory, runs `memory-dream test` against it through
//! an `echo '{prompt}'` command, and asserts the filesystem is untouched.
//!
//! The probe uses:
//!   * A tempdir-scoped `$AGENT_MEMORY_DIR` so the real user DB is not
//!     touched by the store / test calls.
//!   * A PID-suffixed sentinel path so parallel test runs don't stomp
//!     each other's assertions.
//!   * The release binaries built by `cargo build --release` so we exercise
//!     the same artifact users run.
//!
//! A test against an in-process `HeadlessInference::generate` is already
//! covered by a unit test in `inference::headless::tests`. The value of the
//! integration test is that it exercises the *full* invocation path: CLI
//! argument parsing, settings loading, prompt construction, and subprocess
//! spawn — so a regression in any layer surfaces here.

use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

/// Locate the release binary for `bin` under `target/release/`. Workspace
/// tests run with CWD = the crate dir, so walk up one level to the workspace
/// root and then down into `target/release/`.
fn release_binary(bin: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .expect("workspace root")
        .to_path_buf();
    workspace_root.join("target").join("release").join(bin)
}

/// Spawn `memory store` to insert a memory with the given content. Returns
/// the 8-char short id echoed by the CLI on success. Parsing the short id
/// from the light-XML `<result status="stored" id="..."` attribute is the
/// agreed public contract; see `memory/src/render/mod.rs`.
fn store_memory(data_dir: &std::path::Path, memory: &str, bin: &std::path::Path) -> String {
    let out = Command::new(bin)
        .args(["store", memory, "-m", "user", "-t", "probe"])
        .env("AGENT_MEMORY_DIR", data_dir)
        .output()
        .expect("spawn memory store");
    assert!(
        out.status.success(),
        "memory store failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Pluck the `id="..."` attribute. Simple string slicing avoids taking
    // on a regex dep for a once-per-test parse.
    let marker = "id=\"";
    let start = stdout
        .find(marker)
        .unwrap_or_else(|| panic!("no id= in stdout: {stdout}"));
    let rest = &stdout[start + marker.len()..];
    let end = rest.find('"').unwrap();
    rest[..end].to_string()
}

/// Count memory rows via `memory list`. Used to assert the test subcommand
/// performs zero writes.
fn row_count(data_dir: &std::path::Path, bin: &std::path::Path) -> usize {
    let out = Command::new(bin)
        .args(["list", "-k", "100"])
        .env("AGENT_MEMORY_DIR", data_dir)
        .output()
        .expect("spawn memory list");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The rendered header is `<memories count="N">` — or `<memories count="0"/>`
    // when the DB is empty.
    let marker = "count=\"";
    let start = stdout
        .find(marker)
        .unwrap_or_else(|| panic!("no count= in stdout: {stdout}"));
    let rest = &stdout[start + marker.len()..];
    let end = rest.find('"').unwrap();
    rest[..end].parse().unwrap()
}

/// The core probe. Split out from `#[test]` so the setup can assert on the
/// sentinel path before returning — on a CI failure the `stdout` body is
/// critical for diagnosing *which* layer regressed.
fn run_probe() -> Result<(), String> {
    let memory_bin = release_binary("memory");
    let dream_bin = release_binary("memory-dream");
    if !memory_bin.exists() || !dream_bin.exists() {
        return Err(format!(
            "release binaries missing (run `cargo build --release` first): \
             memory={} exists={}, memory-dream={} exists={}",
            memory_bin.display(),
            memory_bin.exists(),
            dream_bin.display(),
            dream_bin.exists()
        ));
    }

    let data_dir = TempDir::new().expect("tempdir");

    // Sentinel path is in the tempdir itself (not /tmp) so a test failure
    // doesn't stash sensitive paths outside the scoped fixture, and we
    // guarantee a clean slate per test run regardless of PID collisions
    // under heavy parallelism.
    let sentinel = data_dir.path().join("DANGER_sentinel");
    assert!(!sentinel.exists(), "sentinel pre-exists: {sentinel:?}");

    // Seed a memory whose content is a textbook shell-injection payload.
    // If spawn went through a shell, the embedded `touch <sentinel>` would
    // fire and the assertion below would trip.
    let malicious = format!("\"; touch {} ; echo \"", sentinel.display());
    let memory_id = store_memory(data_dir.path(), &malicious, &memory_bin);

    let rows_before = row_count(data_dir.path(), &memory_bin);
    assert_eq!(
        rows_before, 1,
        "expected 1 seeded memory, got {rows_before}"
    );

    // Run `memory-dream test` with an echo-based headless command.
    // The --command-override flag wins over whatever dream.toml was
    // auto-detected into, so this test works on CI machines with or
    // without `claude` on PATH.
    let out = Command::new(&dream_bin)
        .args([
            "test",
            &memory_id,
            "--backend",
            "headless",
            "--command-override",
            "echo '{prompt}'",
        ])
        .env("AGENT_MEMORY_DIR", data_dir.path())
        .output()
        .expect("spawn memory-dream test");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // The subprocess must have succeeded — a non-zero exit would hide
    // whether the underlying defense worked.
    assert!(
        out.status.success(),
        "memory-dream test exited non-zero. stdout={stdout} stderr={stderr}"
    );

    // Invariant #1: the `test_failed` surface fires as expected. The
    // `echo '{prompt}'` override produces a plain-text dump of the prompt,
    // which the condenser's JSON parser correctly rejects — that's the
    // expected shape for this test. The stdout MUST contain a hint from
    // the parse-failure path that includes the prompt text (so we know
    // echo received the malicious payload and surfaced it faithfully
    // rather than, say, being word-split by a shell).
    //
    // The condense layer truncates the parse-error detail at 120 chars,
    // so we assert on the `test_failed` status token and one of the
    // prompt-framing tokens that must be present regardless of truncation.
    assert!(
        stdout.contains("status=\"test_failed\""),
        "expected test_failed result; got: {stdout}"
    );
    assert!(
        stdout.contains("condensation assistant") || stdout.contains("condensed"),
        "expected echo to have surfaced the prompt text or condenser envelope; \
         got: {stdout}"
    );

    // Invariant #2 — the hard one: the sentinel file must NOT exist.
    assert!(
        !sentinel.exists(),
        "shell injection succeeded: {sentinel:?} was created. \
         stdout={stdout} stderr={stderr}"
    );

    // Invariant #3: no DB writes from the `test` subcommand. Row count
    // must equal the pre-test snapshot.
    let rows_after = row_count(data_dir.path(), &memory_bin);
    assert_eq!(
        rows_after, rows_before,
        "expected no DB writes; rows went {rows_before} -> {rows_after}"
    );

    Ok(())
}

#[test]
fn memory_dream_test_never_executes_shell_metachars() {
    run_probe().expect("probe run failed");
}
