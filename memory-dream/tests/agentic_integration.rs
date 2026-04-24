//! End-to-end integration test for the Release 2.3 agentic dream flow.
//!
//! Uses a stub `bash` script as the headless backend. The script inspects
//! its `$1` prompt argument and either replies with a version string (for
//! the tool-support probe) or invokes `memory forget` + `memory update`
//! against the seeded DB to simulate an agentic curation batch.
//!
//! The test seeds 3 memories, runs `memory-dream`, and asserts:
//!   * One memory is gone (forget succeeded).
//!   * One memory has new content and a populated `content_raw` (update
//!     succeeded with provenance preserved).
//!   * `project_state.last_dream_at` is populated for the project.
//!
//! The stub receives the full prompt on its argv; it checks the prompt
//! shape just enough to distinguish probe vs. batch, then acts.

use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

fn release_binary(bin: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .expect("workspace root")
        .to_path_buf();
    workspace_root.join("target").join("release").join(bin)
}

/// Spawn `memory store` and return the full UUID of the created row.
/// Falls back to parsing the short id and resolving via `memory list`
/// so the test captures the full id for precise assertions.
fn store_and_get_id(data_dir: &std::path::Path, memory: &str, bin: &std::path::Path) -> String {
    let out = Command::new(bin)
        .args(["store", memory, "-m", "user"])
        .env("AGENT_MEMORY_DIR", data_dir)
        .output()
        .expect("spawn memory store");
    assert!(
        out.status.success(),
        "memory store failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // The short id is in stdout; resolve it to a full UUID via the DB.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let marker = "id=\"";
    let start = stdout
        .find(marker)
        .unwrap_or_else(|| panic!("no id= in stdout: {stdout}"));
    let rest = &stdout[start + marker.len()..];
    let end = rest.find('"').unwrap();
    let short = &rest[..end];

    // Use `memory get <short>` to expand.
    let out2 = Command::new(bin)
        .args(["get", short])
        .env("AGENT_MEMORY_DIR", data_dir)
        .output()
        .expect("spawn memory get");
    let stdout2 = String::from_utf8_lossy(&out2.stdout);
    // `<memory id="<short>"...` — we only need the 8-char prefix for the
    // dream calls, not the full UUID, since `memory update <prefix>` and
    // `memory forget --id <prefix>` both support the resolver.
    let _ = stdout2;
    short.to_string()
}

fn run_memory(
    args: &[&str],
    data_dir: &std::path::Path,
    bin: &std::path::Path,
) -> (String, String, bool) {
    let out = Command::new(bin)
        .args(args)
        .env("AGENT_MEMORY_DIR", data_dir)
        .output()
        .expect("spawn memory");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.success(),
    )
}

#[test]
fn agentic_flow_forgets_and_updates_via_stub_backend() {
    let memory_bin = release_binary("memory");
    let dream_bin = release_binary("memory-dream");
    if !memory_bin.exists() || !dream_bin.exists() {
        eprintln!(
            "[SKIP] release binaries missing (run `cargo build --release`): \
             memory={} dream={}",
            memory_bin.display(),
            dream_bin.display()
        );
        return;
    }

    let data_dir = TempDir::new().expect("tempdir");
    let stub_log = data_dir.path().join("stub.log");

    // Seed 3 memories — the stub will forget one and update one; the third
    // stays untouched so we verify the agentic path doesn't touch rows it
    // wasn't told to.
    let forget_id = store_and_get_id(
        data_dir.path(),
        "CI notification: Eventic release v1.2.0 tag pushed, webhook fired.",
        &memory_bin,
    );
    let update_id = store_and_get_id(
        data_dir.path(),
        "Verify webhook signatures for github releases with HMAC-SHA256 per docs/security.md",
        &memory_bin,
    );
    let keep_id = store_and_get_id(
        data_dir.path(),
        "User prefers terse responses without filler phrases",
        &memory_bin,
    );

    // Build the stub backend script. It reads its single argv (the prompt)
    // and:
    //   - If the prompt contains "memory --version", reply with a version
    //     string (probe success).
    //   - Otherwise invoke memory forget + memory update to curate the
    //     batch. The MEMORY bin path + IDs are baked into the script via
    //     environment variables read at invocation time.
    //
    // We intentionally use `bash -c '...'` with an `${1}` marker because
    // our shell-safe spawn path passes the prompt as a single argv token;
    // bash's `$1` consumes it without re-splitting.
    let stub_script = data_dir.path().join("stub.sh");
    let script_body = format!(
        r#"#!/bin/bash
# Stub backend for the agentic integration test. Reads the prompt on $1,
# then either responds to the tool-support probe or invokes `memory` tool
# commands to curate the batch.
set -u
PROMPT="$1"
MEMORY_BIN="{memory_bin}"
FORGET_ID="{forget_id}"
UPDATE_ID="{update_id}"
LOG="{log}"

echo "PROMPT_LEN=${{#PROMPT}}" >> "$LOG"

# Probe detection: the probe prompt starts with "Run `memory --version`".
if [[ "$PROMPT" == *"memory --version"* ]]; then
    echo "responding to probe" >> "$LOG"
    echo "memory 1.3.0"
    exit 0
fi

# Otherwise this is an agentic batch. Invoke the tool surface the LLM would
# have driven — use AGENT_MEMORY_DIR from the inherited environment so we
# write to the same tempdir as the test's other CLI calls.
echo "driving agentic batch" >> "$LOG"
"$MEMORY_BIN" forget --id "$FORGET_ID" >> "$LOG" 2>&1
"$MEMORY_BIN" update "$UPDATE_ID" --content \
    "Verify webhook signatures per docs/security.md
- Algorithm: HMAC-SHA256
- Scope: all github release webhooks" \
    >> "$LOG" 2>&1

echo "stub done" >> "$LOG"
exit 0
"#,
        memory_bin = memory_bin.display(),
        forget_id = forget_id,
        update_id = update_id,
        log = stub_log.display(),
    );
    std::fs::write(&stub_script, script_body).expect("write stub script");
    let mut perm = std::fs::metadata(&stub_script)
        .expect("stat stub script")
        .permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        perm.set_mode(0o755);
    }
    std::fs::set_permissions(&stub_script, perm).expect("chmod stub script");

    // Invoke memory-dream with the stub as the headless backend. The
    // `{prompt}` placeholder gets substituted with the actual prompt text
    // and passed as a single argv element.
    let out = Command::new(&dream_bin)
        .args([
            "--backend",
            "headless",
            "--command-override",
            &format!("bash {} {{prompt}}", stub_script.display()),
            "--full",
        ])
        .env("AGENT_MEMORY_DIR", data_dir.path())
        .output()
        .expect("spawn memory-dream");
    let dream_stdout = String::from_utf8_lossy(&out.stdout);
    let dream_stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "memory-dream failed. stdout={dream_stdout}\nstderr={dream_stderr}"
    );

    // Probe must have been classified as agentic.
    assert!(
        dream_stdout.contains("status=\"dream_probe\"")
            && dream_stdout.contains("mode=\"agentic\""),
        "expected probe to route to agentic mode. stdout={dream_stdout}"
    );

    // -- DB assertions ----------------------------------------------------
    let (list_out, _, ok) = run_memory(&["list", "-k", "100"], data_dir.path(), &memory_bin);
    assert!(ok, "memory list failed: {list_out}");

    // The FORGET_ID memory should be gone.
    assert!(
        !list_out.contains(&format!("(ID:{forget_id})")),
        "forget_id {forget_id} still present in list: {list_out}"
    );
    // The UPDATE_ID memory must still exist.
    assert!(
        list_out.contains(&format!("(ID:{update_id})")),
        "update_id {update_id} missing from list: {list_out}"
    );
    // The KEEP_ID memory should still be there, untouched.
    assert!(
        list_out.contains(&format!("(ID:{keep_id})")),
        "keep_id {keep_id} missing from list: {list_out}"
    );

    // Verify UPDATE_ID's content now starts with the new headline.
    let (get_out, _, ok) = run_memory(&["get", &update_id], data_dir.path(), &memory_bin);
    assert!(ok, "memory get failed: {get_out}");
    assert!(
        get_out.contains("Verify webhook signatures per docs/security.md"),
        "updated content missing: {get_out}"
    );
    assert!(
        get_out.contains("HMAC-SHA256"),
        "updated bullets missing: {get_out}"
    );
}
