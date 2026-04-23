//! Claude Code `settings.json` merge helper used by `memory setup rules`.
//!
//! When the rules block is installed into a Claude rule file (`CLAUDE.md`),
//! we must also disable Claude Code's native auto-memory feature by writing
//! `"autoMemoryEnabled": false` into the matching `settings.json`. Leaving
//! the native feature on would cause the agent to write memories into BOTH
//! Claude's `MEMORY.md` and this tool's SQLite store in parallel — which is
//! the exact duplication the rules block is designed to avoid.
//!
//! The merge is deliberately conservative:
//!
//! - We parse the existing file as a JSON object and preserve every key the
//!   user already has (`theme`, `model`, `hooks`, permissions, etc.).
//! - We fail loudly on corrupt / non-object JSON rather than silently
//!   overwriting a broken settings file.
//! - Writes are atomic (`.new` + rename) so a crash mid-write cannot leave
//!   the user with a half-serialized settings file.
//! - Re-running is a no-op when the key is already at the target value — no
//!   pointless file rewrites, no spurious mtime churn.
//!
//! `settings.json` ties to the matching scope of the rule file:
//!
//! | Rule file              | Settings file              |
//! |------------------------|----------------------------|
//! | `~/.claude/CLAUDE.md`  | `~/.claude/settings.json`  |
//! | `./CLAUDE.md`          | `./.claude/settings.json`  |
//!
//! Non-Claude rule files (GEMINI.md, AGENTS.md) have no counterpart — the
//! `autoMemoryEnabled` key is a Claude Code concept only.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// The Claude Code settings key that governs native auto-memory. Setting it
/// to `false` prevents Claude Code from writing to its own `MEMORY.md` while
/// our rules block redirects the agent to the `memory` CLI instead.
pub const AUTO_MEMORY_KEY: &str = "autoMemoryEnabled";

/// Outcome of a settings merge — used by callers to emit status lines that
/// distinguish "we actually changed the file" from "already in the desired
/// state". Keeps idempotent re-runs quiet in the rendered install log.
#[derive(Debug, PartialEq, Eq)]
pub enum SettingsOutcome {
    /// File did not exist; created with just the key.
    Created,
    /// Key was added or flipped to the target value.
    Updated,
    /// Key was already at the target value; no write performed.
    AlreadyCorrect,
    /// Removal: key was present and is now absent.
    Removed,
    /// Removal: key was already absent; no write performed.
    AlreadyAbsent,
}

/// Resolve the `settings.json` path that pairs with a given Claude rule
/// file. Returns `None` for non-Claude rule files (GEMINI.md, AGENTS.md),
/// which have no matching Claude Code settings concept.
///
/// The scope is inferred from the rule file's location, not from a flag:
///
/// - `<dir>/CLAUDE.md` where `<dir>` ends with `.claude` → `<dir>/settings.json`
///   (user scope, e.g. `~/.claude/CLAUDE.md` → `~/.claude/settings.json`).
/// - `<dir>/CLAUDE.md` elsewhere → `<dir>/.claude/settings.json`
///   (project scope, e.g. `./CLAUDE.md` → `./.claude/settings.json`).
///
/// This mirrors Claude Code's own scope resolution so we never write to a
/// location the harness wouldn't also read from.
pub fn settings_path_for_rule_file(rule_file: &Path) -> Option<PathBuf> {
    // Only Claude rule files have a matching settings.json. Match on the
    // file name case-insensitively so a user-renamed `claude.md` still pairs
    // correctly on case-insensitive filesystems.
    let name = rule_file
        .file_name()?
        .to_string_lossy()
        .to_ascii_lowercase();
    if name != "claude.md" {
        return None;
    }

    let parent = rule_file.parent()?;

    // User scope: parent directory is literally `.claude`. Pair with a
    // sibling `settings.json` in the same directory.
    let parent_name = parent
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if parent_name == ".claude" {
        return Some(parent.join("settings.json"));
    }

    // Project scope: `CLAUDE.md` sits at the repo root alongside a
    // `.claude/` subdirectory that holds settings.json.
    Some(parent.join(".claude").join("settings.json"))
}

/// Merge `"autoMemoryEnabled": false` into the settings file at `path`.
///
/// Semantics (see module docs for rationale):
///
/// 1. Missing file → create with just `{"autoMemoryEnabled": false}` plus
///    a trailing newline.
/// 2. Existing file that parses to a JSON object → set the key, preserve
///    every other key, write back with sorted keys and 2-space indent.
/// 3. Existing file that is unparseable or not a top-level object → bail
///    with a typed error naming the file so the user can fix it by hand.
/// 4. Key already `false` → no-op (no file rewrite).
pub fn disable_auto_memory(path: &Path) -> Result<SettingsOutcome> {
    match read_object(path)? {
        None => {
            // Fresh install: write a minimal settings.json so we're not
            // creating a key-laden file the user didn't ask for.
            let mut obj = Map::new();
            obj.insert(AUTO_MEMORY_KEY.to_string(), Value::Bool(false));
            write_object(path, &obj)?;
            Ok(SettingsOutcome::Created)
        }
        Some(mut obj) => {
            if obj.get(AUTO_MEMORY_KEY) == Some(&Value::Bool(false)) {
                return Ok(SettingsOutcome::AlreadyCorrect);
            }
            obj.insert(AUTO_MEMORY_KEY.to_string(), Value::Bool(false));
            write_object(path, &obj)?;
            Ok(SettingsOutcome::Updated)
        }
    }
}

/// Remove the `autoMemoryEnabled` key from the settings file at `path`.
///
/// Called by `memory setup rules --remove`. We delete the key rather than
/// force it to `true` because absence and `true` are semantically different
/// in Claude Code and forcing a value would overwrite whatever the user had
/// before `memory setup` ran. When the key is already absent we return
/// `AlreadyAbsent` without touching the file.
///
/// If removing the key leaves the object empty, we still write `{}` back
/// rather than deleting the file — the user may have intentionally created
/// the file and we shouldn't remove it without explicit consent.
pub fn remove_auto_memory(path: &Path) -> Result<SettingsOutcome> {
    match read_object(path)? {
        None => Ok(SettingsOutcome::AlreadyAbsent),
        Some(mut obj) => {
            if obj.remove(AUTO_MEMORY_KEY).is_none() {
                return Ok(SettingsOutcome::AlreadyAbsent);
            }
            write_object(path, &obj)?;
            Ok(SettingsOutcome::Removed)
        }
    }
}

/// Read `path` and parse it as a JSON object. Returns:
///
/// - `Ok(None)` when the file does not exist (the caller decides whether to
///   create it).
/// - `Ok(Some(map))` when the file parses to a top-level object.
/// - `Err(_)` for IO errors, parse failures, and top-level values that are
///   not objects (arrays, scalars, or null). The error names the file path
///   so the user can open and fix it.
fn read_object(path: &Path) -> Result<Option<Map<String, Value>>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow!(e).context(format!("read settings file {}", path.display()))),
    };

    // Empty file is treated as a missing file — same outcome, create fresh.
    if raw.trim().is_empty() {
        return Ok(None);
    }

    let value: Value = serde_json::from_str(&raw).with_context(|| {
        format!(
            "parse settings file {} as JSON (is it JSONC? comments are not supported)",
            path.display()
        )
    })?;

    match value {
        Value::Object(map) => Ok(Some(map)),
        other => bail!(
            "settings file {} has unexpected top-level shape: expected object, got {}",
            path.display(),
            shape_name(&other)
        ),
    }
}

/// Write a JSON object to `path` atomically, with 2-space indent and keys
/// sorted alphabetically. Atomic write prevents half-serialized settings
/// files from a crash mid-write.
///
/// `serde_json::to_string_pretty` defaults to preserving insertion order;
/// we copy into a `BTreeMap` so the output is deterministic and diff-friendly
/// across multiple runs regardless of the order in which keys were inserted.
fn write_object(path: &Path, obj: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent of {}", path.display()))?;
    }

    // Stable key order via BTreeMap: same input always produces the same
    // output file, which is what users expect from a config-management tool.
    let sorted: std::collections::BTreeMap<&String, &Value> = obj.iter().collect();
    let mut body = serde_json::to_string_pretty(&sorted)
        .with_context(|| format!("serialize settings for {}", path.display()))?;
    body.push('\n');

    let tmp = temp_path(path);
    std::fs::write(&tmp, &body)
        .with_context(|| format!("write temp settings file {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("atomically rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Sibling `.new` path used for the atomic-rename staging file. Picked as a
/// sibling rather than a `/tmp` path so the rename stays within one
/// filesystem (cross-filesystem rename would fall back to copy+delete and
/// lose the atomicity guarantee).
fn temp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".new");
    PathBuf::from(s)
}

fn shape_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn read(path: &Path) -> String {
        fs::read_to_string(path).unwrap()
    }

    /// Minimal isolated tempdir under the OS temp dir so tests don't depend
    /// on the `tempfile` crate (not in this project's deps). Each test gets
    /// a unique directory based on a UUID v4 — the existing `uuid` dep is
    /// sufficient. Mirrors the helper already used by the `skill` setup tests.
    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "agent-memory-settings-test-{}",
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            // Best-effort cleanup; leave the dir behind on error so a failing
            // test's artifacts stay inspectable.
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Wrapped in a `Result` so test sites can use `.unwrap()` — matches
    /// the ergonomics of `tempfile::tempdir()` and keeps the test body
    /// readable. Creation currently never fails at this layer (the heavy
    /// lifting lives in `TestDir::new`), but returning `Result` future-
    /// proofs us against an `fs::create_dir_all` failure being surfaced.
    fn tempdir() -> std::io::Result<TestDir> {
        Ok(TestDir::new())
    }

    #[test]
    fn settings_path_user_scope_under_dot_claude() {
        // Simulate `~/.claude/CLAUDE.md` → `~/.claude/settings.json`.
        let p = Path::new("/home/alice/.claude/CLAUDE.md");
        assert_eq!(
            settings_path_for_rule_file(p),
            Some(PathBuf::from("/home/alice/.claude/settings.json"))
        );
    }

    #[test]
    fn settings_path_project_scope_gets_nested_dot_claude() {
        // Simulate `./CLAUDE.md` at a repo root → `./.claude/settings.json`.
        let p = Path::new("/repo/CLAUDE.md");
        assert_eq!(
            settings_path_for_rule_file(p),
            Some(PathBuf::from("/repo/.claude/settings.json"))
        );
    }

    #[test]
    fn settings_path_returns_none_for_non_claude_files() {
        // GEMINI/AGENTS rule files have no Claude settings counterpart.
        assert_eq!(
            settings_path_for_rule_file(Path::new("/home/alice/.gemini/GEMINI.md")),
            None
        );
        assert_eq!(
            settings_path_for_rule_file(Path::new("/home/alice/.codex/AGENTS.md")),
            None
        );
    }

    #[test]
    fn disable_auto_memory_creates_file_when_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let out = disable_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Created);
        let body = read(&path);
        assert!(body.contains("\"autoMemoryEnabled\": false"), "got: {body}");
        assert!(body.ends_with('\n'));
    }

    #[test]
    fn disable_auto_memory_preserves_existing_keys() {
        // The load-bearing case: a user-managed settings.json with theme,
        // model, hooks, etc. must come out the other side with every key
        // intact plus the new autoMemoryEnabled entry.
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(
            &path,
            r#"{
  "theme": "dark",
  "model": "sonnet",
  "hooks": {"PreToolUse": []}
}
"#,
        )
        .unwrap();

        let out = disable_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Updated);

        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("theme").unwrap(), "dark");
        assert_eq!(obj.get("model").unwrap(), "sonnet");
        assert!(obj.get("hooks").is_some(), "hooks key was dropped");
        assert_eq!(obj.get("autoMemoryEnabled").unwrap(), &Value::Bool(false));
    }

    #[test]
    fn disable_auto_memory_flips_true_to_false() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"autoMemoryEnabled": true}"#).unwrap();
        let out = disable_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Updated);
        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        assert_eq!(
            parsed
                .as_object()
                .unwrap()
                .get("autoMemoryEnabled")
                .unwrap(),
            &Value::Bool(false)
        );
    }

    #[test]
    fn disable_auto_memory_is_idempotent_when_already_false() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let body = "{\n  \"autoMemoryEnabled\": false\n}\n";
        fs::write(&path, body).unwrap();

        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = disable_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyCorrect);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(
            before_mtime, after_mtime,
            "idempotent path must not rewrite the file"
        );
    }

    #[test]
    fn disable_auto_memory_fails_loudly_on_corrupt_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let original = r#"{"theme": "dark",,,"#;
        fs::write(&path, original).unwrap();

        let err = disable_auto_memory(&path).unwrap_err();
        // Error message must mention the file path so the user can find it.
        let msg = format!("{err:#}");
        assert!(msg.contains("settings.json"), "error msg: {msg}");
        // File must not have been mutated by the failed operation.
        assert_eq!(read(&path), original, "corrupt file was modified");
    }

    #[test]
    fn disable_auto_memory_rejects_non_object_top_level() {
        // Arrays, strings, numbers are all valid JSON but not valid settings.
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, "[1, 2, 3]").unwrap();
        let err = disable_auto_memory(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("expected object"), "error msg: {msg}");
    }

    #[test]
    fn disable_auto_memory_treats_empty_file_as_missing() {
        // Some editors leave a 0-byte settings.json behind; treat it as a
        // fresh install rather than bailing on a parse error.
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, "").unwrap();
        let out = disable_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Created);
    }

    #[test]
    fn remove_auto_memory_deletes_key_when_present() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"theme": "dark", "autoMemoryEnabled": false}"#).unwrap();
        let out = remove_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        let obj = parsed.as_object().unwrap();
        assert!(
            !obj.contains_key("autoMemoryEnabled"),
            "key was not removed"
        );
        assert_eq!(
            obj.get("theme").unwrap(),
            "dark",
            "unrelated keys must survive removal"
        );
    }

    #[test]
    fn remove_auto_memory_noop_when_key_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"theme": "dark"}"#).unwrap();
        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = remove_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(before_mtime, after_mtime);
    }

    #[test]
    fn remove_auto_memory_noop_when_file_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let out = remove_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        assert!(!path.exists(), "no file should have been created");
    }

    #[test]
    fn remove_auto_memory_leaves_empty_object_when_last_key() {
        // Choosing `{}` over deleting the file: the user may have intentionally
        // created the file; we stick to least-surprise and keep the file alive.
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"autoMemoryEnabled": false}"#).unwrap();
        let out = remove_auto_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        assert!(parsed.as_object().unwrap().is_empty());
    }

    #[test]
    fn write_object_produces_sorted_keys_and_two_space_indent() {
        // Determinism check: two runs with keys inserted in different orders
        // must produce byte-identical output.
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("a.json");
        let p2 = dir.path().join("b.json");

        let mut m1 = Map::new();
        m1.insert("zebra".into(), Value::Bool(true));
        m1.insert("alpha".into(), Value::Bool(false));
        write_object(&p1, &m1).unwrap();

        let mut m2 = Map::new();
        m2.insert("alpha".into(), Value::Bool(false));
        m2.insert("zebra".into(), Value::Bool(true));
        write_object(&p2, &m2).unwrap();

        let body1 = read(&p1);
        let body2 = read(&p2);
        assert_eq!(body1, body2, "sorted output must be deterministic");
        assert!(body1.starts_with("{\n  \"alpha\""));
        assert!(body1.contains("\"zebra\""));
        assert!(body1.ends_with("}\n"));
    }
}
