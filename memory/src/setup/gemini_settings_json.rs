//! Gemini CLI `settings.json` merge helper used by `memory setup rules`.
//!
//! When the memory-rules block is installed into `~/.gemini/GEMINI.md`, we
//! must also disable Gemini CLI's built-in `save_memory` tool so the model
//! stops autonomously appending to `~/.gemini/GEMINI.md`. Leaving it enabled
//! would cause the agent to write into BOTH Gemini's native memory file and
//! this tool's SQLite store in parallel — the exact duplication the rules
//! block exists to prevent.
//!
//! The disable mechanism is documented at
//! <https://geminicli.com/docs/reference/configuration/>: add the tool name
//! to the top-level `excludeTools` array in `~/.gemini/settings.json`.
//!
//! The merge is deliberately conservative, matching the Claude
//! `settings_json` sibling:
//!
//! - We parse the existing file as a JSON object and preserve every key the
//!   user already has (`theme`, `model`, anything else Gemini grows later).
//! - Within `excludeTools` we preserve every pre-existing entry and only
//!   append `"save_memory"` when it is not already present.
//! - We fail loudly on corrupt / non-object JSON and on a present-but-
//!   wrong-shape `excludeTools` field rather than silently overwriting.
//! - Writes are atomic (`.new` + rename) so a crash mid-write cannot leave
//!   the user with a half-serialized settings file.
//! - Re-running is a no-op when the entry is already present — no pointless
//!   file rewrites, no spurious mtime churn.
//!
//! Shape we manage:
//!
//! ```json
//! {
//!   "excludeTools": ["save_memory", "...user's other excludes..."]
//! }
//! ```
//!
//! On remove, if `excludeTools` becomes empty, we drop the key entirely so
//! the file ends up as `{}` or whatever other keys the user kept. We never
//! delete the file itself — consistent with the Claude remove flow.

use crate::setup::settings_json::SettingsOutcome;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// Top-level key whose array holds tool names the model must not invoke.
pub const EXCLUDE_TOOLS_KEY: &str = "excludeTools";

/// The specific Gemini built-in we must suppress — its autonomous writes to
/// `~/.gemini/GEMINI.md` would fight the memory-rules block for control of
/// the agent's memory surface.
pub const SAVE_MEMORY_TOOL: &str = "save_memory";

/// Resolve the Gemini settings file that pairs with a given rule file. Only
/// returns `Some` for Gemini rule files — the `save_memory` exclusion is a
/// Gemini CLI concept and has no meaning for Claude or Codex.
///
/// Scope resolution:
///
/// - `<dir>/GEMINI.md` where `<dir>` ends with `.gemini` → `<dir>/settings.json`
///   (user scope, e.g. `~/.gemini/GEMINI.md` → `~/.gemini/settings.json`).
///
/// Gemini only supports user-scope settings today, so there is no
/// project-scope branch. If Gemini grows project settings later, we'll add
/// a branch that mirrors Claude's `./.claude/settings.json` pattern.
pub fn settings_path_for_rule_file(rule_file: &Path) -> Option<PathBuf> {
    let name = rule_file
        .file_name()?
        .to_string_lossy()
        .to_ascii_lowercase();
    if name != "gemini.md" {
        return None;
    }

    let parent = rule_file.parent()?;
    // Gemini always colocates settings.json with GEMINI.md inside the
    // tool's home directory — we only pair when the parent looks like a
    // Gemini home, matching the Claude helper's discipline.
    let parent_name = parent
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if parent_name == ".gemini" {
        return Some(parent.join("settings.json"));
    }
    None
}

/// Merge `"save_memory"` into `excludeTools` in the settings file at `path`.
///
/// Semantics:
///
/// 1. Missing file → create with `{"excludeTools": ["save_memory"]}` plus a
///    trailing newline.
/// 2. Existing file that parses to a JSON object → ensure `excludeTools` is
///    an array, append `"save_memory"` if absent, preserve every other key
///    and every other array entry, write back with sorted top-level keys
///    and 2-space indent.
/// 3. Existing file that is unparseable or not a top-level object → bail.
/// 4. Existing `excludeTools` that is not an array → bail (don't silently
///    replace a user value we can't interpret).
/// 5. `"save_memory"` already in the array → no-op (no file rewrite).
pub fn disable_save_memory(path: &Path) -> Result<SettingsOutcome> {
    match read_object(path)? {
        None => {
            let mut obj = Map::new();
            obj.insert(
                EXCLUDE_TOOLS_KEY.to_string(),
                Value::Array(vec![Value::String(SAVE_MEMORY_TOOL.to_string())]),
            );
            write_object(path, &obj)?;
            Ok(SettingsOutcome::Created)
        }
        Some(mut obj) => {
            let entry = obj.entry(EXCLUDE_TOOLS_KEY.to_string()).or_insert_with(|| {
                // Fresh key: start with an empty array we'll fill below.
                Value::Array(Vec::new())
            });
            let arr = match entry {
                Value::Array(a) => a,
                other => bail!(
                    "settings file {} has wrong shape for `{}`: expected array, got {}",
                    path.display(),
                    EXCLUDE_TOOLS_KEY,
                    shape_name(other)
                ),
            };
            if arr.iter().any(|v| v.as_str() == Some(SAVE_MEMORY_TOOL)) {
                return Ok(SettingsOutcome::AlreadyCorrect);
            }
            arr.push(Value::String(SAVE_MEMORY_TOOL.to_string()));
            write_object(path, &obj)?;
            Ok(SettingsOutcome::Updated)
        }
    }
}

/// Remove `"save_memory"` from `excludeTools` in the settings file at `path`.
///
/// - Missing file → `AlreadyAbsent`.
/// - Present file, missing key → `AlreadyAbsent`.
/// - Present key, missing entry → `AlreadyAbsent`.
/// - Present entry → remove it; if `excludeTools` becomes empty, drop the
///   key entirely so the file doesn't carry a dangling empty array.
///
/// If the top-level object becomes empty after the key removal we still
/// write `{}` — consistent with the Claude remove flow, which leaves a file
/// alive the user may have intentionally created.
pub fn remove_save_memory(path: &Path) -> Result<SettingsOutcome> {
    match read_object(path)? {
        None => Ok(SettingsOutcome::AlreadyAbsent),
        Some(mut obj) => {
            let Some(entry) = obj.get_mut(EXCLUDE_TOOLS_KEY) else {
                return Ok(SettingsOutcome::AlreadyAbsent);
            };
            let arr = match entry {
                Value::Array(a) => a,
                other => bail!(
                    "settings file {} has wrong shape for `{}`: expected array, got {}",
                    path.display(),
                    EXCLUDE_TOOLS_KEY,
                    shape_name(other)
                ),
            };
            let original_len = arr.len();
            arr.retain(|v| v.as_str() != Some(SAVE_MEMORY_TOOL));
            if arr.len() == original_len {
                return Ok(SettingsOutcome::AlreadyAbsent);
            }
            // Collapse an empty array: leaving `"excludeTools": []` around
            // would be confusing the next time a human opens the file.
            if arr.is_empty() {
                obj.remove(EXCLUDE_TOOLS_KEY);
            }
            write_object(path, &obj)?;
            Ok(SettingsOutcome::Removed)
        }
    }
}

// -- I/O helpers (local copies so the Gemini shape stays self-contained) ----

/// Read `path` and parse it as a JSON object. Mirrors the sibling helper in
/// `settings_json` but kept local so each per-agent module is independently
/// auditable.
fn read_object(path: &Path) -> Result<Option<Map<String, Value>>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow!(e).context(format!("read settings file {}", path.display()))),
    };

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

/// Write a JSON object atomically with sorted top-level keys and 2-space
/// indent. Same deterministic-output discipline as the Claude sibling.
fn write_object(path: &Path, obj: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent of {}", path.display()))?;
    }

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

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "agent-memory-gemini-settings-test-{}",
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
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn tempdir() -> std::io::Result<TestDir> {
        Ok(TestDir::new())
    }

    #[test]
    fn settings_path_user_scope_under_dot_gemini() {
        let p = Path::new("/home/alice/.gemini/GEMINI.md");
        assert_eq!(
            settings_path_for_rule_file(p),
            Some(PathBuf::from("/home/alice/.gemini/settings.json"))
        );
    }

    #[test]
    fn settings_path_returns_none_for_non_gemini_files() {
        assert_eq!(
            settings_path_for_rule_file(Path::new("/home/alice/.claude/CLAUDE.md")),
            None
        );
        assert_eq!(
            settings_path_for_rule_file(Path::new("/home/alice/.codex/AGENTS.md")),
            None
        );
    }

    #[test]
    fn settings_path_returns_none_for_gemini_outside_expected_home() {
        // We only recognize the `~/.gemini/GEMINI.md` placement — a stray
        // GEMINI.md in an unrelated directory shouldn't trigger a settings
        // write to a sibling file the user didn't mean to create.
        assert_eq!(
            settings_path_for_rule_file(Path::new("/tmp/GEMINI.md")),
            None
        );
    }

    #[test]
    fn disable_save_memory_creates_file_when_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let out = disable_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Created);
        let body = read(&path);
        assert!(body.contains("\"excludeTools\""), "got: {body}");
        assert!(body.contains("\"save_memory\""), "got: {body}");
        assert!(body.ends_with('\n'));
    }

    #[test]
    fn disable_save_memory_preserves_existing_keys_and_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(
            &path,
            r#"{
  "theme": "dark",
  "model": "gemini-pro",
  "excludeTools": ["run_shell_command"]
}
"#,
        )
        .unwrap();

        let out = disable_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Updated);

        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("theme").unwrap(), "dark");
        assert_eq!(obj.get("model").unwrap(), "gemini-pro");
        let arr = obj.get("excludeTools").unwrap().as_array().unwrap();
        let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            names.contains(&"run_shell_command"),
            "pre-existing exclude entry was dropped: {names:?}"
        );
        assert!(
            names.contains(&"save_memory"),
            "save_memory was not appended: {names:?}"
        );
    }

    #[test]
    fn disable_save_memory_adds_key_when_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"theme": "dark"}"#).unwrap();
        let out = disable_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Updated);
        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("theme").unwrap(), "dark");
        let arr = obj.get("excludeTools").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0], "save_memory");
    }

    #[test]
    fn disable_save_memory_is_idempotent_when_already_present() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let body = "{\n  \"excludeTools\": [\n    \"save_memory\"\n  ]\n}\n";
        fs::write(&path, body).unwrap();

        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = disable_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyCorrect);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(
            before_mtime, after_mtime,
            "idempotent path must not rewrite the file"
        );
    }

    #[test]
    fn disable_save_memory_fails_loudly_on_corrupt_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let original = r#"{"theme": "dark",,,"#;
        fs::write(&path, original).unwrap();

        let err = disable_save_memory(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("settings.json"), "error msg: {msg}");
        assert_eq!(read(&path), original, "corrupt file was modified");
    }

    #[test]
    fn disable_save_memory_rejects_non_array_exclude_tools() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let original = r#"{"excludeTools": "save_memory"}"#;
        fs::write(&path, original).unwrap();

        let err = disable_save_memory(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("expected array"), "error msg: {msg}");
        // File must not be mutated on the failing typed error.
        assert_eq!(read(&path), original);
    }

    #[test]
    fn disable_save_memory_rejects_non_object_top_level() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, "[1, 2, 3]").unwrap();
        let err = disable_save_memory(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("expected object"), "error msg: {msg}");
    }

    #[test]
    fn disable_save_memory_treats_empty_file_as_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, "").unwrap();
        let out = disable_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Created);
    }

    #[test]
    fn remove_save_memory_deletes_entry_when_present() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(
            &path,
            r#"{"theme": "dark", "excludeTools": ["save_memory", "run_shell_command"]}"#,
        )
        .unwrap();
        let out = remove_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("theme").unwrap(), "dark");
        let arr = obj.get("excludeTools").unwrap().as_array().unwrap();
        let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(!names.contains(&"save_memory"));
        assert!(names.contains(&"run_shell_command"));
    }

    #[test]
    fn remove_save_memory_collapses_empty_array_to_missing_key() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(
            &path,
            r#"{"theme": "dark", "excludeTools": ["save_memory"]}"#,
        )
        .unwrap();
        let out = remove_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("theme").unwrap(), "dark");
        assert!(
            !obj.contains_key("excludeTools"),
            "empty excludeTools must be dropped, not left as []"
        );
    }

    #[test]
    fn remove_save_memory_noop_when_entry_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"excludeTools": ["run_shell_command"]}"#).unwrap();
        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = remove_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(before_mtime, after_mtime);
    }

    #[test]
    fn remove_save_memory_noop_when_key_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"theme": "dark"}"#).unwrap();
        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = remove_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(before_mtime, after_mtime);
    }

    #[test]
    fn remove_save_memory_noop_when_file_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let out = remove_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        assert!(!path.exists());
    }

    #[test]
    fn remove_save_memory_leaves_empty_object_when_last_key_collapsed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, r#"{"excludeTools": ["save_memory"]}"#).unwrap();
        let out = remove_save_memory(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        let parsed: Value = serde_json::from_str(&read(&path)).unwrap();
        assert!(parsed.as_object().unwrap().is_empty());
    }

    #[test]
    fn remove_save_memory_rejects_non_array_exclude_tools() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let original = r#"{"excludeTools": "save_memory"}"#;
        fs::write(&path, original).unwrap();
        let err = remove_save_memory(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("expected array"), "error msg: {msg}");
        assert_eq!(read(&path), original);
    }
}
