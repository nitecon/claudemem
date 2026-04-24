//! Codex `config.toml` merge helper used by `memory setup rules`.
//!
//! When the memory-rules block is installed into Codex's `AGENTS.md`, we
//! must also disable Codex's built-in memory feature ("Chronicle"). Leaving
//! it on would cause the agent to write into BOTH Codex's native memory
//! store and this tool's SQLite store in parallel — the exact duplication
//! the rules block exists to prevent.
//!
//! The disable mechanism is documented at
//! <https://developers.openai.com/codex/memories>: set `memories = false`
//! under `[features]` in Codex's `config.toml`. Today's default happens to
//! be `false`, but we set it explicitly so a later in-app toggle by the
//! user can't silently re-enable native memory behind our back while the
//! rules block is still in place.
//!
//! Path resolution mirrors the rules module's AGENTS.md precedence:
//!
//!   1. `$CODEX_HOME/config.toml` when `CODEX_HOME` is set and points at a
//!      directory that exists.
//!   2. `~/.codex/config.toml` when `~/.codex/` exists.
//!   3. `~/.config/codex/config.toml` when only the XDG fallback exists.
//!   4. Skip Codex entirely.
//!
//! The merge discipline matches the sibling JSON helpers:
//!
//! - Parse the existing file as a TOML document, preserve every top-level
//!   table and every key inside `[features]`.
//! - Fail loudly on corrupt TOML rather than silently overwriting.
//! - Writes are atomic (`.new` + rename).
//! - Re-running is a no-op when the key is already at the target value.
//!
//! On remove, we delete `features.memories` so absence is restored. If
//! `[features]` becomes empty after the removal we drop the table entirely;
//! if the whole file becomes empty we leave an empty string in place rather
//! than deleting the file (least-surprise: the user may have intentionally
//! created the file, same policy as the Claude settings.json remove flow).

use crate::setup::settings_json::SettingsOutcome;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use toml::Value;

/// TOML table that holds Codex feature toggles.
pub const FEATURES_TABLE: &str = "features";

/// Key under `[features]` that governs native memory (Chronicle).
pub const MEMORIES_KEY: &str = "memories";

/// Resolve Codex's `config.toml` path, honoring `CODEX_HOME` then falling
/// back to `~/.codex/` and then `~/.config/codex/`. Returns `None` when no
/// Codex install is visible — callers skip Codex in that case.
///
/// `codex_home_override` lets tests drive resolution without leaking env
/// state across the runner; production callers pass
/// `std::env::var("CODEX_HOME").ok().map(PathBuf::from)`.
pub fn config_path(home: &Path, codex_home_override: Option<&Path>) -> Option<PathBuf> {
    if let Some(override_dir) = codex_home_override {
        if override_dir.is_dir() {
            return Some(override_dir.join("config.toml"));
        }
        // Explicit override pointing at a missing directory → skip Codex.
        // Matches the rules.rs AGENTS.md resolver: we must not second-guess
        // an explicit user choice by silently writing elsewhere.
        return None;
    }
    let primary = home.join(".codex");
    if primary.is_dir() {
        return Some(primary.join("config.toml"));
    }
    let xdg = home.join(".config").join("codex");
    if xdg.is_dir() {
        return Some(xdg.join("config.toml"));
    }
    None
}

/// Resolve a Codex `config.toml` path that pairs with a given rule file.
/// Returns `Some` only for Codex `AGENTS.md` files — the `memories`
/// feature is a Codex concept.
///
/// The `codex_home_override` argument is threaded through so the caller
/// can keep `CODEX_HOME` precedence consistent with the rule-file
/// resolution in `rules.rs`. When the rule file is an AGENTS.md under a
/// known Codex home we use that home as the config target; otherwise we
/// fall back to [`config_path`] with the process's real `$HOME`.
pub fn config_path_for_rule_file(
    rule_file: &Path,
    home: &Path,
    codex_home_override: Option<&Path>,
) -> Option<PathBuf> {
    let name = rule_file
        .file_name()?
        .to_string_lossy()
        .to_ascii_lowercase();
    if name != "agents.md" {
        return None;
    }
    // If the AGENTS.md sits directly inside a directory, use that
    // directory as Codex's home for the matching config.toml. This keeps
    // the pairing unambiguous when the user has set CODEX_HOME to a
    // custom path: the rule file and the config end up in the same
    // directory, as Codex itself expects.
    let parent = rule_file.parent()?;
    if parent.is_dir() {
        return Some(parent.join("config.toml"));
    }
    // Defensive fallback: the rule file exists but its parent doesn't
    // (shouldn't happen in practice). Fall back to the normal resolver.
    config_path(home, codex_home_override)
}

/// Set `[features] memories = false` in the config file at `path`.
///
/// 1. Missing file → create with just the `[features]` table.
/// 2. Existing file that parses to a TOML document → ensure `[features]`
///    exists and is a table, then set `memories = false`. Preserve every
///    other top-level table and every other `[features]` key.
/// 3. Existing file with a `features` field that is not a table, or with a
///    `features.memories` that is not a boolean → bail with a typed error
///    naming the path so the user can fix it by hand.
/// 4. Key already `false` → no-op.
pub fn disable_memories(path: &Path) -> Result<SettingsOutcome> {
    match read_document(path)? {
        None => {
            let mut doc = toml::value::Table::new();
            let mut features = toml::value::Table::new();
            features.insert(MEMORIES_KEY.to_string(), Value::Boolean(false));
            doc.insert(FEATURES_TABLE.to_string(), Value::Table(features));
            write_document(path, &doc)?;
            Ok(SettingsOutcome::Created)
        }
        Some(mut doc) => {
            let features = ensure_features_table(&mut doc, path)?;
            match features.get(MEMORIES_KEY) {
                Some(Value::Boolean(false)) => return Ok(SettingsOutcome::AlreadyCorrect),
                Some(Value::Boolean(_)) => {
                    // Flipping `true` → `false` is exactly the case that
                    // motivated the explicit-write policy, and is
                    // well-typed, so no error.
                }
                Some(other) => bail!(
                    "config file {} has wrong shape for `features.{}`: expected bool, got {}",
                    path.display(),
                    MEMORIES_KEY,
                    shape_name(other)
                ),
                None => {}
            }
            features.insert(MEMORIES_KEY.to_string(), Value::Boolean(false));
            write_document(path, &doc)?;
            Ok(SettingsOutcome::Updated)
        }
    }
}

/// Remove `features.memories` from the config file at `path`.
///
/// - Missing file → `AlreadyAbsent`.
/// - Missing `[features]` table → `AlreadyAbsent`.
/// - Missing `memories` key → `AlreadyAbsent`.
/// - Present key → remove it; if `[features]` becomes empty, drop the table.
///
/// The file itself is never deleted: a user who created `config.toml` for
/// other reasons (a different `[features]` toggle, a `[ui]` theme, etc.)
/// shouldn't lose it just because the only memory-related key went away.
pub fn remove_memories(path: &Path) -> Result<SettingsOutcome> {
    match read_document(path)? {
        None => Ok(SettingsOutcome::AlreadyAbsent),
        Some(mut doc) => {
            let Some(features_val) = doc.get_mut(FEATURES_TABLE) else {
                return Ok(SettingsOutcome::AlreadyAbsent);
            };
            let features = match features_val {
                Value::Table(t) => t,
                other => bail!(
                    "config file {} has wrong shape for `[{}]`: expected table, got {}",
                    path.display(),
                    FEATURES_TABLE,
                    shape_name(other)
                ),
            };
            if features.remove(MEMORIES_KEY).is_none() {
                return Ok(SettingsOutcome::AlreadyAbsent);
            }
            if features.is_empty() {
                doc.remove(FEATURES_TABLE);
            }
            write_document(path, &doc)?;
            Ok(SettingsOutcome::Removed)
        }
    }
}

// -- I/O helpers -------------------------------------------------------------

/// Ensure `[features]` exists as a table and return a mutable ref to it.
/// Creates the table when missing; bails with a typed error when the key
/// exists but is the wrong shape (e.g. a stray scalar a user mistakenly
/// wrote). Never silently replaces a user value.
fn ensure_features_table<'a>(
    doc: &'a mut toml::value::Table,
    path: &Path,
) -> Result<&'a mut toml::value::Table> {
    let entry = doc
        .entry(FEATURES_TABLE.to_string())
        .or_insert_with(|| Value::Table(toml::value::Table::new()));
    match entry {
        Value::Table(t) => Ok(t),
        other => bail!(
            "config file {} has wrong shape for `[{}]`: expected table, got {}",
            path.display(),
            FEATURES_TABLE,
            shape_name(other)
        ),
    }
}

/// Read `path` and parse it as a TOML document. Returns `Ok(None)` when the
/// file does not exist or is empty (the caller decides whether to create
/// it). Parse failures are surfaced with the file path in context.
fn read_document(path: &Path) -> Result<Option<toml::value::Table>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(
                anyhow::Error::new(e).context(format!("read config file {}", path.display()))
            )
        }
    };

    if raw.trim().is_empty() {
        return Ok(None);
    }

    let doc: toml::value::Table = toml::from_str(&raw)
        .with_context(|| format!("parse config file {} as TOML", path.display()))?;
    Ok(Some(doc))
}

/// Serialize a TOML table and write atomically. Using `toml::to_string_pretty`
/// gives us a stable, human-friendly layout that matches what a hand-edited
/// Codex config would look like.
fn write_document(path: &Path, doc: &toml::value::Table) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent of {}", path.display()))?;
    }

    let body = toml::to_string_pretty(doc)
        .with_context(|| format!("serialize config for {}", path.display()))?;

    let tmp = temp_path(path);
    std::fs::write(&tmp, &body)
        .with_context(|| format!("write temp config file {}", tmp.display()))?;
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
        Value::String(_) => "string",
        Value::Integer(_) => "integer",
        Value::Float(_) => "float",
        Value::Boolean(_) => "bool",
        Value::Datetime(_) => "datetime",
        Value::Array(_) => "array",
        Value::Table(_) => "table",
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
                "agent-memory-codex-config-test-{}",
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

    // -- config_path precedence ------------------------------------------------

    #[test]
    fn config_path_honors_codex_home_override() {
        let dir = tempdir().unwrap();
        let home = dir.path().join("home");
        let override_dir = dir.path().join("custom-codex");
        std::fs::create_dir_all(&override_dir).unwrap();
        // Create the fallbacks too to prove the override wins.
        std::fs::create_dir_all(home.join(".codex")).unwrap();
        std::fs::create_dir_all(home.join(".config").join("codex")).unwrap();

        assert_eq!(
            config_path(&home, Some(&override_dir)),
            Some(override_dir.join("config.toml"))
        );
    }

    #[test]
    fn config_path_override_missing_returns_none() {
        let dir = tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(home.join(".codex")).unwrap();
        let missing = dir.path().join("nope");
        assert!(config_path(&home, Some(&missing)).is_none());
    }

    #[test]
    fn config_path_prefers_dot_codex_over_xdg() {
        let dir = tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(home.join(".codex")).unwrap();
        std::fs::create_dir_all(home.join(".config").join("codex")).unwrap();
        assert_eq!(
            config_path(&home, None),
            Some(home.join(".codex").join("config.toml"))
        );
    }

    #[test]
    fn config_path_falls_back_to_xdg() {
        let dir = tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(home.join(".config").join("codex")).unwrap();
        assert_eq!(
            config_path(&home, None),
            Some(home.join(".config").join("codex").join("config.toml"))
        );
    }

    #[test]
    fn config_path_none_when_nothing_installed() {
        let dir = tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        assert!(config_path(&home, None).is_none());
    }

    #[test]
    fn config_path_for_rule_file_uses_rule_parent() {
        // AGENTS.md in a specific dir → config.toml in the same dir.
        let dir = tempdir().unwrap();
        let codex = dir.path().join(".codex");
        std::fs::create_dir_all(&codex).unwrap();
        let agents = codex.join("AGENTS.md");
        std::fs::write(&agents, "").unwrap();

        assert_eq!(
            config_path_for_rule_file(&agents, dir.path(), None),
            Some(codex.join("config.toml"))
        );
    }

    #[test]
    fn config_path_for_rule_file_returns_none_for_non_codex_files() {
        let dir = tempdir().unwrap();
        let claude = dir.path().join("CLAUDE.md");
        std::fs::write(&claude, "").unwrap();
        assert_eq!(config_path_for_rule_file(&claude, dir.path(), None), None);
    }

    // -- disable_memories ------------------------------------------------------

    #[test]
    fn disable_memories_creates_file_when_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let out = disable_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Created);
        let body = read(&path);
        assert!(body.contains("[features]"), "got: {body}");
        assert!(body.contains("memories = false"), "got: {body}");
    }

    #[test]
    fn disable_memories_preserves_unrelated_tables_and_features_keys() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[ui]
theme = "dark"

[features]
other_toggle = true
"#,
        )
        .unwrap();

        let out = disable_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Updated);

        let parsed: toml::value::Table = toml::from_str(&read(&path)).unwrap();
        let ui = parsed.get("ui").and_then(|v| v.as_table()).unwrap();
        assert_eq!(ui.get("theme").unwrap().as_str(), Some("dark"));

        let features = parsed.get("features").and_then(|v| v.as_table()).unwrap();
        assert_eq!(features.get("other_toggle").unwrap().as_bool(), Some(true));
        assert_eq!(features.get("memories").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn disable_memories_flips_true_to_false() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[features]
memories = true
"#,
        )
        .unwrap();
        let out = disable_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Updated);
        let parsed: toml::value::Table = toml::from_str(&read(&path)).unwrap();
        assert_eq!(
            parsed
                .get("features")
                .and_then(|v| v.as_table())
                .and_then(|t| t.get("memories"))
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn disable_memories_is_idempotent_when_already_false() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let body = "[features]\nmemories = false\n";
        fs::write(&path, body).unwrap();
        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = disable_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyCorrect);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(
            before_mtime, after_mtime,
            "idempotent path must not rewrite the file"
        );
    }

    #[test]
    fn disable_memories_adds_features_table_when_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[ui]
theme = "dark"
"#,
        )
        .unwrap();
        let out = disable_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Updated);
        let parsed: toml::value::Table = toml::from_str(&read(&path)).unwrap();
        assert_eq!(
            parsed
                .get("ui")
                .and_then(|v| v.as_table())
                .and_then(|t| t.get("theme"))
                .and_then(|v| v.as_str()),
            Some("dark")
        );
        assert_eq!(
            parsed
                .get("features")
                .and_then(|v| v.as_table())
                .and_then(|t| t.get("memories"))
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn disable_memories_fails_loudly_on_corrupt_toml() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "[features\nmemories = false\n";
        fs::write(&path, original).unwrap();
        let err = disable_memories(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("config.toml"), "error msg: {msg}");
        assert_eq!(read(&path), original, "corrupt file was modified");
    }

    #[test]
    fn disable_memories_rejects_non_table_features() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "features = \"on\"\n";
        fs::write(&path, original).unwrap();
        let err = disable_memories(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("expected table"), "error msg: {msg}");
        assert_eq!(read(&path), original);
    }

    #[test]
    fn disable_memories_rejects_non_bool_memories() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "[features]\nmemories = \"on\"\n";
        fs::write(&path, original).unwrap();
        let err = disable_memories(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("expected bool"), "error msg: {msg}");
        assert_eq!(read(&path), original);
    }

    #[test]
    fn disable_memories_treats_empty_file_as_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "").unwrap();
        let out = disable_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Created);
    }

    // -- remove_memories -------------------------------------------------------

    #[test]
    fn remove_memories_deletes_key_when_present() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[ui]
theme = "dark"

[features]
memories = false
other_toggle = true
"#,
        )
        .unwrap();
        let out = remove_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        let parsed: toml::value::Table = toml::from_str(&read(&path)).unwrap();
        let features = parsed.get("features").and_then(|v| v.as_table()).unwrap();
        assert!(!features.contains_key("memories"));
        assert_eq!(features.get("other_toggle").unwrap().as_bool(), Some(true));
        let ui = parsed.get("ui").and_then(|v| v.as_table()).unwrap();
        assert_eq!(ui.get("theme").unwrap().as_str(), Some("dark"));
    }

    #[test]
    fn remove_memories_collapses_empty_features_table() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[ui]
theme = "dark"

[features]
memories = false
"#,
        )
        .unwrap();
        let out = remove_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        let parsed: toml::value::Table = toml::from_str(&read(&path)).unwrap();
        assert!(
            !parsed.contains_key("features"),
            "empty features table must be dropped, not left as [features]"
        );
        let ui = parsed.get("ui").and_then(|v| v.as_table()).unwrap();
        assert_eq!(ui.get("theme").unwrap().as_str(), Some("dark"));
    }

    #[test]
    fn remove_memories_noop_when_key_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[features]
other_toggle = true
"#,
        )
        .unwrap();
        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = remove_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(before_mtime, after_mtime);
    }

    #[test]
    fn remove_memories_noop_when_features_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[ui]
theme = "dark"
"#,
        )
        .unwrap();
        let before_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let out = remove_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        let after_mtime = fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(before_mtime, after_mtime);
    }

    #[test]
    fn remove_memories_noop_when_file_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let out = remove_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::AlreadyAbsent);
        assert!(!path.exists());
    }

    #[test]
    fn remove_memories_leaves_empty_file_when_last_key_collapsed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"[features]
memories = false
"#,
        )
        .unwrap();
        let out = remove_memories(&path).unwrap();
        assert_eq!(out, SettingsOutcome::Removed);
        // File remains (we don't unlink user-owned files), just serialized empty.
        let parsed: toml::value::Table = toml::from_str(&read(&path)).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn remove_memories_rejects_non_table_features() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "features = \"on\"\n";
        fs::write(&path, original).unwrap();
        let err = remove_memories(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("expected table"), "error msg: {msg}");
        assert_eq!(read(&path), original);
    }
}
