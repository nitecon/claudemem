//! Persistent settings for `memory-dream`.
//!
//! Settings live at `$AGENT_MEMORY_DIR/dream.toml` — human-editable, separate
//! from the SQLite store. This module owns:
//!
//! * The [`Settings`] struct (serde round-trip against `dream.toml`).
//! * First-run auto-detect: when the file is absent, probe `PATH` for a
//!   supported CLI (`claude`, then `gemini`) and default to the
//!   `headless` backend. Falls back to `local` + `gemma3` only when no
//!   CLI is found. Never defaults to `disabled` — that's an explicit opt-in.
//! * Atomic save (write to `.new`, rename) so a crash mid-write cannot leave
//!   a half-parsed TOML file on disk.
//! * [`Settings::effective`] which merges CLI-flag overrides over the
//!   loaded file for a single invocation (no mutation of state).
//!
//! The backend mode is a string-enum so `dream.toml` stays readable — we
//! deliberately use `serde(rename_all = "lowercase")` rather than a custom
//! serializer so hand-edited files just work.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default filename inside `$AGENT_MEMORY_DIR`. Lives next to `memory.db`.
pub const SETTINGS_FILENAME: &str = "dream.toml";

/// Default timeout for the headless subprocess, in milliseconds. Sized
/// for agentic batched curation: Claude running through a full `memory
/// update` / `memory forget` / `memory move` sequence across a batch of
/// up to ~100 memories routinely takes several minutes once cold-start +
/// per-memory deliberation + tool-call round trips are combined. The
/// previous 30s ceiling was tuned for single-memory non-agentic condense
/// and killed real agentic passes mid-flight.
///
/// Ten minutes gives comfortable headroom for the expected worst case
/// (100-memory agentic batch on a slow provider) while still bounding a
/// wedged subprocess so the dream pass can't hang indefinitely.
pub const DEFAULT_HEADLESS_TIMEOUT_MS: u64 = 600_000;

/// Canonical command template for Claude on first-run auto-detect.
///
/// Includes the flags required for agentic dream mode to actually work:
/// - `--permission-mode bypassPermissions` so non-interactive claude doesn't
///   auto-refuse tool invocations (without this, claude -p returns NO_TOOLS).
/// - `--allowedTools "Bash(memory *)"` scopes shell access to memory-CLI
///   invocations only — prompt-injection attempts that try to run arbitrary
///   shell commands get blocked at the tool-permission layer, not just by
///   prompt wording.
/// - `--model sonnet` pins to the current-latest Sonnet alias rather than
///   a version-stamped tag (e.g. `claude-sonnet-4-6`). Sonnet is the right
///   price/competence point for memory curation — Opus is overkill for
///   bulleted condense + classify + forget work and costs ~5× more per
///   token. The alias tracks Anthropic's rolling Sonnet release without
///   requiring a local command update.
pub const DEFAULT_CLAUDE_COMMAND: &str = "claude --permission-mode bypassPermissions \
     --allowedTools 'Bash(memory *)' --model sonnet -p '{prompt}'";

/// Canonical command template for Gemini on first-run auto-detect.
///
/// Pins the `gemini-2.5-flash` model for the same reason Sonnet is pinned
/// in [`DEFAULT_CLAUDE_COMMAND`]: Flash is cheap, fast, and plenty
/// competent for memory condense/classify work; Pro is overkill. Uses
/// `--model` / `-p` per `gemini --help`; no `--yolo` because Gemini's
/// headless path is the non-agentic fallback (it doesn't pass the
/// tool-support probe) so no tool-approval bypass is needed.
pub const DEFAULT_GEMINI_COMMAND: &str = "gemini --model gemini-2.5-flash -p '{prompt}'";

/// Headless command templates from earlier versions that are known to break
/// the agentic probe or carry a stale model pin. When an existing
/// `dream.toml` is loaded with one of these as its stored command, the
/// binary silently upgrades it to the current [`DEFAULT_CLAUDE_COMMAND`]
/// and resaves the file. User-customized templates (anything not
/// byte-matching an entry here) are never touched.
///
/// Add new entries when a subsequent release introduces a new must-have
/// flag or swaps the default model; never remove entries (users upgrading
/// across multiple hops need the whole history to converge).
const KNOWN_STALE_CLAUDE_COMMANDS: &[&str] = &[
    // Pre-v1.3.0 default — `claude -p` without tool-permission flags
    // auto-refuses Bash invocations in non-interactive mode and returns
    // NO_TOOLS, forcing the dream pass to silently downgrade.
    "claude -p '{prompt}'",
    // v1.3.0 – v1.4.0 default — had tool-permission flags but no model
    // pin, so the subprocess ran whatever Claude's CLI picked as its
    // current default (historically Opus). v1.4.1 pins Sonnet for cost.
    "claude --permission-mode bypassPermissions --allowedTools 'Bash(memory *)' -p '{prompt}'",
];

/// Gemini command templates from earlier versions that are auto-upgraded
/// on load. Same contract as [`KNOWN_STALE_CLAUDE_COMMANDS`]: byte match
/// only, never touch a user-customized template.
const KNOWN_STALE_GEMINI_COMMANDS: &[&str] = &[
    // Pre-v1.4.1 default — no model pin, so fresh installs ran whichever
    // model Gemini's CLI picked up (historically Pro). v1.4.1 pins Flash.
    "gemini -p '{prompt}'",
];

/// Default local model short-name when falling back to the `local` backend.
pub const DEFAULT_LOCAL_MODEL: &str = "gemma3";

/// Errors surfaced by the settings layer.
#[derive(Debug, Error)]
pub enum SettingsError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid TOML at {path}: {source}")]
    ParseToml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("failed to serialize settings: {0}")]
    SerializeToml(#[from] toml::ser::Error),

    /// The configured headless command template could not be tokenized by
    /// `shlex::split` — unbalanced quotes, invalid escape, etc. Surfaced at
    /// load time so the user gets a clean error rather than a panic later
    /// during the dream pass.
    #[error("invalid headless command template {template:?}: {reason}")]
    InvalidHeadlessCommand { template: String, reason: String },
}

/// Backend selection for the dream condenser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendMode {
    /// Spawn an external CLI (e.g. `claude -p ...`) and pipe the prompt via
    /// argv. The command template is configured in [`HeadlessConfig`].
    Headless,
    /// Use the in-process candle backend against the local model in
    /// [`LocalConfig::active_model`].
    Local,
    /// Skip condensation entirely. The dream pass still runs dedup.
    Disabled,
}

impl BackendMode {
    /// Short stable string used in light-XML attribute values and in CLI
    /// overrides. Pinned so downstream parsing doesn't drift.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Headless => "headless",
            Self::Local => "local",
            Self::Disabled => "disabled",
        }
    }
}

impl std::str::FromStr for BackendMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "headless" => Ok(Self::Headless),
            "local" => Ok(Self::Local),
            "disabled" => Ok(Self::Disabled),
            other => Err(format!(
                "invalid backend mode '{other}' (expected headless|local|disabled)"
            )),
        }
    }
}

/// `[backend]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub mode: BackendMode,
}

/// `[local]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    /// Short-name of the currently-active local model (used when
    /// `backend.mode = local`).
    pub active_model: String,
    /// Short-names of every model pulled via `memory-dream --pull`. Maintained
    /// by `--pull` (append, dedup) and `rm` (remove). Read by `list` and by
    /// the model-management UX.
    #[serde(default)]
    pub downloaded_models: Vec<String>,
    /// Device preference for the candle backend: `auto|cpu|metal|cuda`.
    /// Defaults to `auto` — pick the best-available accelerator for the host
    /// and fall back to CPU on init failure. Explicit values are strict: if
    /// the user asks for Metal on a machine without Metal we error rather
    /// than silently downgrade.
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_device() -> String {
    "auto".to_string()
}

/// `[headless]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadlessConfig {
    /// Shell-style command template with a literal `{prompt}` placeholder.
    /// Tokenized via `shlex::split` at use time; never re-interpreted by a shell.
    pub command: String,
    /// Per-invocation timeout in milliseconds. `0` = no timeout.
    #[serde(default = "default_headless_timeout")]
    pub timeout_ms: u64,
}

fn default_headless_timeout() -> u64 {
    DEFAULT_HEADLESS_TIMEOUT_MS
}

/// The full `dream.toml` document.
///
/// All three sections are serialized on every save so hand-editors see the
/// complete shape. Missing sections on load fall back to sensible defaults
/// (TinyLlama/gemma3 for local, Claude template for headless).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub backend: BackendConfig,
    pub local: LocalConfig,
    pub headless: HeadlessConfig,
}

impl Settings {
    /// Effective configuration for a single invocation.
    ///
    /// Precedence (highest wins): CLI flags → file contents. Environment
    /// variables are *not* layered here — settings are the single source of
    /// truth. The returned value is detached from `self` so callers can mutate
    /// freely without re-saving.
    pub fn effective(&self, overrides: &CliOverrides) -> EffectiveConfig {
        let mode = overrides.backend.unwrap_or(self.backend.mode);
        let model = overrides
            .model
            .clone()
            .unwrap_or_else(|| self.local.active_model.clone());
        let command = overrides
            .command
            .clone()
            .unwrap_or_else(|| self.headless.command.clone());
        EffectiveConfig {
            mode,
            active_model: model,
            headless_command: command,
            headless_timeout_ms: self.headless.timeout_ms,
            device: self.local.device.clone(),
        }
    }

    /// Load `dream.toml` from `data_dir`, creating a sensible default if
    /// the file is absent.
    ///
    /// Auto-detect rules (first-run only):
    /// 1. `claude` on PATH → `backend = headless`, Claude command.
    /// 2. Else `gemini` on PATH → `backend = headless`, Gemini command.
    /// 3. Else `backend = local, active_model = gemma3` + stderr warning.
    ///
    /// The resolved defaults are persisted atomically before returning so a
    /// second invocation never re-probes.
    pub fn load(data_dir: &Path) -> Result<Self, SettingsError> {
        let path = data_dir.join(SETTINGS_FILENAME);
        if path.exists() {
            let mut settings = Self::load_existing(&path)?;
            if settings.auto_upgrade_stale_fields() {
                eprintln!(
                    "[INFO] upgraded stale dream.toml — pinned default model + \
                     extended agentic-friendly timeout (see {})",
                    path.display()
                );
                settings.save(data_dir)?;
            }
            return Ok(settings);
        }
        let (settings, warning) = Self::auto_detect_default();
        if let Some(msg) = warning {
            eprintln!("[WARN] {msg}");
        }
        settings.save(data_dir)?;
        Ok(settings)
    }

    /// Timeout value written by pre-v1.4.1 releases. Treated as a "not
    /// hand-customized" sentinel by [`auto_upgrade_stale_fields`] — any
    /// other value means the user manually tuned the field and we leave
    /// it alone.
    const STALE_TIMEOUT_MS: u64 = 30_000;

    /// In-place upgrade of known-stale fields loaded from an older
    /// `dream.toml`. Returns `true` if anything changed so the caller knows
    /// to persist + log. Never mutates user-customized values — only byte
    /// matches against the `KNOWN_STALE_*` tables / the pre-v1.4.1 default
    /// timeout sentinel are rewritten.
    ///
    /// Upgrade paths covered:
    ///   * Claude command: any entry in [`KNOWN_STALE_CLAUDE_COMMANDS`] →
    ///     [`DEFAULT_CLAUDE_COMMAND`].
    ///   * Gemini command: any entry in [`KNOWN_STALE_GEMINI_COMMANDS`] →
    ///     [`DEFAULT_GEMINI_COMMAND`].
    ///   * Timeout: the old 30s ceiling, upgraded only when the command
    ///     is recognizable (stale OR current default for its vendor) so a
    ///     user who manually set `timeout_ms = 30000` on a customized
    ///     command keeps their setting. A custom timeout (e.g. 45_000)
    ///     never matches the sentinel and is always preserved.
    fn auto_upgrade_stale_fields(&mut self) -> bool {
        let mut changed = false;
        let trimmed = self.headless.command.trim().to_string();

        // Command upgrade — Claude vendor family first, then Gemini. The
        // two lists never overlap, so order is purely cosmetic.
        let command_is_recognizable_default;
        if KNOWN_STALE_CLAUDE_COMMANDS.contains(&trimmed.as_str()) {
            self.headless.command = DEFAULT_CLAUDE_COMMAND.to_string();
            changed = true;
            command_is_recognizable_default = true;
        } else if KNOWN_STALE_GEMINI_COMMANDS.contains(&trimmed.as_str()) {
            self.headless.command = DEFAULT_GEMINI_COMMAND.to_string();
            changed = true;
            command_is_recognizable_default = true;
        } else {
            // Unchanged — but we still need to decide whether the timeout
            // upgrade is safe. It's safe iff the user is running the
            // CURRENT default command (i.e. they accepted defaults all
            // the way through). Hand-customized commands keep their
            // timeout untouched to avoid surprising users who deliberately
            // picked a short timeout for their specific binary.
            command_is_recognizable_default =
                trimmed == DEFAULT_CLAUDE_COMMAND || trimmed == DEFAULT_GEMINI_COMMAND;
        }

        // Timeout upgrade — only when the user is on a default command
        // lineage AND the stored timeout still matches the v1.4.0 ceiling.
        if command_is_recognizable_default
            && self.headless.timeout_ms == Self::STALE_TIMEOUT_MS
            && DEFAULT_HEADLESS_TIMEOUT_MS != Self::STALE_TIMEOUT_MS
        {
            self.headless.timeout_ms = DEFAULT_HEADLESS_TIMEOUT_MS;
            changed = true;
        }

        changed
    }

    /// Read and parse an existing settings file. Separated from [`load`] so
    /// tests can exercise the parse-error path without touching the filesystem.
    fn load_existing(path: &Path) -> Result<Self, SettingsError> {
        let body = std::fs::read_to_string(path).map_err(|source| SettingsError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let settings: Settings =
            toml::from_str(&body).map_err(|source| SettingsError::ParseToml {
                path: path.to_path_buf(),
                source,
            })?;
        // Validate the headless command template up-front. Tokenization
        // failure at load time is far more debuggable than a panic deep in
        // the dream pipeline. Every backend mode parses this so users can
        // flip `backend.mode` later without a second write.
        if shlex::split(&settings.headless.command).is_none() {
            return Err(SettingsError::InvalidHeadlessCommand {
                template: settings.headless.command.clone(),
                reason: "unbalanced quotes or invalid escape".to_string(),
            });
        }
        Ok(settings)
    }

    /// Atomic save: serialize, write to `<path>.new`, rename into place.
    pub fn save(&self, data_dir: &Path) -> Result<(), SettingsError> {
        std::fs::create_dir_all(data_dir).map_err(|source| SettingsError::Write {
            path: data_dir.to_path_buf(),
            source,
        })?;
        let path = data_dir.join(SETTINGS_FILENAME);
        let staging = data_dir.join(format!("{SETTINGS_FILENAME}.new"));
        let body = toml::to_string_pretty(self)?;
        std::fs::write(&staging, body).map_err(|source| SettingsError::Write {
            path: staging.clone(),
            source,
        })?;
        std::fs::rename(&staging, &path).map_err(|source| SettingsError::Write {
            path: path.clone(),
            source,
        })?;
        Ok(())
    }

    /// Build the first-run default by probing `PATH`. Returns `(settings,
    /// optional warning-to-print)`. Split out from [`load`] so tests can
    /// exercise the branches deterministically via [`Self::auto_detect_with`].
    fn auto_detect_default() -> (Self, Option<String>) {
        Self::auto_detect_with(|bin| which::which(bin).is_ok())
    }

    /// Auto-detect with an injectable PATH-probe closure. Used by unit tests
    /// to simulate "claude present", "gemini present", or "nothing present"
    /// without mutating the real environment.
    pub fn auto_detect_with(probe: impl Fn(&str) -> bool) -> (Self, Option<String>) {
        if probe("claude") {
            (
                Self::default_headless(DEFAULT_CLAUDE_COMMAND),
                Some(
                    "auto-detected `claude` on PATH — defaulting to headless backend.".to_string(),
                ),
            )
        } else if probe("gemini") {
            (
                Self::default_headless(DEFAULT_GEMINI_COMMAND),
                Some(
                    "auto-detected `gemini` on PATH — defaulting to headless backend.".to_string(),
                ),
            )
        } else {
            (
                Self::default_local(),
                Some(format!(
                    "no `claude` or `gemini` CLI on PATH. Defaulting to local \
                     backend with model '{DEFAULT_LOCAL_MODEL}'. Run \
                     `memory-dream --pull` before condensation will work."
                )),
            )
        }
    }

    /// Headless preset builder. The model field is unused by the headless
    /// path but still seeded with the canonical default so flipping to `local`
    /// later doesn't leave `active_model = ""`.
    fn default_headless(command: &str) -> Self {
        Self {
            backend: BackendConfig {
                mode: BackendMode::Headless,
            },
            local: LocalConfig {
                active_model: DEFAULT_LOCAL_MODEL.to_string(),
                downloaded_models: Vec::new(),
                device: default_device(),
            },
            headless: HeadlessConfig {
                command: command.to_string(),
                timeout_ms: DEFAULT_HEADLESS_TIMEOUT_MS,
            },
        }
    }

    /// Local-backend preset builder. Used when no external CLI was detected.
    fn default_local() -> Self {
        Self {
            backend: BackendConfig {
                mode: BackendMode::Local,
            },
            local: LocalConfig {
                active_model: DEFAULT_LOCAL_MODEL.to_string(),
                downloaded_models: Vec::new(),
                device: default_device(),
            },
            headless: HeadlessConfig {
                command: DEFAULT_CLAUDE_COMMAND.to_string(),
                timeout_ms: DEFAULT_HEADLESS_TIMEOUT_MS,
            },
        }
    }

    /// Append a short model name to `local.downloaded_models` if not already
    /// present. Call sites: `--pull` success path. Preserves insertion order.
    pub fn add_downloaded_model(&mut self, short_name: &str) {
        if !self.local.downloaded_models.iter().any(|m| m == short_name) {
            self.local.downloaded_models.push(short_name.to_string());
        }
    }

    /// Remove a short model name from `local.downloaded_models`. Returns
    /// whether the entry was present. Call sites: `rm <model>` subcommand.
    pub fn remove_downloaded_model(&mut self, short_name: &str) -> bool {
        let before = self.local.downloaded_models.len();
        self.local.downloaded_models.retain(|m| m != short_name);
        self.local.downloaded_models.len() != before
    }

    /// Apply a dotted-key mutation like `config set headless.timeout_ms 60000`.
    ///
    /// Supported keys:
    ///   * `backend.mode` → `headless|local|disabled`
    ///   * `local.active_model` → short-name (must already be in `downloaded_models`,
    ///     enforced by caller — this fn accepts any string so migration/seed
    ///     scripts can set the value before a `--pull` completes)
    ///   * `headless.command` → command template string
    ///   * `headless.timeout_ms` → non-negative integer
    ///
    /// Returns an error with a key list on unknown keys so the UX is honest
    /// about what's configurable.
    pub fn apply_dotted_set(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "backend.mode" => {
                self.backend.mode = value.parse::<BackendMode>()?;
            }
            "local.active_model" => {
                self.local.active_model = value.to_string();
            }
            "local.device" => {
                // Validate the string resolves to a known preference so a
                // typo surfaces at `config set` time rather than during the
                // next dream pass. Persist the value verbatim so the TOML
                // reads naturally (`auto` / `cpu` / `metal` / `cuda`).
                value
                    .parse::<crate::inference::DevicePreference>()
                    .map_err(|e| format!("invalid local.device {value:?}: {e}"))?;
                self.local.device = value.to_string();
            }
            "headless.command" => {
                if shlex::split(value).is_none() {
                    return Err(format!(
                        "invalid headless.command {value:?}: unbalanced quotes or invalid escape"
                    ));
                }
                self.headless.command = value.to_string();
            }
            "headless.timeout_ms" => {
                let parsed: u64 = value.parse().map_err(|e| {
                    format!("invalid headless.timeout_ms {value:?}: expected integer ({e})")
                })?;
                self.headless.timeout_ms = parsed;
            }
            other => {
                return Err(format!(
                    "unknown setting '{other}'. Supported keys: backend.mode, \
                     local.active_model, local.device, headless.command, \
                     headless.timeout_ms"
                ));
            }
        }
        Ok(())
    }
}

/// Single-invocation overrides passed from the CLI layer.
///
/// Each field is `None` when the user didn't pass the corresponding flag —
/// [`Settings::effective`] then falls through to the file-configured value.
#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    pub backend: Option<BackendMode>,
    pub model: Option<String>,
    pub command: Option<String>,
}

/// Resolved configuration for a single invocation. Independent of
/// [`Settings`] so the dream pipeline never accidentally mutates persistent
/// state while rendering overrides.
#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub mode: BackendMode,
    pub active_model: String,
    pub headless_command: String,
    pub headless_timeout_ms: u64,
    /// Raw device preference string (`auto|cpu|metal|cuda`). The candle
    /// backend parses this into a [`crate::inference::DevicePreference`] at
    /// load time. Kept as a string here so settings and CLI override flows
    /// share one storage shape.
    pub device: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp() -> TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn backend_mode_round_trip_through_string() {
        for mode in [
            BackendMode::Headless,
            BackendMode::Local,
            BackendMode::Disabled,
        ] {
            let parsed: BackendMode = mode.as_str().parse().unwrap();
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn backend_mode_rejects_unknown_strings() {
        assert!("llama".parse::<BackendMode>().is_err());
    }

    #[test]
    fn load_auto_detects_claude_when_present() {
        // Force "claude on PATH" regardless of the host machine by going
        // through the probe closure.
        let (settings, _warn) = Settings::auto_detect_with(|b| b == "claude");
        assert_eq!(settings.backend.mode, BackendMode::Headless);
        assert_eq!(settings.headless.command, DEFAULT_CLAUDE_COMMAND);
        assert_eq!(settings.headless.timeout_ms, DEFAULT_HEADLESS_TIMEOUT_MS);
    }

    #[test]
    fn load_falls_back_to_gemini_when_claude_absent() {
        let (settings, _warn) = Settings::auto_detect_with(|b| b == "gemini");
        assert_eq!(settings.backend.mode, BackendMode::Headless);
        assert_eq!(settings.headless.command, DEFAULT_GEMINI_COMMAND);
    }

    #[test]
    fn load_falls_back_to_local_when_nothing_on_path() {
        let (settings, warn) = Settings::auto_detect_with(|_| false);
        assert_eq!(settings.backend.mode, BackendMode::Local);
        assert_eq!(settings.local.active_model, DEFAULT_LOCAL_MODEL);
        // Must emit a warning on the fallback path so a fresh-install user
        // knows why dream isn't condensing yet.
        assert!(warn.is_some());
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tmp();
        let mut original = Settings::default_headless(DEFAULT_CLAUDE_COMMAND);
        original.local.active_model = "tinyllama".to_string();
        original.add_downloaded_model("gemma3");
        original.add_downloaded_model("tinyllama");
        original.save(dir.path()).unwrap();

        let loaded = Settings::load_existing(&dir.path().join(SETTINGS_FILENAME)).unwrap();
        assert_eq!(loaded.backend.mode, BackendMode::Headless);
        assert_eq!(loaded.local.active_model, "tinyllama");
        assert_eq!(
            loaded.local.downloaded_models,
            vec!["gemma3".to_string(), "tinyllama".to_string()]
        );
    }

    #[test]
    fn load_missing_file_writes_default_and_returns_it() {
        let dir = tmp();
        // Force the deterministic fallback path by clearing PATH inside the
        // process — `which` resolves against `$PATH`, and an empty one
        // guarantees nothing is found regardless of the host.
        let old = std::env::var_os("PATH");
        unsafe {
            std::env::set_var("PATH", "/nonexistent-path-that-should-not-exist");
        }

        let s = Settings::load(dir.path()).expect("load ok");
        // Restore PATH before the assertion so a failure doesn't leave the
        // test runtime in a broken state.
        match old {
            Some(v) => unsafe { std::env::set_var("PATH", v) },
            None => unsafe { std::env::remove_var("PATH") },
        }
        assert_eq!(s.backend.mode, BackendMode::Local);
        // File was materialized on disk — second `load` must not re-probe.
        assert!(dir.path().join(SETTINGS_FILENAME).exists());
    }

    #[test]
    fn corrupt_file_surfaces_parse_error() {
        let dir = tmp();
        std::fs::write(dir.path().join(SETTINGS_FILENAME), "not = valid = toml").unwrap();
        let err = Settings::load_existing(&dir.path().join(SETTINGS_FILENAME)).unwrap_err();
        match err {
            SettingsError::ParseToml { .. } => {}
            other => panic!("expected ParseToml, got {other:?}"),
        }
    }

    #[test]
    fn invalid_headless_template_surfaces_error_at_load() {
        let dir = tmp();
        // Unbalanced quote → shlex::split returns None. Surface it cleanly
        // rather than blowing up later during the dream pass.
        let body = r#"
[backend]
mode = "headless"

[local]
active_model = "gemma3"
downloaded_models = []

[headless]
command = "claude -p 'unbalanced"
timeout_ms = 30000
"#;
        std::fs::write(dir.path().join(SETTINGS_FILENAME), body).unwrap();
        let err = Settings::load_existing(&dir.path().join(SETTINGS_FILENAME)).unwrap_err();
        match err {
            SettingsError::InvalidHeadlessCommand { template, .. } => {
                assert_eq!(template, "claude -p 'unbalanced");
            }
            other => panic!("expected InvalidHeadlessCommand, got {other:?}"),
        }
    }

    #[test]
    fn stale_claude_command_is_auto_upgraded_on_load() {
        let dir = tmp();
        // Write a dream.toml with the pre-v1.3.0 default that the probe
        // refuses. Loading must silently upgrade it in place, and also
        // bump the stale 30s timeout — both fields belonged to the
        // pre-v1.4.1 default lineage.
        let body = r#"
[backend]
mode = "headless"

[local]
active_model = "gemma3"
downloaded_models = []

[headless]
command = "claude -p '{prompt}'"
timeout_ms = 30000
"#;
        std::fs::write(dir.path().join(SETTINGS_FILENAME), body).unwrap();

        let loaded = Settings::load(dir.path()).expect("load ok");
        assert_eq!(loaded.headless.command, DEFAULT_CLAUDE_COMMAND);
        assert_eq!(loaded.headless.timeout_ms, DEFAULT_HEADLESS_TIMEOUT_MS);

        // The upgrade persisted — a second load finds the new command on disk
        // and does NOT touch it (idempotent).
        let second = Settings::load(dir.path()).expect("load ok");
        assert_eq!(second.headless.command, DEFAULT_CLAUDE_COMMAND);
        assert_eq!(second.headless.timeout_ms, DEFAULT_HEADLESS_TIMEOUT_MS);

        // On-disk file reflects the upgrade.
        let raw = std::fs::read_to_string(dir.path().join(SETTINGS_FILENAME)).unwrap();
        assert!(
            raw.contains("--model sonnet"),
            "expected upgraded command on disk; got:\n{raw}"
        );
    }

    /// v1.3.0 – v1.4.0 stored the command with tool flags but no `--model`
    /// pin. v1.4.1 must also upgrade that form to the Sonnet-pinned
    /// default, carrying the stale 30s timeout with it.
    #[test]
    fn v140_claude_command_is_upgraded_to_sonnet_pinned_default() {
        let dir = tmp();
        let body = r#"
[backend]
mode = "headless"

[local]
active_model = "gemma3"
downloaded_models = []

[headless]
command = "claude --permission-mode bypassPermissions --allowedTools 'Bash(memory *)' -p '{prompt}'"
timeout_ms = 30000
"#;
        std::fs::write(dir.path().join(SETTINGS_FILENAME), body).unwrap();

        let loaded = Settings::load(dir.path()).expect("load ok");
        assert_eq!(loaded.headless.command, DEFAULT_CLAUDE_COMMAND);
        assert_eq!(loaded.headless.timeout_ms, DEFAULT_HEADLESS_TIMEOUT_MS);
    }

    /// Pre-v1.4.1 Gemini default (no model pin) must upgrade to the Flash
    /// pin, and the stale 30s timeout bumps alongside it.
    #[test]
    fn stale_gemini_command_is_auto_upgraded_on_load() {
        let dir = tmp();
        let body = r#"
[backend]
mode = "headless"

[local]
active_model = "gemma3"
downloaded_models = []

[headless]
command = "gemini -p '{prompt}'"
timeout_ms = 30000
"#;
        std::fs::write(dir.path().join(SETTINGS_FILENAME), body).unwrap();

        let loaded = Settings::load(dir.path()).expect("load ok");
        assert_eq!(loaded.headless.command, DEFAULT_GEMINI_COMMAND);
        assert_eq!(loaded.headless.timeout_ms, DEFAULT_HEADLESS_TIMEOUT_MS);
        let raw = std::fs::read_to_string(dir.path().join(SETTINGS_FILENAME)).unwrap();
        assert!(
            raw.contains("gemini-2.5-flash"),
            "expected Flash-pinned command on disk; got:\n{raw}"
        );
    }

    #[test]
    fn customized_headless_command_is_not_touched() {
        let dir = tmp();
        // User tweaked the template (e.g. picked Opus manually). Command
        // MUST remain byte-identical through load — we never stomp user
        // intent. The timeout must ALSO stay at the user's value: a
        // custom command short-circuits the timeout-upgrade branch so a
        // deliberate 30s setting on a hand-crafted CLI sticks.
        let custom = "claude --model opus -p '{prompt}'";
        let body = format!(
            r#"
[backend]
mode = "headless"

[local]
active_model = "gemma3"
downloaded_models = []

[headless]
command = "{custom}"
timeout_ms = 30000
"#
        );
        std::fs::write(dir.path().join(SETTINGS_FILENAME), body).unwrap();

        let loaded = Settings::load(dir.path()).expect("load ok");
        assert_eq!(loaded.headless.command, custom);
        assert_eq!(
            loaded.headless.timeout_ms, 30_000,
            "custom command must leave the user's timeout alone"
        );
    }

    /// A hand-customized timeout (any value that isn't the exact stale
    /// 30s sentinel) on the CURRENT default command must survive load.
    /// Otherwise users who set `timeout_ms = 45000` to give slow batches
    /// more room lose their tuning on every upgrade.
    #[test]
    fn custom_timeout_on_default_command_is_preserved() {
        let dir = tmp();
        let body = format!(
            r#"
[backend]
mode = "headless"

[local]
active_model = "gemma3"
downloaded_models = []

[headless]
command = "{cmd}"
timeout_ms = 45000
"#,
            cmd = DEFAULT_CLAUDE_COMMAND
        );
        std::fs::write(dir.path().join(SETTINGS_FILENAME), body).unwrap();
        let loaded = Settings::load(dir.path()).expect("load ok");
        assert_eq!(loaded.headless.command, DEFAULT_CLAUDE_COMMAND);
        assert_eq!(loaded.headless.timeout_ms, 45_000);
    }

    #[test]
    fn current_default_command_is_not_re_upgraded() {
        let dir = tmp();
        let s = Settings::default_headless(DEFAULT_CLAUDE_COMMAND);
        s.save(dir.path()).unwrap();
        let before_mtime = std::fs::metadata(dir.path().join(SETTINGS_FILENAME))
            .unwrap()
            .modified()
            .unwrap();

        // Sleep-free mtime check: we only care the loaded value matches.
        let loaded = Settings::load(dir.path()).expect("load ok");
        assert_eq!(loaded.headless.command, DEFAULT_CLAUDE_COMMAND);

        // If the upgrader mis-fired, it would rewrite the file. Compare
        // contents exactly.
        let after_mtime = std::fs::metadata(dir.path().join(SETTINGS_FILENAME))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(
            before_mtime, after_mtime,
            "upgrader rewrote file when no upgrade was needed"
        );
    }

    #[test]
    fn idempotent_save() {
        let dir = tmp();
        let s = Settings::default_headless(DEFAULT_CLAUDE_COMMAND);
        s.save(dir.path()).unwrap();
        let first = std::fs::read_to_string(dir.path().join(SETTINGS_FILENAME)).unwrap();
        s.save(dir.path()).unwrap();
        let second = std::fs::read_to_string(dir.path().join(SETTINGS_FILENAME)).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn add_downloaded_model_dedups_preserving_order() {
        let mut s = Settings::default_local();
        s.add_downloaded_model("gemma3");
        s.add_downloaded_model("tinyllama");
        s.add_downloaded_model("gemma3"); // duplicate — no change
        assert_eq!(
            s.local.downloaded_models,
            vec!["gemma3".to_string(), "tinyllama".to_string()]
        );
    }

    #[test]
    fn remove_downloaded_model_reports_presence() {
        let mut s = Settings::default_local();
        s.add_downloaded_model("gemma3");
        assert!(s.remove_downloaded_model("gemma3"));
        assert!(!s.remove_downloaded_model("gemma3")); // already gone
        assert!(s.local.downloaded_models.is_empty());
    }

    #[test]
    fn apply_dotted_set_mutates_every_key() {
        let mut s = Settings::default_local();
        s.apply_dotted_set("backend.mode", "headless").unwrap();
        assert_eq!(s.backend.mode, BackendMode::Headless);
        s.apply_dotted_set("local.active_model", "tinyllama")
            .unwrap();
        assert_eq!(s.local.active_model, "tinyllama");
        s.apply_dotted_set("headless.command", "claude -p '{prompt}'")
            .unwrap();
        assert_eq!(s.headless.command, "claude -p '{prompt}'");
        s.apply_dotted_set("headless.timeout_ms", "60000").unwrap();
        assert_eq!(s.headless.timeout_ms, 60_000);
    }

    #[test]
    fn apply_dotted_set_rejects_unknown_key() {
        let mut s = Settings::default_local();
        let err = s
            .apply_dotted_set("backend.temperature", "0.7")
            .unwrap_err();
        assert!(err.contains("backend.mode"));
    }

    #[test]
    fn apply_dotted_set_rejects_bad_headless_template() {
        let mut s = Settings::default_local();
        let err = s
            .apply_dotted_set("headless.command", "claude -p 'unbalanced")
            .unwrap_err();
        assert!(err.contains("unbalanced"));
    }

    #[test]
    fn effective_cli_overrides_win_over_file() {
        let s = Settings::default_headless(DEFAULT_CLAUDE_COMMAND);
        let overrides = CliOverrides {
            backend: Some(BackendMode::Local),
            model: Some("tinyllama".to_string()),
            command: None,
        };
        let eff = s.effective(&overrides);
        assert_eq!(eff.mode, BackendMode::Local);
        assert_eq!(eff.active_model, "tinyllama");
        // Command flows through from the file since no override was passed.
        assert_eq!(eff.headless_command, DEFAULT_CLAUDE_COMMAND);
    }

    #[test]
    fn effective_no_overrides_equals_file() {
        let s = Settings::default_headless(DEFAULT_CLAUDE_COMMAND);
        let eff = s.effective(&CliOverrides::default());
        assert_eq!(eff.mode, BackendMode::Headless);
        assert_eq!(eff.active_model, DEFAULT_LOCAL_MODEL);
        assert_eq!(eff.headless_command, DEFAULT_CLAUDE_COMMAND);
    }
}
