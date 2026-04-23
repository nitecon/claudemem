//! Auto-update support for the agent-memory binary.
//!
//! Checks GitHub releases for newer versions and performs in-place binary
//! replacement. The auto-update path (`auto_update`) is rate-limited to once
//! per hour and never panics or aborts the process on failure. The manual
//! path (`manual_update`) surfaces errors normally.

use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::MemoryError;

/// GitHub repository in `owner/repo` format.
const REPO: &str = "nitecon/agent-memory";

/// Compiled-in version.
///
/// Prefers `AGENT_MEMORY_VERSION` (set by `build.rs` from the release tag in
/// CI, e.g. `1.0.3`) and falls back to `CARGO_PKG_VERSION` for local
/// development builds. This lets release binaries report the real tagged
/// version even when `Cargo.toml` has not been bumped, which is required for
/// the auto-updater to stop looping once it reaches the latest release.
const CURRENT_VERSION: &str = env!("AGENT_MEMORY_VERSION");

/// Minimum interval (in seconds) between automatic update checks.
const CHECK_INTERVAL_SECS: u64 = 3600;

// ---------------------------------------------------------------------------
// Rate-limiting helpers
// ---------------------------------------------------------------------------

/// Returns the path to the timestamp file used for rate-limiting checks.
fn check_file_path(data_dir: &Path) -> PathBuf {
    data_dir.join(".last_update_check")
}

/// Returns `true` if enough time has elapsed since the last check (or if the
/// check file does not exist / cannot be read).
fn should_check(data_dir: &Path) -> bool {
    let path = check_file_path(data_dir);
    let contents = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return true,
    };
    let last: u64 = match contents.trim().parse() {
        Ok(v) => v,
        Err(_) => return true,
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    now.saturating_sub(last) >= CHECK_INTERVAL_SECS
}

/// Persists the current unix timestamp so subsequent calls to `should_check`
/// respect the rate limit.
fn mark_checked(data_dir: &Path) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let _ = fs::write(check_file_path(data_dir), now.to_string());
}

// ---------------------------------------------------------------------------
// GitHub release helpers
// ---------------------------------------------------------------------------

/// Builds a shared blocking reqwest client with appropriate timeouts and
/// User-Agent header.
fn http_client() -> Result<reqwest::blocking::Client, MemoryError> {
    reqwest::blocking::Client::builder()
        .user_agent("agent-memory-updater")
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| MemoryError::Update(format!("failed to build HTTP client: {e}")))
}

/// Fetches the latest release from GitHub.
///
/// Returns `(version_without_v, full_tag)` — e.g. `("1.2.3", "v1.2.3")`.
fn get_latest_release() -> Result<(String, String), MemoryError> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let client = http_client()?;
    let resp = client
        .get(&url)
        .send()
        .map_err(|e| MemoryError::Update(format!("GitHub API request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(MemoryError::Update(format!(
            "GitHub API returned status {}",
            resp.status()
        )));
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| MemoryError::Update(format!("failed to parse GitHub response: {e}")))?;

    let tag = body["tag_name"]
        .as_str()
        .ok_or_else(|| MemoryError::Update("missing tag_name in release".into()))?
        .to_string();

    let version = tag.strip_prefix('v').unwrap_or(&tag).to_string();
    Ok((version, tag))
}

/// Returns `true` when `latest` is strictly greater than `CURRENT_VERSION`
/// according to semver.
fn is_newer(latest: &str) -> bool {
    let Ok(latest_ver) = semver::Version::parse(latest) else {
        return false;
    };
    let Ok(current_ver) = semver::Version::parse(CURRENT_VERSION) else {
        return false;
    };
    latest_ver > current_ver
}

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------

/// Per-platform suffix for the combined release archive. The full asset
/// name is `agent-memory-<suffix>` (see [`get_asset_name`]).
fn get_platform_suffix() -> Result<&'static str, MemoryError> {
    let suffix = match (
        cfg!(target_os = "linux"),
        cfg!(target_os = "macos"),
        cfg!(target_os = "windows"),
    ) {
        (true, _, _) => {
            if cfg!(target_arch = "x86_64") {
                "linux-x86_64.tar.gz"
            } else if cfg!(target_arch = "aarch64") {
                "linux-aarch64.tar.gz"
            } else {
                return Err(MemoryError::Update("unsupported Linux architecture".into()));
            }
        }
        (_, true, _) => {
            if cfg!(target_arch = "x86_64") {
                "macos-x86_64.tar.gz"
            } else if cfg!(target_arch = "aarch64") {
                "macos-aarch64.tar.gz"
            } else {
                return Err(MemoryError::Update("unsupported macOS architecture".into()));
            }
        }
        (_, _, true) => {
            if cfg!(target_arch = "x86_64") {
                "windows-x86_64.zip"
            } else {
                return Err(MemoryError::Update(
                    "unsupported Windows architecture".into(),
                ));
            }
        }
        _ => return Err(MemoryError::Update("unsupported operating system".into())),
    };
    Ok(suffix)
}

/// Determines the correct release asset filename for the running platform.
/// Combined archives include both `memory` and `memory-dream` binaries:
/// `agent-memory-<platform>.{tar.gz|zip}`. No tag embedded — older
/// binaries must be able to resolve this name for any future tag so
/// `memory update` works across version jumps.
fn get_asset_name() -> Result<String, MemoryError> {
    let suffix = get_platform_suffix()?;
    Ok(format!("agent-memory-{suffix}"))
}

/// Resolves the real path of the currently running executable, following
/// symlinks.
fn get_current_exe_path() -> Result<PathBuf, MemoryError> {
    let exe = std::env::current_exe()
        .map_err(|e| MemoryError::Update(format!("cannot determine current exe: {e}")))?;
    exe.canonicalize()
        .map_err(|e| MemoryError::Update(format!("cannot canonicalize exe path: {e}")))
}

// ---------------------------------------------------------------------------
// Download & replace
// ---------------------------------------------------------------------------

/// Names of every binary packaged in the combined release archive.
///
/// Release 2 ships both `memory` and `memory-dream` together per platform
/// (one archive, two binaries). The updater atomically swaps each binary
/// next to `memory`'s own install location so users who never invoked
/// `memory-dream` still get it force-bundled on the next `memory update`.
fn bundled_binary_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["memory.exe", "memory-dream.exe"]
    } else {
        &["memory", "memory-dream"]
    }
}

/// Downloads the combined release archive for `tag`, extracts every bundled
/// binary, and atomically replaces each next to the currently running
/// `memory` executable. The running `memory` binary is replaced LAST so a
/// crash mid-update still leaves a functional pair.
///
/// Force-bundle semantics: if `memory-dream` is not currently installed
/// alongside `memory`, this function creates it anyway. Users who never
/// invoke the compactor pay ~28MB of disk but gain zero cognitive overhead;
/// install and updater logic become symmetric per the Release 2 plan.
fn download_and_replace(tag: &str) -> Result<(), MemoryError> {
    let asset_name = get_asset_name()?;
    let url = format!("https://github.com/{REPO}/releases/download/{tag}/{asset_name}");

    let client = http_client()?;
    let resp = client
        .get(&url)
        .send()
        .map_err(|e| MemoryError::Update(format!("download failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(MemoryError::Update(format!(
            "download returned status {}",
            resp.status()
        )));
    }

    let bytes = resp
        .bytes()
        .map_err(|e| MemoryError::Update(format!("failed to read download body: {e}")))?;

    // The install dir is derived from the running memory executable's
    // location so side-by-side binaries land in the same directory as the
    // one the user installed originally.
    let memory_exe = get_current_exe_path()?;
    let install_dir = memory_exe
        .parent()
        .ok_or_else(|| MemoryError::Update("cannot locate install directory".into()))?
        .to_path_buf();

    // Pull every bundled binary out of the archive. We defer the actual
    // replace() calls until all extractions succeed — avoids leaving the
    // install dir in a half-updated state if extraction of the later
    // binary fails.
    let mut staged: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    for bin_name in bundled_binary_names() {
        let data = if asset_name.ends_with(".tar.gz") {
            extract_from_tar_gz(&bytes, bin_name)
        } else if asset_name.ends_with(".zip") {
            extract_from_zip(&bytes, bin_name)
        } else {
            return Err(MemoryError::Update("unknown archive format".into()));
        };

        match data {
            Ok(bytes) => staged.push((install_dir.join(bin_name), bytes)),
            Err(e) => {
                // `memory-dream` has been in the archive since the Release 2
                // cutover. If a pre-R2 archive tag ever gets served by
                // mistake it won't contain `memory-dream`; treat that as
                // a soft-failure for the companion binary — still update
                // `memory` itself, just warn about the missing companion.
                if *bin_name != memory_binary_name() {
                    eprintln!(
                        "[WARN] {bin_name} not found in {asset_name}; skipping \
                         companion binary update: {e}"
                    );
                    continue;
                }
                return Err(e);
            }
        }
    }

    // Replace in order: every companion first, memory itself last. That
    // way a crash during replacement leaves a mismatched-version pair at
    // worst, never an empty install dir.
    staged.sort_by_key(|(path, _)| {
        path.file_name() == Some(std::ffi::OsStr::new(memory_binary_name()))
    });
    for (path, data) in &staged {
        replace_binary(path, data)?;
    }

    Ok(())
}

/// Name of the main `memory` binary on the current platform.
fn memory_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "memory.exe"
    } else {
        "memory"
    }
}

/// Extracts the target binary from a `.tar.gz` archive held in memory.
fn extract_from_tar_gz(data: &[u8], binary_name: &str) -> Result<Vec<u8>, MemoryError> {
    let decoder = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive
        .entries()
        .map_err(|e| MemoryError::Update(format!("tar read error: {e}")))?
    {
        let mut entry = entry.map_err(|e| MemoryError::Update(format!("tar entry error: {e}")))?;
        let path = entry
            .path()
            .map_err(|e| MemoryError::Update(format!("tar path error: {e}")))?;

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        if file_name == binary_name {
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .map_err(|e| MemoryError::Update(format!("tar extract error: {e}")))?;
            return Ok(buf);
        }
    }
    Err(MemoryError::Update(format!(
        "binary '{binary_name}' not found in archive"
    )))
}

/// Extracts the target binary from a `.zip` archive held in memory.
fn extract_from_zip(data: &[u8], binary_name: &str) -> Result<Vec<u8>, MemoryError> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| MemoryError::Update(format!("zip open error: {e}")))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| MemoryError::Update(format!("zip entry error: {e}")))?;

        let name = file.name().to_string();
        let file_name = Path::new(&name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        if file_name == binary_name {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .map_err(|e| MemoryError::Update(format!("zip extract error: {e}")))?;
            return Ok(buf);
        }
    }
    Err(MemoryError::Update(format!(
        "binary '{binary_name}' not found in zip archive"
    )))
}

/// Atomically replaces (or creates) the binary at `exe_path` with
/// `new_binary`. Handles both in-place upgrades and force-bundle cases
/// where the target doesn't exist yet (Release 2: `memory-dream` may not
/// be installed when a user runs `memory update` for the first time).
///
/// - Unix: writes to `{exe_path}.new`, sets executable permissions, then
///   renames over the target. Atomic on the same filesystem; works when
///   `exe_path` is missing because `fs::rename` handles both overwrite
///   and create-new.
/// - Windows: when the target exists, renames it to `.old` first so the
///   currently-running executable isn't held-open during the write; when
///   the target doesn't exist (force-bundle), the rename step is skipped.
fn replace_binary(exe_path: &Path, new_binary: &[u8]) -> Result<(), MemoryError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let new_path = exe_path.with_extension("new");
        fs::write(&new_path, new_binary)
            .map_err(|e| MemoryError::Update(format!("failed to write new binary: {e}")))?;
        fs::set_permissions(&new_path, fs::Permissions::from_mode(0o755))
            .map_err(|e| MemoryError::Update(format!("failed to set permissions: {e}")))?;
        fs::rename(&new_path, exe_path)
            .map_err(|e| MemoryError::Update(format!("failed to replace binary: {e}")))?;
    }

    #[cfg(windows)]
    {
        // Only rename the current binary out of the way if it actually
        // exists — for the Release 2 force-bundle case `memory-dream.exe`
        // may be absent on first upgrade.
        let old_path = exe_path.with_extension("old");
        let had_prior = exe_path.exists();
        if had_prior {
            fs::rename(exe_path, &old_path).map_err(|e| {
                MemoryError::Update(format!("failed to rename current binary: {e}"))
            })?;
        }
        if let Err(e) = fs::write(exe_path, new_binary) {
            // Attempt to restore the original on failure.
            if had_prior {
                let _ = fs::rename(&old_path, exe_path);
            }
            return Err(MemoryError::Update(format!(
                "failed to write new binary: {e}"
            )));
        }
        if had_prior {
            let _ = fs::remove_file(&old_path);
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        return Err(MemoryError::Update(
            "binary replacement not supported on this platform".into(),
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Automatic update check intended to be called on every CLI invocation.
///
/// This function is rate-limited to at most once per `CHECK_INTERVAL_SECS`
/// and will **never** panic or return an error. All failures are logged to
/// stderr so they do not interfere with normal JSON output on stdout.
pub fn auto_update(data_dir: &Path) {
    // Respect opt-out via environment variable.
    if std::env::var("AGENT_MEMORY_NO_UPDATE").unwrap_or_default() == "1" {
        return;
    }

    if !should_check(data_dir) {
        return;
    }

    // Mark early so a failure does not cause repeated attempts.
    mark_checked(data_dir);

    let (latest_version, tag) = match get_latest_release() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[WARN] Update check failed: {e}");
            return;
        }
    };

    if !is_newer(&latest_version) {
        return;
    }

    eprintln!("[INFO] Updating memory from {CURRENT_VERSION} to {latest_version}...");

    if let Err(e) = download_and_replace(&tag) {
        eprintln!("[WARN] Auto-update failed: {e}");
        return;
    }

    eprintln!("[INFO] Update complete. Changes will take effect on next invocation.");
}

/// Manual update triggered by the `update` CLI sub-command.
///
/// Unlike `auto_update`, this function surfaces errors to the caller so they
/// can be reported with full context.
pub fn manual_update() -> Result<(), MemoryError> {
    let (latest_version, tag) = get_latest_release()?;

    if !is_newer(&latest_version) {
        println!("Already up to date (v{CURRENT_VERSION})");
        return Ok(());
    }

    eprintln!("[INFO] Updating memory from {CURRENT_VERSION} to {latest_version}...");
    download_and_replace(&tag)?;
    eprintln!("[INFO] Update complete. Changes will take effect on next invocation.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_newer_greater() {
        assert!(is_newer("99.0.0"));
    }

    #[test]
    fn test_is_newer_same() {
        assert!(!is_newer(CURRENT_VERSION));
    }

    #[test]
    fn test_is_newer_older() {
        assert!(!is_newer("0.0.1"));
    }

    #[test]
    fn test_is_newer_invalid() {
        assert!(!is_newer("not-a-version"));
    }

    #[test]
    fn test_should_check_missing_file() {
        let tmp = std::env::temp_dir().join("agent-memory-test-update-check");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        assert!(should_check(&tmp));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_mark_checked_then_not_due() {
        let tmp = std::env::temp_dir().join("agent-memory-test-mark-check");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        mark_checked(&tmp);
        assert!(!should_check(&tmp));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_get_asset_name_has_no_tag() {
        // Asset name must NOT embed the tag so older binaries can resolve
        // the URL for any future release.
        let name = get_asset_name().unwrap();
        assert!(name.starts_with("agent-memory-"), "got: {name}");
        assert!(!name.contains("v1."), "tag leaked into asset name: {name}");
    }

    #[test]
    fn test_get_platform_suffix_is_nonempty() {
        // Should not error on the platform running the test suite.
        let suffix = get_platform_suffix().unwrap();
        assert!(!suffix.is_empty());
        assert!(suffix.ends_with(".tar.gz") || suffix.ends_with(".zip"));
    }

    #[test]
    fn test_bundled_binary_names_includes_both() {
        // Force-bundle contract: every release ships both memory and
        // memory-dream; the updater must know to extract both.
        let names = bundled_binary_names();
        assert!(names.contains(&memory_binary_name()));
        let companion = if cfg!(target_os = "windows") {
            "memory-dream.exe"
        } else {
            "memory-dream"
        };
        assert!(names.contains(&companion));
    }
}
