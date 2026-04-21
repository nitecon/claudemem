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

/// Determines the correct release asset filename for the running platform.
fn get_asset_name() -> Result<String, MemoryError> {
    let name = match (cfg!(target_os = "linux"), cfg!(target_os = "macos"), cfg!(target_os = "windows")) {
        (true, _, _) => {
            if cfg!(target_arch = "x86_64") {
                "agent-memory-linux-x86_64.tar.gz"
            } else if cfg!(target_arch = "aarch64") {
                "agent-memory-linux-aarch64.tar.gz"
            } else {
                return Err(MemoryError::Update("unsupported Linux architecture".into()));
            }
        }
        (_, true, _) => {
            if cfg!(target_arch = "x86_64") {
                "agent-memory-macos-x86_64.tar.gz"
            } else if cfg!(target_arch = "aarch64") {
                "agent-memory-macos-aarch64.tar.gz"
            } else {
                return Err(MemoryError::Update("unsupported macOS architecture".into()));
            }
        }
        (_, _, true) => {
            if cfg!(target_arch = "x86_64") {
                "agent-memory-windows-x86_64.zip"
            } else {
                return Err(MemoryError::Update("unsupported Windows architecture".into()));
            }
        }
        _ => return Err(MemoryError::Update("unsupported operating system".into())),
    };
    Ok(name.to_string())
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

/// Downloads the release asset for `tag`, extracts the binary, and replaces
/// the currently running executable in-place.
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

    // Determine the binary name we are looking for inside the archive.
    let binary_name = if cfg!(target_os = "windows") {
        "memory.exe"
    } else {
        "memory"
    };

    let new_binary = if asset_name.ends_with(".tar.gz") {
        extract_from_tar_gz(&bytes, binary_name)?
    } else if asset_name.ends_with(".zip") {
        extract_from_zip(&bytes, binary_name)?
    } else {
        return Err(MemoryError::Update("unknown archive format".into()));
    };

    let exe_path = get_current_exe_path()?;
    replace_binary(&exe_path, &new_binary)?;

    Ok(())
}

/// Extracts the target binary from a `.tar.gz` archive held in memory.
fn extract_from_tar_gz(data: &[u8], binary_name: &str) -> Result<Vec<u8>, MemoryError> {
    let decoder = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive
        .entries()
        .map_err(|e| MemoryError::Update(format!("tar read error: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| MemoryError::Update(format!("tar entry error: {e}")))?;
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

/// Atomically replaces the binary at `exe_path` with `new_binary`.
///
/// - Unix: writes to `{exe_path}.new`, sets permissions, then renames over
///   the original (atomic on the same filesystem).
/// - Windows: renames current to `.old`, writes new binary, then attempts to
///   remove `.old`.
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
        let old_path = exe_path.with_extension("old");
        fs::rename(exe_path, &old_path)
            .map_err(|e| MemoryError::Update(format!("failed to rename current binary: {e}")))?;
        if let Err(e) = fs::write(exe_path, new_binary) {
            // Attempt to restore the original on failure.
            let _ = fs::rename(&old_path, exe_path);
            return Err(MemoryError::Update(format!(
                "failed to write new binary: {e}"
            )));
        }
        let _ = fs::remove_file(&old_path);
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
    fn test_get_asset_name() {
        // Should not error on the platform running the test suite.
        let name = get_asset_name().unwrap();
        assert!(name.starts_with("agent-memory-"));
    }
}
