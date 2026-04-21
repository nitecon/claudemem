//! Project identification from the current working directory.
//!
//! Mirrors `agent-core::storage::resolve_project_identifier` from
//! github.com/nitecon/agent-tools so both systems agree on a single project
//! identity string: normalized git remote origin URL, or the canonical absolute
//! path of the project root as a fallback.
//!
//! Used by the memory system to (a) auto-tag stored memories with the current
//! project and (b) boost the relevance of current-project memories at
//! search/context time while still surfacing cross-project results.

use std::path::Path;

/// Resolve a stable project identifier from a project root path.
pub fn project_ident(project_root: &Path) -> String {
    if let Ok(output) = std::process::Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(project_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
    {
        if output.status.success() {
            let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !url.is_empty() {
                return normalize_git_url(&url);
            }
        }
    }

    project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf())
        .to_string_lossy()
        .to_string()
}

/// Resolve the project ident for the current working directory.
pub fn project_ident_from_cwd() -> std::io::Result<String> {
    let cwd = std::env::current_dir()?;
    Ok(project_ident(&cwd))
}

/// Normalize a git remote URL so SSH and HTTPS URLs for the same repo
/// produce the same identifier.
fn normalize_git_url(url: &str) -> String {
    let mut s = url.to_string();

    for proto in &["https://", "http://", "ssh://"] {
        if let Some(rest) = s.strip_prefix(proto) {
            s = rest.to_string();
            break;
        }
    }

    if let Some(at_pos) = s.find('@') {
        let slash_pos = s.find('/').unwrap_or(usize::MAX);
        let colon_pos = s.find(':').unwrap_or(usize::MAX);
        if at_pos < slash_pos && at_pos < colon_pos {
            s = s[at_pos + 1..].to_string();
        }
    }

    if let Some(colon_pos) = s.find(':') {
        let slash_pos = s.find('/').unwrap_or(usize::MAX);
        if colon_pos < slash_pos {
            s = format!("{}/{}", &s[..colon_pos], &s[colon_pos + 1..]);
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_ssh_shorthand() {
        assert_eq!(
            normalize_git_url("git@github.com:nitecon/agent-memory.git"),
            "github.com/nitecon/agent-memory.git"
        );
    }

    #[test]
    fn normalize_https() {
        assert_eq!(
            normalize_git_url("https://github.com/nitecon/agent-memory.git"),
            "github.com/nitecon/agent-memory.git"
        );
    }

    #[test]
    fn normalize_ssh_explicit() {
        assert_eq!(
            normalize_git_url("ssh://git@github.com/nitecon/agent-memory.git"),
            "github.com/nitecon/agent-memory.git"
        );
    }

    #[test]
    fn normalize_https_with_user() {
        assert_eq!(
            normalize_git_url("https://user@github.com/nitecon/agent-memory.git"),
            "github.com/nitecon/agent-memory.git"
        );
    }

    #[test]
    fn ssh_and_https_match() {
        assert_eq!(
            normalize_git_url("git@github.com:nitecon/agent-memory.git"),
            normalize_git_url("https://github.com/nitecon/agent-memory.git"),
        );
    }
}
