//! Project identification from the current working directory.
//!
//! Returns the repository shortname (e.g. `eventic` for
//! `git@github.com:nitecon/eventic.git`) so that memories auto-tagged by the
//! cwd resolver share an ident with the logical, human-written project labels
//! most agents already use (`rithmic`, `traderx`, `agent-tools`, ...). For
//! non-git directories, falls back to the directory basename.
//!
//! Trade-off: two repos with the same basename across different orgs will
//! collide on ident. This is intentional -- the alternative (full host/org/repo
//! path) makes cwd-derived idents look foreign to corpora written by hand.
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

    let canonical = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());
    canonical
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| canonical.to_string_lossy().to_string())
}

/// Resolve the project ident for the current working directory.
pub fn project_ident_from_cwd() -> std::io::Result<String> {
    let cwd = std::env::current_dir()?;
    Ok(project_ident(&cwd))
}

/// Extract the repository shortname from a git remote URL.
///
/// Handles HTTPS, SSH-explicit (`ssh://git@...`), and SSH-shorthand
/// (`git@host:org/repo.git`) forms. Returns the final path segment with
/// any `.git` suffix stripped -- e.g. `agent-memory` for
/// `https://github.com/nitecon/agent-memory.git`.
fn normalize_git_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    // Split on both `/` and `:` so SSH shorthand (`host:org/repo`) and HTTPS
    // paths yield the same final segment.
    let last = trimmed
        .rsplit(['/', ':'])
        .find(|seg| !seg.is_empty())
        .unwrap_or(trimmed);
    last.strip_suffix(".git").unwrap_or(last).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_ssh_shorthand() {
        assert_eq!(
            normalize_git_url("git@github.com:nitecon/agent-memory.git"),
            "agent-memory"
        );
    }

    #[test]
    fn normalize_https() {
        assert_eq!(
            normalize_git_url("https://github.com/nitecon/agent-memory.git"),
            "agent-memory"
        );
    }

    #[test]
    fn normalize_ssh_explicit() {
        assert_eq!(
            normalize_git_url("ssh://git@github.com/nitecon/agent-memory.git"),
            "agent-memory"
        );
    }

    #[test]
    fn normalize_https_with_user() {
        assert_eq!(
            normalize_git_url("https://user@github.com/nitecon/agent-memory.git"),
            "agent-memory"
        );
    }

    #[test]
    fn ssh_and_https_match() {
        assert_eq!(
            normalize_git_url("git@github.com:nitecon/agent-memory.git"),
            normalize_git_url("https://github.com/nitecon/agent-memory.git"),
        );
    }

    #[test]
    fn eventic_shortname() {
        assert_eq!(
            normalize_git_url("git@github.com:nitecon/eventic.git"),
            "eventic"
        );
        assert_eq!(
            normalize_git_url("https://github.com/nitecon/eventic.git"),
            "eventic"
        );
    }

    #[test]
    fn no_dot_git_suffix() {
        assert_eq!(
            normalize_git_url("https://github.com/nitecon/eventic"),
            "eventic"
        );
    }

    #[test]
    fn trailing_slash_is_ignored() {
        assert_eq!(
            normalize_git_url("https://github.com/nitecon/eventic.git/"),
            "eventic"
        );
    }

    #[test]
    fn gitlab_nested_group_uses_final_segment() {
        assert_eq!(
            normalize_git_url("https://gitlab.com/group/subgroup/my-repo.git"),
            "my-repo"
        );
    }
}
