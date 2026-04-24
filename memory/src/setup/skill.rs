//! `memory setup skill` — install an Agent Skill that advertises the `memory`
//! CLI to the model via auto-loaded description.
//!
//! Skills published under `<skills-root>/<name>/SKILL.md` have their
//! frontmatter `description` injected into the session system prompt (~100
//! tokens each) at start. The full body is only loaded on-demand when the
//! model judges the skill relevant, so this file stays out of context until
//! it earns its place.
//!
//! Install targets. As of April 2026 two paths cover every known frontend:
//!
//!   - Claude Code: `~/.claude/skills/<name>/SKILL.md` (tool-native only).
//!   - Cross-agent alias: `~/.agents/skills/<name>/SKILL.md` — read by Gemini
//!     CLI (both `~/.gemini/skills/` and `~/.agents/skills/`, with the latter
//!     taking precedence) and by Codex (user-scope at `$HOME/.agents/skills/`
//!     exclusively — no `~/.codex/skills/` equivalent). Any future frontend
//!     honoring the cross-agent convention picks this up for free.
//!
//! Both honor the same YAML frontmatter + Markdown body, so a single
//! [`SKILL_BODY`] constant is written byte-for-byte to every target. No
//! auto-detection — the user opted in to `memory setup skill` explicitly, so
//! we unconditionally write both files.
//!
//! Legacy path migration. v1.4.0 and v1.4.1 installed the Gemini copy under
//! `~/.gemini/skills/agent-memory/SKILL.md`. That path is now abandoned in
//! favor of the cross-agent alias. Every install and uninstall scrubs the
//! legacy path so users never end up with two copies (or a stale SKILL.md
//! after uninstall).
//!
//! Idempotent: overwrites SKILL.md in place per target. Writes a `.bak`
//! sibling before each destructive overwrite so the user can recover prior
//! content.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Skill body. The frontmatter `description` is deliberately worded to pull
/// the model toward the `memory` CLI for persistent context across sessions.
/// It explicitly names the **Memory First/Last** rule so the skill picker
/// reaches for this surface at task start (`context`) and task end (`store`)
/// rather than treating memory as optional. The body covers the full CLI
/// surface in one screen.
const SKILL_BODY: &str = r#"---
name: agent-memory
description: Persistent hybrid-search memory for AI coding agents via the `memory` CLI — semantic + BM25 search with project (1.5×) and global (1.25×) scope tiers, cross-session recall. Apply the Memory First/Last rule: every task MUST begin with `memory context "<task>"` (pre-task recall, returns both scopes in one call) and end with `memory store` (post-task save) when functionality changed, classifying scope as project or `--scope global` for universal preferences. Use for storing user directives, project decisions, debugging insights, and reusable patterns.
allowed-tools: Bash(memory *)
---

# agent-memory

Persistent hybrid-search memory (BM25 + vector embeddings) shared across all
your Claude Code sessions. The `memory` CLI is the canonical surface — direct
calls are as fast as MCP and avoid persistent server overhead.

## The Memory First/Last Rule (MANDATORY)

Every task must begin with a `context` or `search` call and end with a `store`
call if functionality changed. This is non-negotiable — it is how prior work
informs the current session and how the current session informs future ones.

## Project auto-detection and scope tiers

`store`, `search`, and `context` derive the current project from the cwd's git
remote (reduced to the repo shortname, e.g. `eventic`). Retrieval applies two
independent boosts:

- **Current project** (cwd-derived ident): **1.5×** — local context wins ties.
- **Global scope** (reserved `__global__` ident): **1.25×** — universal user
  preferences surface in every repo, but still lose ties to strong local
  context.
- Other projects (`1.0×`) surface only as prior art, flagged via the `hint`
  field in JSON output.

A single `context` call returns both scopes at once — no second query.

## Global vs project scope

When saving a memory, classify its scope:

- **Project** (default) — specific to this repo, service, or codebase. Stored
  under the cwd-derived ident.
- **Global** (`--scope global`) — universal preference. Stored under the
  reserved sentinel `__global__` so retrieval boosts it across every repo.

Signals for global: "I always / never", "from now on", "I prefer", "whenever
we", "in general" — any phrasing that reads like a personal policy. Signals
for project: "in this repo", "for this service", "here we". When the phrasing
is ambiguous, **MUST ask** the user before storing — don't silently default.

```bash
memory store "User never wants PRs opened unless they explicitly ask" \
  -m feedback --scope global -t "workflow,pr"
```

## Pre-task recall

```bash
memory context "<task description>" -k <limit>   # top-K relevant memories
memory search "<query>" -k <limit>                # hybrid BM25 + vector search
memory get <uuid> [<uuid>...]                     # fetch full content for IDs
```

Default output is `brief` (id + tags + 160-char preview). Pair `search --brief`
with `memory get <id>` for cheap two-stage retrieval — scan, then pull full
content for the handful you actually need.

## Post-task save

```bash
memory store "<content>" -m <type> -t "<tags>"                  # project-scoped (default)
memory store "<content>" -m <type> --scope global -t "<tags>"   # universal preference
# types: user, feedback, project, reference
```

Write audit-ready descriptions — explain the **why**, not just the what.

## Filter, list, inspect

```bash
memory recall -m <type> -t "<tags>" -p "<project>" -k <limit>
memory list -k 50 --project <proj>
memory projects                                   # distinct idents + counts
```

## Move, copy, forget, prune

```bash
memory move --from "<old>" --to "<new>" [--dry-run]   # bulk reassign
memory move --id <uuid> --to "<proj>"                  # single memory
memory copy --from "<old>" --to "<new>" [--dry-run]   # duplicate under new ident
memory forget --id <uuid>                              # remove by ID
memory forget --query "<query>"                        # remove top search hits
memory prune --max-age-days 90 [--dry-run]             # decay stale entries
```

## Updating

```bash
memory update                                          # check + install latest
```

## Memory types

| Type | Purpose |
|------|---------|
| `user` | Facts about the user — role, preferences, expertise |
| `feedback` | How to approach work — corrections and confirmed approaches |
| `project` | Ongoing work context — decisions, deadlines, constraints |
| `reference` | Pointers to external resources — URLs, dashboards, systems |
"#;

/// Legacy install paths that prior releases used. Each entry is a tuple of
/// `(agent_label, relative_path_from_home)` and is scrubbed unconditionally
/// on every install and every `--remove` so users migrating from an older
/// version never end up with a stale copy they didn't ask for.
///
/// Order of entries: oldest first. When adding a new migration, append.
///
/// Rationale for this being a const list: future path changes become a
/// one-line append here plus a single test entry, rather than bespoke
/// cleanup code scattered through the installer.
const LEGACY_SKILL_PATHS: &[(&str, &[&str])] = &[
    // v1.4.0 / v1.4.1 wrote Gemini's copy here. Superseded in v1.4.2 by the
    // cross-agent `~/.agents/skills/` alias (which Gemini also reads, and
    // which Codex reads exclusively).
    ("gemini", &[".gemini", "skills", "agent-memory", "SKILL.md"]),
];

/// One place the skill should be written. The `agent` label is the
/// tool-native name (e.g. `"claude"`, `"shared"`) used in the `<setup>`
/// status lines so a user can tell at a glance which frontend got updated.
#[derive(Debug, Clone)]
pub struct SkillTarget {
    /// Frontend identifier — stable short string used in status output.
    pub agent: &'static str,
    /// Absolute path to the `SKILL.md` file for this frontend.
    pub path: PathBuf,
}

/// Enumerate every install destination for the agent-memory skill.
///
/// Order is stable: Claude first (the original target and what backward-compat
/// callers of [`skill_path`] rely on), then the cross-agent `~/.agents/skills/`
/// alias second. Both paths are always returned — no filesystem probing, no
/// auto-detection of whether the agent is installed. The user opted in to
/// `memory setup skill`, so we write every known target.
///
/// The label `"shared"` is used for the cross-agent path instead of
/// `"agents"` because `<setup agent="agents"/>` reads awkwardly — the
/// double-up of "agent" makes it look like a typo. `"shared"` conveys the
/// relationship (one SKILL.md, multiple frontends read it) without conflict.
/// The label is surfaced to the user via `<setup>` status lines, so it is a
/// soft contract — tests pin the exact string.
pub fn skill_targets() -> Result<Vec<SkillTarget>> {
    let home = dirs::home_dir().context("could not resolve home directory")?;
    Ok(vec![
        SkillTarget {
            agent: "claude",
            path: home
                .join(".claude")
                .join("skills")
                .join("agent-memory")
                .join("SKILL.md"),
        },
        SkillTarget {
            agent: "shared",
            path: home
                .join(".agents")
                .join("skills")
                .join("agent-memory")
                .join("SKILL.md"),
        },
    ])
}

/// Resolve the absolute paths of every legacy install target that predated
/// the current [`skill_targets`] list. Same ordering semantics as
/// [`LEGACY_SKILL_PATHS`] (oldest first).
fn legacy_skill_targets() -> Result<Vec<SkillTarget>> {
    let home = dirs::home_dir().context("could not resolve home directory")?;
    Ok(LEGACY_SKILL_PATHS
        .iter()
        .map(|(agent, parts)| {
            let mut p = home.clone();
            for part in *parts {
                p.push(part);
            }
            SkillTarget { agent, path: p }
        })
        .collect())
}

/// Backward-compat alias: returns the Claude install path. Kept so external
/// callers that hardcoded "the" skill path before the multi-target addition
/// continue to compile unchanged. Internal probing should prefer
/// [`skill_targets`]. Marked `#[allow(dead_code)]` because the menu probe now
/// walks every target directly rather than going through this helper —
/// deleting the function would be a breaking API change for anyone depending
/// on the 1.3.x surface.
#[allow(dead_code)]
pub fn skill_path() -> Result<PathBuf> {
    let mut targets = skill_targets()?;
    // First entry is Claude by construction. Guard with a clear message in
    // case the enumeration is ever reordered by accident.
    let first = targets
        .drain(..)
        .next()
        .context("skill_targets() returned an empty list")?;
    Ok(first.path)
}

/// True iff *any* known target has a SKILL.md file present. OR semantics so a
/// user who wired up only one frontend still gets a "something is there" signal
/// from external callers. The interactive menu implements its own stricter
/// "all targets" check inline since half-installed should still prompt a
/// re-run. Marked `#[allow(dead_code)]` for the same API-stability reason as
/// [`skill_path`].
#[allow(dead_code)]
pub fn is_installed() -> bool {
    match skill_targets() {
        Ok(targets) => targets.iter().any(|t| t.path.exists()),
        Err(_) => false,
    }
}

/// Entry point invoked from `cli.rs` for `memory setup skill`.
///
/// Arguments:
/// - `dry_run` — print what would be written (or removed) for every target
///   without touching disk.
/// - `print` — dump `SKILL_BODY` to stdout and exit (no filesystem IO, no
///   per-target preamble).
/// - `remove` — inverse of install: delete the SKILL.md file at each known
///   target. Missing files are a silent no-op per target (parity with
///   `setup rules --remove` which silently skips files that lack the block).
pub fn run(dry_run: bool, print: bool, remove: bool) -> Result<()> {
    if print {
        print!("{SKILL_BODY}");
        return Ok(());
    }

    let targets = skill_targets().context("could not resolve skill install paths")?;
    let legacy = legacy_skill_targets().context("could not resolve legacy skill paths")?;

    if remove {
        return run_remove(&targets, &legacy, dry_run);
    }

    run_install(&targets, &legacy, dry_run)
}

/// Install path for [`run`]. Iterates every current target; a failure on one
/// does not short-circuit the others — we gather errors and fail at the end
/// so the user sees the full picture. Legacy paths are scrubbed before the
/// new files are written so a partial cleanup failure still surfaces alongside
/// any install failures.
fn run_install(targets: &[SkillTarget], legacy: &[SkillTarget], dry_run: bool) -> Result<()> {
    let mut any_failed = false;

    // Legacy cleanup runs first: removing a stale file has no downstream
    // effect on the fresh install, so any failure here is a standalone signal
    // rather than a blocker for the install itself.
    for t in legacy {
        if let Err(e) = cleanup_legacy_one(t, dry_run) {
            eprintln!(
                "Failed to clean legacy skill for {} at {}: {e:#}",
                t.agent,
                t.path.display()
            );
            any_failed = true;
        }
    }

    for t in targets {
        if dry_run {
            println!(
                r#"<setup status="skill_dry_run" agent="{}" path="{}"/>"#,
                t.agent,
                t.path.display()
            );
            continue;
        }
        match install_one(t) {
            Ok(()) => {}
            Err(e) => {
                eprintln!(
                    "Failed to install skill for {} at {}: {e:#}",
                    t.agent,
                    t.path.display()
                );
                any_failed = true;
            }
        }
    }
    if any_failed {
        anyhow::bail!("one or more skill targets could not be installed");
    }
    Ok(())
}

fn install_one(target: &SkillTarget) -> Result<()> {
    let parent = target
        .path
        .parent()
        .context("skill path has no parent directory")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create skill directory {}", parent.display()))?;

    if let Some(backup) = backup_if_exists(&target.path)? {
        println!(
            r#"<setup status="skill_backup" agent="{}" path="{}"/>"#,
            target.agent,
            backup.display()
        );
    }

    std::fs::write(&target.path, SKILL_BODY)
        .with_context(|| format!("write skill file {}", target.path.display()))?;
    println!(
        r#"<setup status="skill_installed" agent="{}" path="{}"/>"#,
        target.agent,
        target.path.display()
    );
    Ok(())
}

/// Remove path for [`run`]. Mirrors [`run_install`]: every target is visited,
/// missing files emit `skill_absent`, successful removals emit
/// `skill_removed`. Legacy paths are scrubbed unconditionally so
/// `memory setup skill --remove` produces a clean uninstall regardless of
/// which version originally installed the skill. Dry-run prints what *would*
/// happen without touching disk.
fn run_remove(targets: &[SkillTarget], legacy: &[SkillTarget], dry_run: bool) -> Result<()> {
    let mut any_failed = false;

    for t in targets {
        if dry_run {
            let status = if t.path.exists() {
                "skill_dry_run_remove"
            } else {
                "skill_absent"
            };
            println!(
                r#"<setup status="{}" agent="{}" path="{}"/>"#,
                status,
                t.agent,
                t.path.display()
            );
            continue;
        }
        match remove_one(t) {
            Ok(()) => {}
            Err(e) => {
                eprintln!(
                    "Failed to remove skill for {} at {}: {e:#}",
                    t.agent,
                    t.path.display()
                );
                any_failed = true;
            }
        }
    }

    for t in legacy {
        if let Err(e) = cleanup_legacy_one(t, dry_run) {
            eprintln!(
                "Failed to clean legacy skill for {} at {}: {e:#}",
                t.agent,
                t.path.display()
            );
            any_failed = true;
        }
    }

    if any_failed {
        anyhow::bail!("one or more skill targets could not be removed");
    }
    Ok(())
}

fn remove_one(target: &SkillTarget) -> Result<()> {
    if !target.path.exists() {
        println!(
            r#"<setup status="skill_absent" agent="{}" path="{}"/>"#,
            target.agent,
            target.path.display()
        );
        return Ok(());
    }
    std::fs::remove_file(&target.path)
        .with_context(|| format!("remove skill file {}", target.path.display()))?;
    println!(
        r#"<setup status="skill_removed" agent="{}" path="{}"/>"#,
        target.agent,
        target.path.display()
    );
    Ok(())
}

/// Delete a single legacy skill file if present, then prune empty parent
/// directories up through the `.../skills/agent-memory/` and
/// `.../skills/` levels. Prunes stop at the first non-empty parent so we
/// never touch a directory the user populated with unrelated skills.
///
/// Missing files are a silent no-op — this function is called on every
/// install and every uninstall, so most of the time there is nothing to do.
/// Dry-run mode emits a `skill_legacy_dry_run` line per existing file
/// without modifying disk.
fn cleanup_legacy_one(target: &SkillTarget, dry_run: bool) -> Result<()> {
    if !target.path.exists() {
        return Ok(());
    }

    if dry_run {
        println!(
            r#"<setup status="skill_legacy_dry_run" agent="{}" path="{}"/>"#,
            target.agent,
            target.path.display()
        );
        return Ok(());
    }

    std::fs::remove_file(&target.path)
        .with_context(|| format!("remove legacy skill file {}", target.path.display()))?;
    println!(
        r#"<setup status="skill_legacy_removed" agent="{}" path="{}"/>"#,
        target.agent,
        target.path.display()
    );

    // Best-effort prune of the per-skill directory and its `skills/` parent.
    // `remove_dir` only succeeds on an empty directory, so any unrelated
    // content blocks the prune and we silently stop. This is exactly the
    // conservative behavior we want — we only tidy up what we created.
    if let Some(skill_dir) = target.path.parent() {
        let _ = std::fs::remove_dir(skill_dir);
        if let Some(skills_root) = skill_dir.parent() {
            let _ = std::fs::remove_dir(skills_root);
        }
    }
    Ok(())
}

fn backup_if_exists(path: &Path) -> Result<Option<PathBuf>> {
    if !path.exists() {
        return Ok(None);
    }
    let existing = std::fs::read_to_string(path)
        .with_context(|| format!("read existing skill {}", path.display()))?;
    let backup = backup_path(path);
    std::fs::write(&backup, existing)
        .with_context(|| format!("write backup to {}", backup.display()))?;
    Ok(Some(backup))
}

fn backup_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".bak");
    PathBuf::from(s)
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_starts_with_frontmatter_fence() {
        // Skills require the YAML-style `---` fence as the very first bytes.
        assert!(SKILL_BODY.starts_with("---\n"));
    }

    #[test]
    fn frontmatter_has_required_name_field() {
        // Skill discovery is keyed on the name field; must match the install
        // directory (`.../skills/agent-memory/SKILL.md` under each frontend).
        assert!(SKILL_BODY.contains("name: agent-memory"));
    }

    #[test]
    fn frontmatter_description_mentions_memory_first_last_rule() {
        // The description is the always-loaded steering text. It must call
        // out the Memory First/Last rule by name so the model reaches for the
        // skill at task start and end without having to be reminded.
        assert!(
            SKILL_BODY.contains("Memory First/Last"),
            "description must reference the Memory First/Last rule explicitly"
        );
    }

    #[test]
    fn frontmatter_grants_bash_memory_without_prompting() {
        // `allowed-tools` must be present and authorize bare `memory *` calls
        // so the user is not prompted to approve every CLI invocation. Claude
        // Code honors this key; Gemini CLI and Codex silently ignore unknown
        // keys, so a single body stays compatible with all three.
        assert!(SKILL_BODY.contains("allowed-tools: Bash(memory *)"));
    }

    #[test]
    fn body_covers_key_cli_commands() {
        // Body must document the full surface a model needs in-context once
        // the skill is picked. Spot-check the load-bearing subcommands.
        for cmd in [
            "memory context",
            "memory search",
            "memory store",
            "memory get",
            "memory recall",
            "memory list",
            "memory projects",
            "memory move",
            "memory copy",
            "memory forget",
            "memory prune",
            "memory update",
        ] {
            assert!(SKILL_BODY.contains(cmd), "skill body must document `{cmd}`");
        }
    }

    #[test]
    fn body_documents_memory_types() {
        // The types vocabulary (user/feedback/project/reference) is part of
        // the contract for `memory store -m <type>`.
        for ty in ["user", "feedback", "project", "reference"] {
            assert!(
                SKILL_BODY.contains(ty),
                "skill body must document the `{ty}` memory type"
            );
        }
    }

    /// Global-scope surfaces were added to steer the model toward the
    /// reflection-friendly behavior. Lock in the load-bearing substrings
    /// so a future body rewrite can't silently drop them.
    #[test]
    fn body_documents_global_scope_and_boosts() {
        // The `--scope global` CLI flag is the user-visible API — must be
        // documented both as a concept and in a ready-to-copy example.
        assert!(
            SKILL_BODY.contains("--scope global"),
            "skill body must document the --scope global flag"
        );
        // Both boost multipliers must appear so the model understands the
        // relative priority of current-project vs global vs other.
        assert!(
            SKILL_BODY.contains("1.5×"),
            "skill body must reference the 1.5× current-project boost"
        );
        assert!(
            SKILL_BODY.contains("1.25×"),
            "skill body must reference the 1.25× global boost"
        );
        // The mandatory-ask clause is the teeth behind Rule B.
        assert!(
            SKILL_BODY.contains("MUST ask"),
            "skill body must carry the mandatory-ask clause for ambiguous scope"
        );
    }

    #[test]
    fn skill_path_lands_in_claude_skills_agent_memory_dir() {
        let p = skill_path().expect("home dir resolves on test platforms");
        assert!(
            p.ends_with(".claude/skills/agent-memory/SKILL.md"),
            "unexpected skill path: {}",
            p.display()
        );
    }

    /// `skill_targets` must return exactly two tool-neutral destinations in
    /// stable order: Claude first (tool-native), then the cross-agent
    /// `~/.agents/skills/` alias that covers Gemini CLI and Codex. The exact
    /// labels (`"claude"`, `"shared"`) are part of the `<setup>` status
    /// contract — the README documents them — so pin them here too.
    #[test]
    fn skill_targets_returns_claude_and_shared_in_order() {
        let targets = skill_targets().expect("home dir resolves on test platforms");
        assert_eq!(targets.len(), 2, "expected exactly two skill targets");
        assert_eq!(targets[0].agent, "claude");
        assert_eq!(targets[1].agent, "shared");
        assert!(
            targets[0]
                .path
                .ends_with(".claude/skills/agent-memory/SKILL.md"),
            "claude target path: {}",
            targets[0].path.display()
        );
        assert!(
            targets[1]
                .path
                .ends_with(".agents/skills/agent-memory/SKILL.md"),
            "shared target path: {}",
            targets[1].path.display()
        );
        // Both targets must be rooted under $HOME — tests run with whatever
        // `dirs::home_dir()` returns, so compare against that directly.
        let home = dirs::home_dir().expect("home dir resolves");
        for t in &targets {
            assert!(
                t.path.starts_with(&home),
                "target {} not rooted under $HOME ({})",
                t.path.display(),
                home.display()
            );
        }
    }

    /// `~/.gemini/skills/agent-memory/SKILL.md` is a v1.4.0/v1.4.1 legacy
    /// install path. The migration table must list it with the `gemini`
    /// label so cleanup status lines are attributable.
    #[test]
    fn legacy_skill_paths_include_gemini_migration() {
        let legacy = legacy_skill_targets().expect("home dir resolves");
        assert!(
            legacy
                .iter()
                .any(|t| t.agent == "gemini"
                    && t.path.ends_with(".gemini/skills/agent-memory/SKILL.md")),
            "legacy list must contain the .gemini migration entry: {:?}",
            legacy
        );
    }

    #[test]
    fn backup_path_appends_bak() {
        let p = PathBuf::from("/tmp/foo/SKILL.md");
        assert_eq!(backup_path(&p), PathBuf::from("/tmp/foo/SKILL.md.bak"));
    }

    #[test]
    fn backup_if_exists_creates_bak_then_returns_path() {
        let tmp = tempdir_in_target();
        let target = tmp.join("SKILL.md");
        std::fs::write(&target, "old contents").unwrap();
        let backup = backup_if_exists(&target)
            .expect("backup_if_exists should succeed")
            .expect("backup path returned because target existed");
        assert_eq!(backup, backup_path(&target));
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), "old contents");
    }

    #[test]
    fn backup_if_exists_returns_none_when_missing() {
        let tmp = tempdir_in_target();
        let target = tmp.join("missing.md");
        let backup = backup_if_exists(&target).expect("missing target is not an error");
        assert!(backup.is_none());
    }

    /// Install → verify both targets got SKILL.md with identical byte content.
    /// Uses `install_one` directly with synthesized `SkillTarget`s rooted
    /// under a tempdir so the test doesn't depend on `HOME` being writable
    /// or on the `run()` dispatch layer.
    #[test]
    fn install_writes_identical_skill_to_every_target() {
        let tmp = tempdir_in_target();
        let claude_target = SkillTarget {
            agent: "claude",
            path: tmp
                .join(".claude")
                .join("skills")
                .join("agent-memory")
                .join("SKILL.md"),
        };
        let shared_target = SkillTarget {
            agent: "shared",
            path: tmp
                .join(".agents")
                .join("skills")
                .join("agent-memory")
                .join("SKILL.md"),
        };
        install_one(&claude_target).expect("claude install");
        install_one(&shared_target).expect("shared install");

        let claude_body = std::fs::read_to_string(&claude_target.path).unwrap();
        let shared_body = std::fs::read_to_string(&shared_target.path).unwrap();
        assert_eq!(
            claude_body, shared_body,
            "claude and shared SKILL.md contents should match byte-for-byte"
        );
        assert_eq!(claude_body, SKILL_BODY);
    }

    /// A second install over existing content must write a `.bak` sidecar
    /// and leave the final SKILL.md body equal to the shared constant.
    #[test]
    fn install_twice_is_idempotent_with_backup() {
        let tmp = tempdir_in_target();
        let target = SkillTarget {
            agent: "claude",
            path: tmp.join("skills").join("agent-memory").join("SKILL.md"),
        };
        install_one(&target).expect("first install");
        install_one(&target).expect("second install");
        let backup = backup_path(&target.path);
        assert!(
            backup.exists(),
            "backup file should exist after second install"
        );
        assert_eq!(std::fs::read_to_string(&target.path).unwrap(), SKILL_BODY);
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), SKILL_BODY);
    }

    /// `remove_one` deletes the file cleanly and is a no-op on a missing
    /// target (mirrors the silent-skip contract for `--remove`).
    #[test]
    fn remove_one_deletes_then_noop_on_missing() {
        let tmp = tempdir_in_target();
        let target = SkillTarget {
            agent: "claude",
            path: tmp.join("skills").join("agent-memory").join("SKILL.md"),
        };
        install_one(&target).expect("install");
        assert!(target.path.exists());
        remove_one(&target).expect("first remove");
        assert!(!target.path.exists());
        // Second remove on missing file must not error.
        remove_one(&target).expect("second remove is a no-op");
    }

    /// `cleanup_legacy_one` removes the file AND prunes the now-empty
    /// per-skill and `skills/` parent directories so a subsequent `ls`
    /// shows no trace of the old install layout.
    #[test]
    fn cleanup_legacy_one_removes_file_and_empty_parents() {
        let tmp = tempdir_in_target();
        let path = tmp
            .join(".gemini")
            .join("skills")
            .join("agent-memory")
            .join("SKILL.md");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "stale body").unwrap();

        let target = SkillTarget {
            agent: "gemini",
            path: path.clone(),
        };
        cleanup_legacy_one(&target, false).expect("cleanup should succeed");

        assert!(!path.exists(), "stale SKILL.md should be gone");
        let per_skill_dir = path.parent().unwrap();
        assert!(
            !per_skill_dir.exists(),
            "empty agent-memory/ parent should be pruned"
        );
        let skills_root = per_skill_dir.parent().unwrap();
        assert!(
            !skills_root.exists(),
            "empty skills/ parent should be pruned"
        );
        // The tool home (`.gemini/`) itself must survive — that's the
        // user's settings directory, not ours to delete.
        assert!(
            skills_root.parent().unwrap().exists(),
            "tool home directory must not be pruned"
        );
    }

    /// When the `skills/` sibling contains other skills, the prune must
    /// stop at the first non-empty parent — we only tidy up empties we
    /// created, never directories the user populated.
    #[test]
    fn cleanup_legacy_one_stops_prune_at_populated_parent() {
        let tmp = tempdir_in_target();
        let stale = tmp
            .join(".gemini")
            .join("skills")
            .join("agent-memory")
            .join("SKILL.md");
        let sibling = tmp
            .join(".gemini")
            .join("skills")
            .join("other-skill")
            .join("SKILL.md");
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::create_dir_all(sibling.parent().unwrap()).unwrap();
        std::fs::write(&stale, "stale").unwrap();
        std::fs::write(&sibling, "unrelated user skill").unwrap();

        let target = SkillTarget {
            agent: "gemini",
            path: stale.clone(),
        };
        cleanup_legacy_one(&target, false).expect("cleanup should succeed");

        assert!(!stale.exists(), "stale SKILL.md should be gone");
        assert!(
            !stale.parent().unwrap().exists(),
            "empty agent-memory/ parent should be pruned"
        );
        assert!(
            sibling.exists(),
            "unrelated user skill must survive the prune"
        );
        assert!(
            sibling.parent().unwrap().parent().unwrap().exists(),
            "populated skills/ root must survive"
        );
    }

    /// `cleanup_legacy_one` on a missing file must be a silent no-op — it's
    /// called on every install and every uninstall, so most invocations
    /// find nothing to do.
    #[test]
    fn cleanup_legacy_one_noop_when_missing() {
        let tmp = tempdir_in_target();
        let target = SkillTarget {
            agent: "gemini",
            path: tmp
                .join(".gemini")
                .join("skills")
                .join("agent-memory")
                .join("SKILL.md"),
        };
        cleanup_legacy_one(&target, false).expect("missing file is not an error");
    }

    /// End-to-end: run_install against a tempdir pre-seeded with a stale
    /// legacy SKILL.md must delete the stale file and install the two
    /// current targets.
    #[test]
    fn run_install_cleans_legacy_and_writes_current_targets() {
        let tmp = tempdir_in_target();
        let legacy_path = tmp
            .join(".gemini")
            .join("skills")
            .join("agent-memory")
            .join("SKILL.md");
        std::fs::create_dir_all(legacy_path.parent().unwrap()).unwrap();
        std::fs::write(&legacy_path, "stale body").unwrap();

        let targets = vec![
            SkillTarget {
                agent: "claude",
                path: tmp
                    .join(".claude")
                    .join("skills")
                    .join("agent-memory")
                    .join("SKILL.md"),
            },
            SkillTarget {
                agent: "shared",
                path: tmp
                    .join(".agents")
                    .join("skills")
                    .join("agent-memory")
                    .join("SKILL.md"),
            },
        ];
        let legacy = vec![SkillTarget {
            agent: "gemini",
            path: legacy_path.clone(),
        }];

        run_install(&targets, &legacy, false).expect("install should succeed");

        assert!(!legacy_path.exists(), "legacy path should be cleaned");
        for t in &targets {
            assert_eq!(
                std::fs::read_to_string(&t.path).unwrap(),
                SKILL_BODY,
                "current target {} should have fresh body",
                t.path.display()
            );
        }
    }

    /// `--remove` semantics: even when only the legacy path exists (a user
    /// who installed v1.4.1 but never upgraded), uninstall wipes it cleanly.
    #[test]
    fn run_remove_cleans_legacy_even_when_current_absent() {
        let tmp = tempdir_in_target();
        let legacy_path = tmp
            .join(".gemini")
            .join("skills")
            .join("agent-memory")
            .join("SKILL.md");
        std::fs::create_dir_all(legacy_path.parent().unwrap()).unwrap();
        std::fs::write(&legacy_path, "stale body").unwrap();

        let targets = vec![
            SkillTarget {
                agent: "claude",
                path: tmp
                    .join(".claude")
                    .join("skills")
                    .join("agent-memory")
                    .join("SKILL.md"),
            },
            SkillTarget {
                agent: "shared",
                path: tmp
                    .join(".agents")
                    .join("skills")
                    .join("agent-memory")
                    .join("SKILL.md"),
            },
        ];
        let legacy = vec![SkillTarget {
            agent: "gemini",
            path: legacy_path.clone(),
        }];

        run_remove(&targets, &legacy, false).expect("remove should succeed");
        assert!(
            !legacy_path.exists(),
            "legacy path must be removed during --remove even when current targets are absent"
        );
    }

    /// Minimal isolated tempdir under `target/` so tests don't depend on the
    /// `tempfile` crate (not in this project's deps). Each test gets a unique
    /// directory based on a UUID v4 — the existing `uuid` dep is sufficient.
    fn tempdir_in_target() -> PathBuf {
        let base =
            std::env::temp_dir().join(format!("agent-memory-skill-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        base
    }
}
