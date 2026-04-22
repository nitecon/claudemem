//! `memory setup skill` — install a Claude Code Agent Skill that advertises
//! the `memory` CLI to the model via auto-loaded description.
//!
//! Skills published under `~/.claude/skills/<name>/SKILL.md` have their
//! frontmatter `description` injected into the session system prompt (~100
//! tokens each) at start. The full body is only loaded on-demand when the
//! model judges the skill relevant, so this file stays out of context until
//! it earns its place.
//!
//! Idempotent: overwrites SKILL.md in place. Writes a `.bak` sibling before
//! the first destructive overwrite so the user can recover prior content.

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
description: Persistent hybrid-search memory for AI coding agents via the `memory` CLI — semantic + BM25 search, current-project boost, cross-session recall. Apply the Memory First/Last rule: every task MUST begin with `memory context "<task>"` (pre-task recall) and end with `memory store` (post-task save) when functionality changed. Use for storing user preferences, project decisions, debugging insights, and reusable patterns; recalling prior context before reading code; and migrating/pruning memories across project idents.
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

## Project auto-detection

`store`, `search`, and `context` derive the current project from the cwd's git
remote (reduced to the repo shortname, e.g. `eventic`). Memories tagged with
the current project receive a 1.5× retrieval boost; strong cross-project hits
still surface as prior art (flagged via the `hint` field in JSON output).

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
memory store "<content>" -m <type> -t "<tags>"   # auto-detects project
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

/// Entry point invoked from `cli.rs` for `memory setup skill`.
pub fn run(dry_run: bool, print: bool) -> Result<()> {
    if print {
        print!("{SKILL_BODY}");
        return Ok(());
    }

    let target = skill_path().context("could not resolve skill install path")?;

    if dry_run {
        println!("--- DRY RUN: {} ---", target.display());
        print!("{SKILL_BODY}");
        println!("--- end preview ---");
        return Ok(());
    }

    let parent = target
        .parent()
        .context("skill path has no parent directory")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create skill directory {}", parent.display()))?;

    if let Some(backup) = backup_if_exists(&target)? {
        println!("  backup: {}", backup.display());
    }

    std::fs::write(&target, SKILL_BODY)
        .with_context(|| format!("write skill file {}", target.display()))?;
    println!("Installed skill at {}", target.display());
    Ok(())
}

/// True iff the skill file is already installed at the global path.
/// Returns false if the home directory cannot be resolved (the skill cannot
/// have been installed without a home dir).
pub fn is_installed() -> bool {
    skill_path().map(|p| p.exists()).unwrap_or(false)
}

/// Resolve the canonical install path for the agent-memory skill. Returns
/// `Err` only when `dirs::home_dir()` cannot determine a home directory —
/// effectively never on the platforms agent-memory targets, but surfaced as
/// an error rather than panicking so callers can degrade gracefully.
pub fn skill_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not resolve home directory")?;
    Ok(home
        .join(".claude")
        .join("skills")
        .join("agent-memory")
        .join("SKILL.md"))
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
        // directory (`~/.claude/skills/agent-memory/SKILL.md`).
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
        // so the user is not prompted to approve every CLI invocation.
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
            assert!(
                SKILL_BODY.contains(cmd),
                "skill body must document `{cmd}`"
            );
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

    #[test]
    fn skill_path_lands_in_claude_skills_agent_memory_dir() {
        let p = skill_path().expect("home dir resolves on test platforms");
        assert!(
            p.ends_with(".claude/skills/agent-memory/SKILL.md"),
            "unexpected skill path: {}",
            p.display()
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

    /// Minimal isolated tempdir under `target/` so tests don't depend on the
    /// `tempfile` crate (not in this project's deps). Each test gets a unique
    /// directory based on a UUID v4 — the existing `uuid` dep is sufficient.
    fn tempdir_in_target() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "agent-memory-skill-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }
}
