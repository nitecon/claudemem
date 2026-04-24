//! `memory setup rules` — inject the agent-memory usage protocols into known
//! agent rule files (e.g. `~/.claude/CLAUDE.md`, `~/.gemini/GEMINI.md`).
//!
//! Idempotent: uses `<memory-rules>...</memory-rules>` markers so re-runs
//! replace the block in place rather than duplicating it. A `.bak` sibling is
//! written before each modification so the user can recover.
//!
//! Placement rule: if an `<agent-tools-rules>` block is already present (the
//! sibling `agent-tools setup rules` command writes one), the memory block is
//! inserted directly after it so the two protocols stay grouped at the top of
//! the file. Otherwise, the memory block is prepended.
//!
//! Install semantics:
//!
//! - When the target rule file **exists**, its contents are augmented in
//!   place (replace existing block, insert-after-sibling, or prepend).
//! - When the target rule file is **missing** but its parent tool directory
//!   exists (e.g. `~/.codex/` is present but `AGENTS.md` was never created),
//!   a fresh file is written containing a minimal one-line header plus the
//!   memory-rules block. This covers brand-new agent installs that haven't
//!   accumulated any local instructions yet — without it, the user would
//!   have to hand-create an empty AGENTS.md just to give us somewhere to
//!   write, which is exactly the kind of yak-shave the installer should
//!   eliminate.
//! - When the **parent tool directory is missing**, the target is skipped
//!   silently. We never create a tool's home directory — that would be an
//!   install the user didn't ask for.
//!
//! Codex path resolution. Codex honors a `CODEX_HOME` environment variable
//! with `~/.codex/` as the default. `~/.config/codex/` is the legacy
//! XDG-style fallback (some installers still use it). Both must never be
//! written to at the same time: they would target the same Codex instance
//! twice and the user would see double rules. Resolution order:
//!
//!   1. `$CODEX_HOME/AGENTS.md` if `CODEX_HOME` is set and the directory
//!      exists — explicit user override wins.
//!   2. `~/.codex/AGENTS.md` if `~/.codex/` exists (modern default).
//!   3. `~/.config/codex/AGENTS.md` if only the XDG path is present.
//!   4. Skip Codex entirely if none of the above.
//!
//! Native-memory coupling: when the rules block is installed, we must also
//! disable each supported agent tool's native memory system. The rules
//! block redirects the agent to route ALL memory operations through the
//! `memory` CLI, so leaving native memory on would cause the agent to
//! write into two stores in parallel — the tool's built-in memory file
//! and this tool's SQLite DB — producing silent duplication and drift.
//! The `--remove` path reverses each merge by deleting the key rather
//! than forcing it back to `true` (absence and `true` are semantically
//! different and we don't know what the user had before install).
//!
//! Per-agent coupling:
//!
//! | Agent    | Target file                   | Key written                       |
//! |----------|-------------------------------|-----------------------------------|
//! | Claude   | `~/.claude/settings.json`     | `"autoMemoryEnabled": false`      |
//! | Gemini   | `~/.gemini/settings.json`     | `"save_memory"` in `excludeTools` |
//! | Codex    | `~/.codex/config.toml`        | `[features] memories = false`     |

use crate::setup::codex_config_toml;
use crate::setup::gemini_settings_json;
use crate::setup::settings_json::{self, SettingsOutcome};
use anyhow::{Context, Result};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

const OPEN_MARKER: &str = "<memory-rules>";
const CLOSE_MARKER: &str = "</memory-rules>";

const SIBLING_OPEN: &str = "<agent-tools-rules>";
const SIBLING_CLOSE: &str = "</agent-tools-rules>";

/// Minimal header written as the first line of a freshly-created rule file
/// so the output doesn't start with a bare `<memory-rules>` tag (which reads
/// like malformed content to a human opening the file). A single markdown
/// heading keeps things obviously intentional without looking authored —
/// subsequent installs find the marker block and update it in place, they
/// don't touch the header.
const FRESH_FILE_HEADER: &str = "# Agent instructions\n\n";

const HEADER: &str = r#"## Agent Memory — Mandatory Protocols

These directives are auto-injected by `memory setup`. Re-run that command to
refresh; do not edit between the marker tags — your edits will be overwritten
on the next update.
"#;

const MEMORY_SECTION: &str = r#"
### Memory Operations (MANDATORY)

**Binary:** `memory` (installed at `/opt/agentic/bin/memory` on Linux/macOS,
`%USERPROFILE%\.agentic\bin\memory.exe` on Windows) — call directly via Bash.
Do NOT use MCP or skills for memory during normal workflow.

**The "Memory First/Last" Rule:** Every task must begin with a `context` or
`search` call and end with a `store` call if functionality changed.

### Scope tiers

Every memory is stored under one of two scopes; retrieval boosts both:

| Scope                      | Boost  | When to use                                    |
|----------------------------|--------|------------------------------------------------|
| **Current project** (cwd)  | 1.5×   | Repo-specific decisions, patterns, bugs        |
| **Global** (`__global__`)  | 1.25×  | Universal user preferences / directives        |
| Other project              | 1.0×   | Surfaces only as prior art via the `hint` field |

`store`, `search`, and `context` auto-detect the current project from the
cwd's git remote (reduced to the repo shortname). A single `context` call
returns both current-project and global hits — no second query needed.

```bash
# Context — top-K relevant memories for a task (boost cwd + global)
memory context "<task description>" -k <limit>

# Search — hybrid BM25 + vector search (boost cwd + global)
memory search "<query>" -k <limit>

# Store — save a new project-scoped memory (cwd auto-detected)
memory store "<content>" -m <type> -t "<tags>"

# Store — save a universal preference (applies across every repo)
memory store "<content>" -m <type> --scope global -t "<tags>"
# types: user, feedback, project, reference

# Get — fetch full content for specific IDs (pair with brief search)
memory get <uuid> [<uuid>...]

# Recall — filter by project/agent/tags/type
memory recall -m <type> -t "<tags>" -p "<project>" -k <limit>

# List — list all memories (optionally filtered)
memory list -k 50 --project <proj>

# Projects — list distinct project idents (spot alias mismatches)
memory projects

# Move — reassign the project ident on one or many memories
memory move --from "<old>" --to "<new>" [--dry-run]
memory move --id <uuid> --to "<proj>"

# Copy — duplicate memories under a new project ident
memory copy --from "<old>" --to "<new>" [--dry-run]
memory copy --id <uuid> --to "<proj>"

# Forget — remove a memory by ID (or by search query)
memory forget --id <uuid>
memory forget --query "<query>"

# Prune — decay stale/low-access memories
memory prune --max-age-days 90 [--dry-run]

# Update — check for and install the latest version
memory update
```

### Retrieval strategy

1. **Pre-Task**: run `memory context "<task>"` before reading code. If a similar
   pattern exists, refactor or extend it rather than re-inventing.
2. **Two-stage fetch**: default `brief` output returns previews; follow up with
   `memory get <id>` for the handful you actually want to read in full.
3. **Cross-project hits**: when the response includes a `hint` field, treat
   those memories as prior-art or general guidance, not direct context.
4. **Global-scope hits**: the `hint` field also surfaces the count of
   global-scope preferences in your top-K. Treat them as directives, not
   suggestions — they encode rules the user has already stated once.
5. **Post-Task**: run `memory store` for any non-obvious decisions, user
   preferences, or reusable patterns. Audit-ready descriptions — explain the
   "why," not just the "what."

### Rule A — Pre-action behavior recall (MANDATORY)

Before starting **any** user-requested task — development, SRE, writing,
design, research, any domain — run **one** `memory context "<task>"` call
first. A single call returns both general directives (global scope, 1.25×
boost) and project-specific directives (current-project, 1.5× boost). Do
not skip this step for "quick" tasks: directives the user has already
stated must never need to be re-stated.

If the response's `hint` field flags zero global-scope matches, pause and
reflect: has the user stated a preference relevant to this task's domain
that you simply aren't finding? If unsure, ask them directly before acting.

### Rule B — Post-action scope classification (MANDATORY)

After completing an action, if the user stated or implied any directive,
preference, or corrective rule during the session, you **MUST** store it —
and you **MUST** classify its scope.

Classification rules:

- **Global** (`--scope global`) — universal preference. Signals include:
  "I always", "I never", "from now on", "I prefer", "don't ever",
  "whenever we", "in general", any phrasing that sounds like a personal
  policy or work style.
- **Project** (`--scope project`, the default) — specific to this repo,
  service, or codebase. Signals include: "in this repo", "for this
  service", "on this project", "here we", any phrasing tied to the
  current codebase or stack.
- **Ambiguous** — phrasing could reasonably apply either way. You
  **MUST ask** the user before storing: *"Is this a general preference
  (applies across all projects) or specific to this project?"* Do
  **not** silently default to either scope. Do **not** skip the ask to
  save tokens — a silent mis-classification costs more than one question.

Example:
```bash
memory store "User never wants PRs opened unless they explicitly ask" \
  -m feedback --scope global -t "workflow,pr"
```
"#;

/// Entry point invoked from `cli.rs` for `memory setup rules`.
///
/// Arguments:
/// - `target` — explicit file path (bypasses auto-detection).
/// - `all` — install to every detected rule file without prompting.
/// - `dry_run` — print the would-be output and exit without writing.
/// - `print` — emit the rules block to stdout and exit (no filesystem IO).
/// - `remove` — inverse of install: strip the `<memory-rules>` block from
///   each target and restore the prior state by also removing the
///   `autoMemoryEnabled` key from any paired Claude `settings.json`.
pub fn run(
    target: Option<PathBuf>,
    all: bool,
    dry_run: bool,
    print: bool,
    remove: bool,
) -> Result<()> {
    if print {
        print!("{}", build_block());
        return Ok(());
    }

    let candidates = match &target {
        Some(t) => vec![t.clone()],
        None => detect_agent_files(),
    };

    if candidates.is_empty() {
        anyhow::bail!(
            "No agent rule files detected. Tried:\n  \
             ~/.claude/CLAUDE.md\n  \
             ~/.gemini/GEMINI.md\n  \
             ~/.codex/AGENTS.md\n  \
             ~/.config/codex/AGENTS.md\n\
             Re-run with `--target <path>` to point at a specific file."
        );
    }

    let chosen = if all || target.is_some() || candidates.len() == 1 {
        candidates
    } else {
        prompt_user_for_selection(&candidates)?
    };

    if chosen.is_empty() {
        println!("Cancelled — no files modified.");
        return Ok(());
    }

    let block = build_block();
    let mut any_failed = false;
    for path in &chosen {
        let result = if remove {
            uninstall(path, dry_run).map(OperationOutcome::Remove)
        } else {
            inject(path, &block, dry_run).map(OperationOutcome::Install)
        };
        match result {
            Ok(OperationOutcome::Install(InjectOutcome::DryRun(preview))) => {
                println!("--- DRY RUN: {} ---", path.display());
                print!("{preview}");
                println!("--- end preview ---");
            }
            Ok(OperationOutcome::Install(InjectOutcome::Replaced { backup })) => {
                println!("Updated existing block in {}", path.display());
                println!("  backup: {}", backup.display());
                sync_auto_memory(path, AutoMemoryAction::Disable, dry_run, &mut any_failed);
            }
            Ok(OperationOutcome::Install(InjectOutcome::InsertedAfterSibling { backup })) => {
                println!(
                    "Inserted new block after <agent-tools-rules> in {}",
                    path.display()
                );
                println!("  backup: {}", backup.display());
                sync_auto_memory(path, AutoMemoryAction::Disable, dry_run, &mut any_failed);
            }
            Ok(OperationOutcome::Install(InjectOutcome::Prepended { backup })) => {
                println!("Prepended new block to {}", path.display());
                println!("  backup: {}", backup.display());
                sync_auto_memory(path, AutoMemoryAction::Disable, dry_run, &mut any_failed);
            }
            Ok(OperationOutcome::Install(InjectOutcome::Created)) => {
                // Emit the light-XML status line the README documents for
                // fresh-file installs. The human-readable line mirrors the
                // other install-outcome messages so scripted consumers and
                // casual users both have something useful.
                let agent = agent_label_for_rule_file(path);
                println!(
                    r#"<setup status="rules_created" agent="{}" path="{}"/>"#,
                    agent,
                    path.display()
                );
                println!("Created new rule file {}", path.display());
                sync_auto_memory(path, AutoMemoryAction::Disable, dry_run, &mut any_failed);
            }
            Ok(OperationOutcome::Remove(RemoveOutcome::DryRun(preview))) => {
                println!("--- DRY RUN (remove): {} ---", path.display());
                print!("{preview}");
                println!("--- end preview ---");
            }
            Ok(OperationOutcome::Remove(RemoveOutcome::Removed { backup })) => {
                println!("Removed block from {}", path.display());
                println!("  backup: {}", backup.display());
                sync_auto_memory(path, AutoMemoryAction::Remove, dry_run, &mut any_failed);
            }
            Ok(OperationOutcome::Remove(RemoveOutcome::NotPresent)) => {
                println!(
                    "No <memory-rules> block in {} (nothing to remove)",
                    path.display()
                );
                // Still attempt the settings.json key removal — a user who
                // manually stripped the block may still have the key lingering.
                sync_auto_memory(path, AutoMemoryAction::Remove, dry_run, &mut any_failed);
            }
            Err(e) => {
                eprintln!("Failed to update {}: {e:#}", path.display());
                any_failed = true;
            }
        }
    }
    if any_failed {
        anyhow::bail!("one or more files could not be updated");
    }
    Ok(())
}

// -- Internals ---------------------------------------------------------------

/// Built-in detection list. Only home-dir global rule files; project-local
/// instruction files (e.g. `./CLAUDE.md`) are intentionally left alone since
/// they're per-repo content the user should edit directly.
///
/// A candidate qualifies when **either** the file already exists **or** the
/// tool's parent directory exists (meaning the tool is installed and we
/// have a valid target to write to). The second case is what lets fresh
/// installs — e.g. a just-installed Codex that has never written
/// `AGENTS.md` — pick up the memory-rules block on their first
/// `memory setup rules` run.
///
/// We never create a tool's home directory on the user's behalf: a missing
/// parent means the tool isn't installed and any write would be surprising.
///
/// Codex has two historical locations; [`resolve_codex_rule_file`] picks
/// the single correct one based on `CODEX_HOME`, then `~/.codex/`, then
/// `~/.config/codex/`. Writing to both would double-install into the same
/// Codex instance.
pub(crate) fn detect_agent_files() -> Vec<PathBuf> {
    detect_agent_files_with_env(std::env::var("CODEX_HOME").ok().map(PathBuf::from))
}

/// Testable core of [`detect_agent_files`]. `codex_home_override` mirrors
/// the `CODEX_HOME` env var and lets unit tests drive Codex path resolution
/// without leaking env state across the test runner.
fn detect_agent_files_with_env(codex_home_override: Option<PathBuf>) -> Vec<PathBuf> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return vec![],
    };
    let mut out = Vec::with_capacity(3);

    let claude = home.join(".claude").join("CLAUDE.md");
    if file_or_parent_exists(&claude) {
        out.push(claude);
    }

    let gemini = home.join(".gemini").join("GEMINI.md");
    if file_or_parent_exists(&gemini) {
        out.push(gemini);
    }

    if let Some(codex) = resolve_codex_rule_file(&home, codex_home_override.as_deref()) {
        out.push(codex);
    }

    out
}

/// Resolve the single Codex `AGENTS.md` target, honoring `CODEX_HOME` and
/// the legacy XDG fallback. Returns `None` when no Codex install is
/// visible.
///
/// Precedence:
///   1. `CODEX_HOME` env var — explicit override always wins, provided its
///      directory exists.
///   2. `~/.codex/` — current Codex default.
///   3. `~/.config/codex/` — legacy XDG-style location some installers
///      still ship.
///
/// Rule-file creation is always allowed when the parent directory exists,
/// matching the file-or-parent semantics of the other candidates. We never
/// create a fresh Codex home, only a missing `AGENTS.md` inside an
/// already-installed one.
fn resolve_codex_rule_file(home: &Path, codex_home_override: Option<&Path>) -> Option<PathBuf> {
    if let Some(override_dir) = codex_home_override {
        if override_dir.is_dir() {
            return Some(override_dir.join("AGENTS.md"));
        }
        // If the override is set but points at a missing directory, skip
        // Codex entirely rather than falling through — the user told us
        // explicitly where their Codex lives, and we must not silently
        // second-guess that by writing to a different location.
        return None;
    }

    let primary = home.join(".codex");
    if primary.is_dir() {
        return Some(primary.join("AGENTS.md"));
    }

    let xdg = home.join(".config").join("codex");
    if xdg.is_dir() {
        return Some(xdg.join("AGENTS.md"));
    }

    None
}

/// True when the file already exists or its immediate parent directory
/// exists. Used as the install-candidate filter so we pick up both
/// established and fresh tool installs, but never create a tool's home
/// directory ourselves.
fn file_or_parent_exists(path: &Path) -> bool {
    if path.exists() {
        return true;
    }
    match path.parent() {
        Some(parent) => parent.is_dir(),
        None => false,
    }
}

/// Map a rule-file path to a short agent label for `<setup>` status lines.
/// Falls back to `"unknown"` when the file's parent doesn't match any
/// recognized tool home — that path is only reachable via explicit
/// `--target`, where the user already knows which tool they're targeting.
fn agent_label_for_rule_file(path: &Path) -> &'static str {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match name.as_str() {
        "claude.md" => "claude",
        "gemini.md" => "gemini",
        "agents.md" => "codex",
        _ => "unknown",
    }
}

/// Probe helper exposed to the interactive menu: does the given file already
/// contain the `<memory-rules>` block?
pub(crate) fn file_has_rules_block(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .map(|s| s.contains(OPEN_MARKER) && s.contains(CLOSE_MARKER))
        .unwrap_or(false)
}

fn build_block() -> String {
    let mut body = String::new();
    body.push_str(HEADER);
    body.push_str(MEMORY_SECTION);
    format!("{OPEN_MARKER}\n{body}{CLOSE_MARKER}\n")
}

enum InjectOutcome {
    DryRun(String),
    Replaced {
        backup: PathBuf,
    },
    InsertedAfterSibling {
        backup: PathBuf,
    },
    Prepended {
        backup: PathBuf,
    },
    /// Target file did not exist; a fresh file was written containing the
    /// header plus the memory-rules block. No `.bak` sibling is produced
    /// because there was nothing to back up.
    Created,
}

/// Result of an uninstall pass. Parallel to [`InjectOutcome`] but narrower:
/// uninstall either emits a dry-run preview, a successful removal with a
/// `.bak` sibling, or reports the block wasn't present to begin with.
enum RemoveOutcome {
    DryRun(String),
    Removed {
        backup: PathBuf,
    },
    /// Block markers not found. The caller still runs the paired
    /// `settings.json` cleanup in case only the block was manually
    /// stripped but the setting remained.
    NotPresent,
}

/// Unifies install + uninstall so the dispatch loop in `run()` has one
/// match to drive status output from.
enum OperationOutcome {
    Install(InjectOutcome),
    Remove(RemoveOutcome),
}

/// Direction for the `settings.json` sync that pairs with each rule-file
/// mutation. Install disables native auto-memory; remove strips the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoMemoryAction {
    Disable,
    Remove,
}

fn inject(path: &Path, block: &str, dry_run: bool) -> Result<InjectOutcome> {
    // Capture existence up front — we need it to distinguish "augment an
    // existing file" from "create a fresh file with a header" after the
    // write has happened.
    let file_existed = path.exists();
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let already_present = existing.contains(OPEN_MARKER) && existing.contains(CLOSE_MARKER);
    let has_sibling = existing.contains(SIBLING_OPEN) && existing.contains(SIBLING_CLOSE);
    let (new_content, mode) =
        compute_new_content(&existing, block, already_present, has_sibling, file_existed);

    if dry_run {
        return Ok(InjectOutcome::DryRun(new_content));
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent of {}", path.display()))?;
    }

    // Only write a `.bak` sidecar when there was prior content to preserve.
    // Freshly-created files need no backup — there is nothing to roll back
    // to, and leaving a zero-byte `.bak` would be confusing.
    let backup = if file_existed {
        let b = backup_path(path);
        std::fs::write(&b, &existing)
            .with_context(|| format!("write backup to {}", b.display()))?;
        Some(b)
    } else {
        None
    };
    std::fs::write(path, &new_content)
        .with_context(|| format!("write updated file {}", path.display()))?;

    Ok(match mode {
        InjectMode::Replaced => InjectOutcome::Replaced {
            backup: backup.expect("replace mode only reachable when file existed"),
        },
        InjectMode::InsertedAfterSibling => InjectOutcome::InsertedAfterSibling {
            backup: backup.expect("insert-after-sibling only reachable when file existed"),
        },
        InjectMode::Prepended => InjectOutcome::Prepended {
            backup: backup.expect("prepend only reachable when file existed"),
        },
        InjectMode::Created => InjectOutcome::Created,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InjectMode {
    Replaced,
    InsertedAfterSibling,
    Prepended,
    /// Target file did not exist — write a fresh file with the
    /// [`FRESH_FILE_HEADER`] prelude followed by the memory-rules block.
    Created,
}

/// Strip the `<memory-rules>` block from `path`, writing a `.bak` sibling
/// first. Matches the semantics of [`inject`]: dry-run returns the computed
/// output without touching the disk; real runs write atomically after backing
/// up the original.
///
/// When the block markers are absent we short-circuit to `NotPresent` without
/// writing a backup — the file is already in the desired post-remove state
/// and creating a redundant `.bak` would only confuse the user.
fn uninstall(path: &Path, dry_run: bool) -> Result<RemoveOutcome> {
    let existing = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(RemoveOutcome::NotPresent),
        Err(e) => {
            return Err(anyhow::Error::new(e).context(format!("read rule file {}", path.display())))
        }
    };

    let has_block = existing.contains(OPEN_MARKER) && existing.contains(CLOSE_MARKER);
    if !has_block {
        return Ok(RemoveOutcome::NotPresent);
    }

    let new_content = strip_block(&existing);
    if dry_run {
        return Ok(RemoveOutcome::DryRun(new_content));
    }

    let backup = backup_path(path);
    std::fs::write(&backup, &existing)
        .with_context(|| format!("write backup to {}", backup.display()))?;
    std::fs::write(path, &new_content)
        .with_context(|| format!("write updated file {}", path.display()))?;
    Ok(RemoveOutcome::Removed { backup })
}

/// Remove the first `<memory-rules>…</memory-rules>` span from `existing`,
/// along with at most one trailing newline so the file doesn't end up with
/// a stray blank line where the block used to be. Pure helper so unit tests
/// can verify the stripped output byte-for-byte.
fn strip_block(existing: &str) -> String {
    let open_idx = match existing.find(OPEN_MARKER) {
        Some(i) => i,
        None => return existing.to_string(),
    };
    let close_idx = match existing.find(CLOSE_MARKER) {
        Some(i) => i,
        None => return existing.to_string(),
    };
    let close_end = close_idx + CLOSE_MARKER.len();
    let after_start = if existing[close_end..].starts_with('\n') {
        close_end + 1
    } else {
        close_end
    };
    let before = &existing[..open_idx];
    let after = &existing[after_start..];

    // If the block was followed by a blank separator line (i.e. the
    // install helpers inserted an extra `\n` between it and the next
    // content), consume that too.
    let trimmed_after = after.strip_prefix('\n').unwrap_or(after);

    format!("{before}{trimmed_after}")
}

/// Drive the paired native-memory-disable mutation after a successful
/// rule-file change. Dispatches on the rule file's agent kind:
///
/// - Claude (`CLAUDE.md`) → write `autoMemoryEnabled: false` into
///   `settings.json`.
/// - Gemini (`GEMINI.md`) → add `save_memory` to `excludeTools` in
///   `settings.json`.
/// - Codex (`AGENTS.md`) → set `[features] memories = false` in
///   `config.toml`.
/// - Anything else → silent no-op.
///
/// Status lines are emitted as light-XML `<setup .../>` records matching
/// the installer's existing output vocabulary. The `key` and `value`
/// attributes name the concrete setting touched so scripted consumers and
/// humans both have a single line to grep against.
///
/// Failures here do NOT unwind the rule-file mutation — the `.bak` sibling
/// lets the user recover from that if needed. Instead we flag `any_failed`
/// so the caller still exits non-zero, while the successfully-mutated rule
/// file stays in place. This trade-off matches the existing loop semantics
/// (per-target failures don't abort the whole batch).
fn sync_auto_memory(
    rule_file: &Path,
    action: AutoMemoryAction,
    dry_run: bool,
    any_failed: &mut bool,
) {
    // Claude — JSON scalar key.
    if let Some(settings_path) = settings_json::settings_path_for_rule_file(rule_file) {
        run_settings_op(
            &settings_path,
            settings_json::AUTO_MEMORY_KEY,
            "false",
            "(removed)",
            action,
            dry_run,
            any_failed,
            settings_json::disable_auto_memory,
            settings_json::remove_auto_memory,
        );
        return;
    }

    // Gemini — JSON array membership.
    if let Some(settings_path) = gemini_settings_json::settings_path_for_rule_file(rule_file) {
        run_settings_op(
            &settings_path,
            gemini_settings_json::EXCLUDE_TOOLS_KEY,
            gemini_settings_json::SAVE_MEMORY_TOOL,
            // Remove `value` attribute is always `(removed)`; the paired
            // `key` already names the entry being pulled.
            "(removed)",
            action,
            dry_run,
            any_failed,
            gemini_settings_json::disable_save_memory,
            gemini_settings_json::remove_save_memory,
        );
        return;
    }

    // Codex — TOML nested key. Derived from the rule file's own parent
    // directory so `CODEX_HOME` / XDG precedence stays consistent with
    // the AGENTS.md target the caller just mutated.
    let home = dirs::home_dir();
    let codex_home_override = std::env::var("CODEX_HOME").ok().map(PathBuf::from);
    if let Some(home) = home.as_deref() {
        if let Some(config_path) = codex_config_toml::config_path_for_rule_file(
            rule_file,
            home,
            codex_home_override.as_deref(),
        ) {
            run_settings_op(
                &config_path,
                "features.memories",
                "false",
                "(removed)",
                action,
                dry_run,
                any_failed,
                codex_config_toml::disable_memories,
                codex_config_toml::remove_memories,
            );
        }
    }
    // All other rule files → no paired settings to touch.
}

/// Shared executor for the three per-agent settings mutations. Folds the
/// common dry-run handling, outcome-to-status mapping, and status-line
/// emission into one place so each per-agent caller only has to supply the
/// display strings and the two operation closures.
#[allow(clippy::too_many_arguments)]
fn run_settings_op<D, R>(
    settings_path: &Path,
    key_label: &str,
    install_value: &str,
    remove_value: &str,
    action: AutoMemoryAction,
    dry_run: bool,
    any_failed: &mut bool,
    disable: D,
    remove: R,
) where
    D: FnOnce(&Path) -> Result<SettingsOutcome>,
    R: FnOnce(&Path) -> Result<SettingsOutcome>,
{
    if dry_run {
        println!(
            r#"<setup status="settings_dry_run" path="{}" key="{}" action="{}"/>"#,
            settings_path.display(),
            key_label,
            match action {
                AutoMemoryAction::Disable => "disable",
                AutoMemoryAction::Remove => "remove",
            }
        );
        return;
    }

    let outcome = match action {
        AutoMemoryAction::Disable => disable(settings_path),
        AutoMemoryAction::Remove => remove(settings_path),
    };

    match outcome {
        Ok(state) => {
            let status = match state {
                SettingsOutcome::Created => "settings_created",
                SettingsOutcome::Updated => "settings_updated",
                SettingsOutcome::AlreadyCorrect => "settings_already_correct",
                SettingsOutcome::Removed => "settings_key_removed",
                SettingsOutcome::AlreadyAbsent => "settings_key_already_absent",
            };
            let value = match action {
                AutoMemoryAction::Disable => install_value,
                AutoMemoryAction::Remove => remove_value,
            };
            println!(
                r#"<setup status="{}" path="{}" key="{}" value="{}"/>"#,
                status,
                settings_path.display(),
                key_label,
                value,
            );
        }
        Err(e) => {
            eprintln!(
                "Failed to update settings at {}: {e:#}",
                settings_path.display()
            );
            *any_failed = true;
        }
    }
}

/// Pure helper: build the new file body from existing content + the rules
/// block. Factored out so unit tests can verify idempotency and placement
/// without touching the filesystem.
///
/// `file_existed` distinguishes two empty-content cases: a pre-existing but
/// empty file (treated as Prepended — the user had a file we're augmenting)
/// vs. a brand-new file we're creating for them (treated as Created and
/// prefixed with [`FRESH_FILE_HEADER`] so the output doesn't look like a
/// bare dump of XML-shaped markers).
fn compute_new_content(
    existing: &str,
    block: &str,
    already_present: bool,
    has_sibling: bool,
    file_existed: bool,
) -> (String, InjectMode) {
    if already_present {
        (replace_block(existing, block), InjectMode::Replaced)
    } else if !file_existed {
        // Brand-new file — give it a minimal header so opening the file in
        // an editor doesn't greet the user with a naked `<memory-rules>`.
        (format!("{FRESH_FILE_HEADER}{block}"), InjectMode::Created)
    } else if existing.is_empty() {
        (block.to_string(), InjectMode::Prepended)
    } else if has_sibling {
        (
            insert_after_sibling(existing, block),
            InjectMode::InsertedAfterSibling,
        )
    } else {
        (format!("{block}\n{existing}"), InjectMode::Prepended)
    }
}

fn replace_block(existing: &str, new_block: &str) -> String {
    let open_idx = existing.find(OPEN_MARKER).unwrap_or(0);
    let close_idx = existing.find(CLOSE_MARKER).unwrap_or(existing.len());
    let close_end = close_idx + CLOSE_MARKER.len();
    let after_start = if existing[close_end..].starts_with('\n') {
        close_end + 1
    } else {
        close_end
    };
    let before = &existing[..open_idx];
    let after = &existing[after_start..];
    format!("{before}{new_block}{after}")
}

/// Insert the memory block immediately after the closing `</agent-tools-rules>`
/// tag, consuming one trailing newline if present so spacing stays tidy.
fn insert_after_sibling(existing: &str, new_block: &str) -> String {
    let sibling_close = existing
        .find(SIBLING_CLOSE)
        .expect("caller verified sibling close present");
    let sibling_end = sibling_close + SIBLING_CLOSE.len();
    let after_start = if existing[sibling_end..].starts_with('\n') {
        sibling_end + 1
    } else {
        sibling_end
    };
    let before = &existing[..after_start];
    let after = &existing[after_start..];
    // Ensure blank line between sibling and memory block, and between memory
    // block and whatever follows.
    format!("{before}\n{new_block}{after}")
}

fn backup_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".bak");
    PathBuf::from(s)
}

fn prompt_user_for_selection(candidates: &[PathBuf]) -> Result<Vec<PathBuf>> {
    println!("Detected agent rule files:");
    for (i, p) in candidates.iter().enumerate() {
        println!("  {}) {}", i + 1, p.display());
    }
    print!(
        "Update [a]ll, [1-{}] specific, [c]ancel: ",
        candidates.len()
    );
    io::stdout().flush().context("flush stdout")?;

    let mut input = String::new();
    io::stdin()
        .lock()
        .read_line(&mut input)
        .context("read selection")?;
    let s = input.trim().to_ascii_lowercase();

    if s.is_empty() || s == "c" || s == "cancel" {
        return Ok(vec![]);
    }
    if s == "a" || s == "all" {
        return Ok(candidates.to_vec());
    }
    if let Ok(n) = s.parse::<usize>() {
        if n >= 1 && n <= candidates.len() {
            return Ok(vec![candidates[n - 1].clone()]);
        }
    }
    anyhow::bail!("invalid selection: {s:?}");
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_block_is_wrapped_with_markers() {
        let b = build_block();
        assert!(b.starts_with(OPEN_MARKER));
        assert!(b.trim_end().ends_with(CLOSE_MARKER));
        assert!(b.contains("memory context"));
        assert!(b.contains("memory store"));
        assert!(b.contains("memory get"));
        assert!(b.contains("memory projects"));
        assert!(b.contains("memory move"));
        assert!(b.contains("memory copy"));
        // Markers appear exactly once each.
        assert_eq!(b.matches(OPEN_MARKER).count(), 1);
        assert_eq!(b.matches(CLOSE_MARKER).count(), 1);
    }

    /// The new scope/rules additions must appear verbatim so a regression
    /// that drops the mandatory-ask clause is caught at build time rather
    /// than discovered in the wild.
    #[test]
    fn build_block_documents_scope_and_new_rules() {
        let b = build_block();
        // CLI example with the scope flag.
        assert!(
            b.contains("--scope global"),
            "block must show the --scope global CLI example"
        );
        // Scope tier table references both boost values.
        assert!(
            b.contains("1.5×"),
            "block must reference the 1.5× project boost"
        );
        assert!(
            b.contains("1.25×"),
            "block must reference the 1.25× global boost"
        );
        // The sentinel must be named so users inspecting the DB aren't
        // confused by stray `__global__` rows.
        assert!(
            b.contains("__global__"),
            "block must name the sentinel project ident"
        );
        // Rule headings — exact strings the agent learns to look for.
        assert!(
            b.contains("Pre-action behavior recall"),
            "Rule A heading must be present"
        );
        assert!(
            b.contains("Post-action scope classification"),
            "Rule B heading must be present"
        );
        // The MUST-ask clause is the load-bearing one. Spelled "MUST ask"
        // (two tokens) so the assertion is forgiving about the surrounding
        // Markdown emphasis.
        assert!(
            b.contains("MUST ask"),
            "Rule B must include the mandatory-ask clause for ambiguous phrasing"
        );
    }

    #[test]
    fn compute_new_content_prepends_when_absent_and_no_sibling() {
        let existing = "# Existing instructions\n\nfoo bar\n";
        let block = build_block();
        // `file_existed = true` — the user already had a CLAUDE.md-style
        // file we're augmenting.
        let (out, mode) = compute_new_content(existing, &block, false, false, true);
        assert_eq!(mode, InjectMode::Prepended);
        assert!(out.starts_with(OPEN_MARKER));
        assert!(out.contains("# Existing instructions"));
        assert!(out.contains(&format!("{CLOSE_MARKER}\n\n# Existing instructions")));
    }

    #[test]
    fn compute_new_content_inserts_after_sibling_when_present() {
        let block = build_block();
        let existing =
            format!("{SIBLING_OPEN}\nagent-tools stuff\n{SIBLING_CLOSE}\n\n# Rest of file\n");
        let (out, mode) = compute_new_content(&existing, &block, false, true, true);
        assert_eq!(mode, InjectMode::InsertedAfterSibling);
        // Sibling block still first.
        let sibling_pos = out.find(SIBLING_CLOSE).unwrap();
        let memory_pos = out.find(OPEN_MARKER).unwrap();
        assert!(sibling_pos < memory_pos);
        // Both blocks present, original rest preserved.
        assert!(out.contains("agent-tools stuff"));
        assert!(out.contains("# Rest of file"));
        assert!(out.contains("memory context"));
    }

    #[test]
    fn compute_new_content_replaces_in_place() {
        let block = build_block();
        let existing =
            format!("# Header\n\n{OPEN_MARKER}\nold memory body\n{CLOSE_MARKER}\n\n# Footer\n");
        let (out, mode) = compute_new_content(&existing, &block, true, false, true);
        assert_eq!(mode, InjectMode::Replaced);
        assert!(out.starts_with("# Header"));
        assert!(out.contains("# Footer"));
        assert!(!out.contains("old memory body"));
        assert!(out.contains("memory context"));
        assert_eq!(out.matches(OPEN_MARKER).count(), 1);
        assert_eq!(out.matches(CLOSE_MARKER).count(), 1);
    }

    #[test]
    fn compute_new_content_is_idempotent() {
        let block = build_block();
        let existing = "# Header\n";
        let (once, _) = compute_new_content(existing, &block, false, false, true);
        let (twice, mode) = compute_new_content(&once, &block, true, false, true);
        assert_eq!(mode, InjectMode::Replaced);
        assert_eq!(once, twice);
    }

    #[test]
    fn compute_new_content_handles_empty_file() {
        let block = build_block();
        // Empty-but-existing file — Prepended, not Created, because the
        // file is already on disk (possibly touched by another tool).
        let (out, mode) = compute_new_content("", &block, false, false, true);
        assert_eq!(mode, InjectMode::Prepended);
        assert_eq!(out, block);
    }

    /// The fresh-file path: no pre-existing file at all. Output must begin
    /// with [`FRESH_FILE_HEADER`] so opening the file in an editor greets
    /// the user with something that looks intentional, not a bare block.
    #[test]
    fn compute_new_content_creates_fresh_file_with_header() {
        let block = build_block();
        let (out, mode) = compute_new_content("", &block, false, false, false);
        assert_eq!(mode, InjectMode::Created);
        assert!(
            out.starts_with(FRESH_FILE_HEADER),
            "fresh file must start with the header: {out}"
        );
        assert!(
            out.contains(OPEN_MARKER),
            "fresh file must contain the rules block"
        );
        assert!(out.contains("memory context"));
    }

    /// Once installed, a re-run of the installer on the same file must fall
    /// through to Replaced regardless of the `file_existed` flag coming back
    /// as true — the block-is-present check wins.
    #[test]
    fn compute_new_content_created_file_round_trips_to_replaced_on_refresh() {
        let block = build_block();
        let (fresh, _) = compute_new_content("", &block, false, false, false);
        let (refreshed, mode) = compute_new_content(&fresh, &block, true, false, true);
        assert_eq!(mode, InjectMode::Replaced);
        // Exactly one block after the refresh — no accumulation.
        assert_eq!(refreshed.matches(OPEN_MARKER).count(), 1);
        assert_eq!(refreshed.matches(CLOSE_MARKER).count(), 1);
        // Header survives the in-place replace.
        assert!(refreshed.starts_with(FRESH_FILE_HEADER));
    }

    #[test]
    fn re_running_after_sibling_added_keeps_memory_block_once() {
        // User first runs `memory setup` on a fresh file (prepends).
        // Later they run `agent-tools setup rules` (prepends its own block).
        // Third run of `memory setup` detects its own existing block and
        // replaces in place, *not* re-inserting after the sibling.
        let block = build_block();
        let (first, _) = compute_new_content("# Header\n", &block, false, false, true);
        let with_sibling = format!("{SIBLING_OPEN}\nfoo\n{SIBLING_CLOSE}\n\n{first}");
        let (after_refresh, mode) = compute_new_content(&with_sibling, &block, true, true, true);
        assert_eq!(mode, InjectMode::Replaced);
        assert_eq!(after_refresh.matches(OPEN_MARKER).count(), 1);
        assert_eq!(after_refresh.matches(CLOSE_MARKER).count(), 1);
    }

    #[test]
    fn backup_path_appends_bak() {
        let p = PathBuf::from("/tmp/foo/CLAUDE.md");
        assert_eq!(backup_path(&p), PathBuf::from("/tmp/foo/CLAUDE.md.bak"));
    }

    /// `strip_block` must remove the markers AND one trailing newline so the
    /// file doesn't accumulate blank lines after repeated install/remove
    /// cycles. This is the inverse round-trip of `compute_new_content`.
    #[test]
    fn strip_block_removes_block_and_trailing_newline() {
        let block = build_block();
        let before_block = "# Header\n\n";
        let after_block = "# Footer\n";
        let installed = format!("{before_block}{block}\n{after_block}");
        let stripped = strip_block(&installed);
        assert_eq!(stripped, format!("{before_block}{after_block}"));
        assert!(!stripped.contains(OPEN_MARKER));
        assert!(!stripped.contains(CLOSE_MARKER));
    }

    /// Install → strip must return to the original content byte-for-byte
    /// so users can cleanly back out of a rules install.
    #[test]
    fn install_then_strip_is_lossless() {
        let block = build_block();
        let original = "# Existing instructions\n\nfoo bar\n";
        let (installed, _) = compute_new_content(original, &block, false, false, true);
        let stripped = strip_block(&installed);
        assert_eq!(stripped, original);
    }

    /// Stripping a file that never had the block must be a no-op rather than
    /// a panic — `uninstall` short-circuits before calling `strip_block`,
    /// but the pure helper still needs to behave sanely for defense in depth.
    #[test]
    fn strip_block_no_markers_returns_unchanged() {
        let input = "# Just a header\n\nno markers here\n";
        assert_eq!(strip_block(input), input);
    }

    // -- fresh-install / codex-precedence tests --------------------------------

    /// Minimal isolated tempdir — mirrors the helper used by the skill
    /// tests so the file stays dependency-free (no `tempfile` crate).
    fn tempdir_in_target() -> PathBuf {
        let base =
            std::env::temp_dir().join(format!("agent-memory-rules-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    #[test]
    fn file_or_parent_exists_true_when_file_exists() {
        let tmp = tempdir_in_target();
        let p = tmp.join("file.md");
        std::fs::write(&p, "hello").unwrap();
        assert!(file_or_parent_exists(&p));
    }

    #[test]
    fn file_or_parent_exists_true_when_only_parent_exists() {
        let tmp = tempdir_in_target();
        let p = tmp.join("missing.md");
        // `tmp` exists, but `missing.md` does not — qualifies as installable.
        assert!(file_or_parent_exists(&p));
    }

    #[test]
    fn file_or_parent_exists_false_when_parent_missing() {
        let tmp = tempdir_in_target();
        let p = tmp.join("no-such-dir").join("missing.md");
        assert!(!file_or_parent_exists(&p));
    }

    /// `inject` on a path whose parent exists but whose file is missing
    /// must write a fresh file with the header, not a bare block.
    #[test]
    fn inject_creates_file_with_header_when_missing_but_parent_exists() {
        let tmp = tempdir_in_target();
        let parent = tmp.join(".codex");
        std::fs::create_dir_all(&parent).unwrap();
        let target = parent.join("AGENTS.md");
        assert!(!target.exists());

        let block = build_block();
        let outcome = inject(&target, &block, false).expect("inject should succeed");
        assert!(matches!(outcome, InjectOutcome::Created));
        assert!(target.exists(), "target file should have been created");
        let body = std::fs::read_to_string(&target).unwrap();
        assert!(body.starts_with(FRESH_FILE_HEADER));
        assert!(body.contains(OPEN_MARKER));
        assert!(body.contains("memory context"));
        // No `.bak` sibling on a fresh create — nothing to back up.
        let backup = backup_path(&target);
        assert!(
            !backup.exists(),
            "no .bak should be written when creating a fresh file"
        );
    }

    /// Re-running over a file we just created must behave like any other
    /// augment: the block is replaced in place, NOT duplicated.
    #[test]
    fn inject_is_idempotent_after_create() {
        let tmp = tempdir_in_target();
        let parent = tmp.join(".codex");
        std::fs::create_dir_all(&parent).unwrap();
        let target = parent.join("AGENTS.md");
        let block = build_block();
        inject(&target, &block, false).expect("first inject (create)");
        let outcome2 = inject(&target, &block, false).expect("second inject (replace)");
        assert!(matches!(outcome2, InjectOutcome::Replaced { .. }));
        let body = std::fs::read_to_string(&target).unwrap();
        assert_eq!(body.matches(OPEN_MARKER).count(), 1);
        assert_eq!(body.matches(CLOSE_MARKER).count(), 1);
    }

    /// Parent missing → inject still creates both dir and file (it calls
    /// `create_dir_all` first). This is the fallback for `--target` with a
    /// user-supplied path that points at a non-existent directory — if
    /// they passed it explicitly, they want us to write there.
    #[test]
    fn inject_creates_parent_dir_when_using_explicit_target() {
        let tmp = tempdir_in_target();
        let target = tmp.join("fresh-dir").join("CLAUDE.md");
        let block = build_block();
        let outcome = inject(&target, &block, false).expect("inject should succeed");
        assert!(matches!(outcome, InjectOutcome::Created));
        assert!(target.exists());
    }

    // -- detect_agent_files_with_env ------------------------------------------

    #[test]
    fn resolve_codex_honors_codex_home_override() {
        let tmp = tempdir_in_target();
        let override_dir = tmp.join("custom-codex");
        std::fs::create_dir_all(&override_dir).unwrap();
        let fake_home = tmp.join("fake-home");
        std::fs::create_dir_all(&fake_home).unwrap();
        // Also create both fallback dirs to prove the override wins even
        // when the legacy paths are present.
        std::fs::create_dir_all(fake_home.join(".codex")).unwrap();
        std::fs::create_dir_all(fake_home.join(".config").join("codex")).unwrap();

        let resolved = resolve_codex_rule_file(&fake_home, Some(&override_dir)).unwrap();
        assert_eq!(resolved, override_dir.join("AGENTS.md"));
    }

    #[test]
    fn resolve_codex_prefers_dot_codex_over_xdg_when_both_present() {
        let tmp = tempdir_in_target();
        let fake_home = tmp.join("home");
        std::fs::create_dir_all(fake_home.join(".codex")).unwrap();
        std::fs::create_dir_all(fake_home.join(".config").join("codex")).unwrap();

        let resolved = resolve_codex_rule_file(&fake_home, None).unwrap();
        assert_eq!(resolved, fake_home.join(".codex").join("AGENTS.md"));
    }

    #[test]
    fn resolve_codex_falls_back_to_xdg_when_only_xdg_present() {
        let tmp = tempdir_in_target();
        let fake_home = tmp.join("home");
        std::fs::create_dir_all(&fake_home).unwrap();
        std::fs::create_dir_all(fake_home.join(".config").join("codex")).unwrap();

        let resolved = resolve_codex_rule_file(&fake_home, None).unwrap();
        assert_eq!(
            resolved,
            fake_home.join(".config").join("codex").join("AGENTS.md")
        );
    }

    #[test]
    fn resolve_codex_returns_none_when_no_install_visible() {
        let tmp = tempdir_in_target();
        let fake_home = tmp.join("home");
        std::fs::create_dir_all(&fake_home).unwrap();
        assert!(resolve_codex_rule_file(&fake_home, None).is_none());
    }

    /// An explicit but invalid `CODEX_HOME` must NOT silently fall through
    /// to `~/.codex/` — the user told us exactly where Codex lives, and
    /// writing elsewhere would be surprising.
    #[test]
    fn resolve_codex_override_to_missing_dir_returns_none() {
        let tmp = tempdir_in_target();
        let fake_home = tmp.join("home");
        std::fs::create_dir_all(fake_home.join(".codex")).unwrap();
        let missing = tmp.join("does-not-exist");
        assert!(resolve_codex_rule_file(&fake_home, Some(&missing)).is_none());
    }

    // -- agent labels ---------------------------------------------------------

    #[test]
    fn agent_label_maps_known_rule_file_names() {
        assert_eq!(
            agent_label_for_rule_file(Path::new("/x/.claude/CLAUDE.md")),
            "claude"
        );
        assert_eq!(
            agent_label_for_rule_file(Path::new("/x/.gemini/GEMINI.md")),
            "gemini"
        );
        assert_eq!(
            agent_label_for_rule_file(Path::new("/x/.codex/AGENTS.md")),
            "codex"
        );
        assert_eq!(
            agent_label_for_rule_file(Path::new("/x/custom/FOOBAR.md")),
            "unknown"
        );
    }

    // -- integration: native-memory sync across all three agents --------------
    //
    // The install/remove cycle below drives `sync_auto_memory` directly
    // (rather than spawning the CLI) so the assertions stay fast and
    // deterministic. For each agent we:
    //
    //   1. Stage a fake home with pre-existing unrelated settings/config
    //      content the user might reasonably have.
    //   2. Run the install sync and assert the native-memory-disable key
    //      lands alongside the pre-existing content.
    //   3. Run the remove sync and assert the pre-existing content is
    //      unchanged while the key is gone.
    //
    // Covers Claude (scalar JSON key), Gemini (JSON array membership), and
    // Codex (nested TOML key) in a single exercise.

    #[test]
    fn sync_auto_memory_install_and_remove_cycle_across_all_agents() {
        let home_dir = tempdir_in_target();

        // Stage Claude: user has a theme preference already.
        let claude_dir = home_dir.join(".claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let claude_rule = claude_dir.join("CLAUDE.md");
        std::fs::write(&claude_rule, "# Claude instructions\n").unwrap();
        let claude_settings = claude_dir.join("settings.json");
        std::fs::write(&claude_settings, r#"{"theme": "dark"}"#).unwrap();

        // Stage Gemini: user has a theme and a non-save_memory exclude.
        let gemini_dir = home_dir.join(".gemini");
        std::fs::create_dir_all(&gemini_dir).unwrap();
        let gemini_rule = gemini_dir.join("GEMINI.md");
        std::fs::write(&gemini_rule, "# Gemini instructions\n").unwrap();
        let gemini_settings = gemini_dir.join("settings.json");
        std::fs::write(
            &gemini_settings,
            r#"{"theme": "dark", "excludeTools": ["run_shell_command"]}"#,
        )
        .unwrap();

        // Stage Codex: user has a UI theme set.
        let codex_dir = home_dir.join(".codex");
        std::fs::create_dir_all(&codex_dir).unwrap();
        let codex_rule = codex_dir.join("AGENTS.md");
        std::fs::write(&codex_rule, "# Codex instructions\n").unwrap();
        let codex_config = codex_dir.join("config.toml");
        std::fs::write(&codex_config, "[ui]\ntheme = \"dark\"\n").unwrap();

        // -- install --------------------------------------------------------
        let mut any_failed = false;
        sync_auto_memory(
            &claude_rule,
            AutoMemoryAction::Disable,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &gemini_rule,
            AutoMemoryAction::Disable,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &codex_rule,
            AutoMemoryAction::Disable,
            false,
            &mut any_failed,
        );
        assert!(
            !any_failed,
            "install must not surface any per-agent failures"
        );

        // Claude: autoMemoryEnabled=false added, theme preserved.
        let claude_body = std::fs::read_to_string(&claude_settings).unwrap();
        let claude_parsed: serde_json::Value = serde_json::from_str(&claude_body).unwrap();
        let claude_obj = claude_parsed.as_object().unwrap();
        assert_eq!(claude_obj.get("theme").unwrap(), "dark");
        assert_eq!(
            claude_obj.get("autoMemoryEnabled").unwrap(),
            &serde_json::Value::Bool(false)
        );

        // Gemini: save_memory appended, prior entry + theme preserved.
        let gemini_body = std::fs::read_to_string(&gemini_settings).unwrap();
        let gemini_parsed: serde_json::Value = serde_json::from_str(&gemini_body).unwrap();
        let gemini_obj = gemini_parsed.as_object().unwrap();
        assert_eq!(gemini_obj.get("theme").unwrap(), "dark");
        let gemini_arr = gemini_obj.get("excludeTools").unwrap().as_array().unwrap();
        let gemini_names: Vec<&str> = gemini_arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(gemini_names.contains(&"run_shell_command"));
        assert!(gemini_names.contains(&"save_memory"));

        // Codex: features.memories=false added, [ui] preserved.
        let codex_body = std::fs::read_to_string(&codex_config).unwrap();
        let codex_parsed: toml::value::Table = toml::from_str(&codex_body).unwrap();
        let ui = codex_parsed.get("ui").and_then(|v| v.as_table()).unwrap();
        assert_eq!(ui.get("theme").unwrap().as_str(), Some("dark"));
        let features = codex_parsed
            .get("features")
            .and_then(|v| v.as_table())
            .unwrap();
        assert_eq!(features.get("memories").unwrap().as_bool(), Some(false));

        // -- remove ---------------------------------------------------------
        let mut any_failed = false;
        sync_auto_memory(
            &claude_rule,
            AutoMemoryAction::Remove,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &gemini_rule,
            AutoMemoryAction::Remove,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &codex_rule,
            AutoMemoryAction::Remove,
            false,
            &mut any_failed,
        );
        assert!(
            !any_failed,
            "remove must not surface any per-agent failures"
        );

        // Claude reverts to theme-only.
        let claude_parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&claude_settings).unwrap()).unwrap();
        let claude_obj = claude_parsed.as_object().unwrap();
        assert_eq!(claude_obj.get("theme").unwrap(), "dark");
        assert!(!claude_obj.contains_key("autoMemoryEnabled"));

        // Gemini reverts: save_memory gone, run_shell_command stays.
        let gemini_parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&gemini_settings).unwrap()).unwrap();
        let gemini_obj = gemini_parsed.as_object().unwrap();
        assert_eq!(gemini_obj.get("theme").unwrap(), "dark");
        let gemini_arr = gemini_obj.get("excludeTools").unwrap().as_array().unwrap();
        let gemini_names: Vec<&str> = gemini_arr.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(gemini_names, vec!["run_shell_command"]);

        // Codex reverts: features table gone, [ui] stays.
        let codex_parsed: toml::value::Table =
            toml::from_str(&std::fs::read_to_string(&codex_config).unwrap()).unwrap();
        assert!(
            !codex_parsed.contains_key("features"),
            "empty features table must be dropped on remove"
        );
        let ui = codex_parsed.get("ui").and_then(|v| v.as_table()).unwrap();
        assert_eq!(ui.get("theme").unwrap().as_str(), Some("dark"));
    }

    /// Fresh install case: no pre-existing settings/config at all. The
    /// sync helpers must create the target files from scratch for each
    /// agent and still revert cleanly on remove.
    #[test]
    fn sync_auto_memory_creates_fresh_files_per_agent() {
        let home_dir = tempdir_in_target();

        let claude_dir = home_dir.join(".claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let claude_rule = claude_dir.join("CLAUDE.md");

        let gemini_dir = home_dir.join(".gemini");
        std::fs::create_dir_all(&gemini_dir).unwrap();
        let gemini_rule = gemini_dir.join("GEMINI.md");

        let codex_dir = home_dir.join(".codex");
        std::fs::create_dir_all(&codex_dir).unwrap();
        let codex_rule = codex_dir.join("AGENTS.md");

        let mut any_failed = false;
        sync_auto_memory(
            &claude_rule,
            AutoMemoryAction::Disable,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &gemini_rule,
            AutoMemoryAction::Disable,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &codex_rule,
            AutoMemoryAction::Disable,
            false,
            &mut any_failed,
        );
        assert!(!any_failed);

        assert!(claude_dir.join("settings.json").exists());
        assert!(gemini_dir.join("settings.json").exists());
        assert!(codex_dir.join("config.toml").exists());

        // Remove cycle should collapse each file back to a benign state.
        let mut any_failed = false;
        sync_auto_memory(
            &claude_rule,
            AutoMemoryAction::Remove,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &gemini_rule,
            AutoMemoryAction::Remove,
            false,
            &mut any_failed,
        );
        sync_auto_memory(
            &codex_rule,
            AutoMemoryAction::Remove,
            false,
            &mut any_failed,
        );
        assert!(!any_failed);

        // Claude collapses to `{}` (keeps the file per documented policy).
        let claude_parsed: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(claude_dir.join("settings.json")).unwrap(),
        )
        .unwrap();
        assert!(claude_parsed.as_object().unwrap().is_empty());

        // Gemini likewise collapses to `{}`.
        let gemini_parsed: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(gemini_dir.join("settings.json")).unwrap(),
        )
        .unwrap();
        assert!(gemini_parsed.as_object().unwrap().is_empty());

        // Codex config collapses to an empty document.
        let codex_parsed: toml::value::Table =
            toml::from_str(&std::fs::read_to_string(codex_dir.join("config.toml")).unwrap())
                .unwrap();
        assert!(codex_parsed.is_empty());
    }
}
