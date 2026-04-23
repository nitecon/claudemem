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

use anyhow::{Context, Result};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

const OPEN_MARKER: &str = "<memory-rules>";
const CLOSE_MARKER: &str = "</memory-rules>";

const SIBLING_OPEN: &str = "<agent-tools-rules>";
const SIBLING_CLOSE: &str = "</agent-tools-rules>";

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
pub fn run(target: Option<PathBuf>, all: bool, dry_run: bool, print: bool) -> Result<()> {
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
        match inject(path, &block, dry_run) {
            Ok(InjectOutcome::DryRun(preview)) => {
                println!("--- DRY RUN: {} ---", path.display());
                print!("{preview}");
                println!("--- end preview ---");
            }
            Ok(InjectOutcome::Replaced { backup }) => {
                println!("Updated existing block in {}", path.display());
                println!("  backup: {}", backup.display());
            }
            Ok(InjectOutcome::InsertedAfterSibling { backup }) => {
                println!(
                    "Inserted new block after <agent-tools-rules> in {}",
                    path.display()
                );
                println!("  backup: {}", backup.display());
            }
            Ok(InjectOutcome::Prepended { backup }) => {
                println!("Prepended new block to {}", path.display());
                println!("  backup: {}", backup.display());
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
pub(crate) fn detect_agent_files() -> Vec<PathBuf> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return vec![],
    };
    let candidates = [
        home.join(".claude").join("CLAUDE.md"),
        home.join(".gemini").join("GEMINI.md"),
        home.join(".codex").join("AGENTS.md"),
        home.join(".config").join("codex").join("AGENTS.md"),
    ];
    candidates.into_iter().filter(|p| p.exists()).collect()
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
    Replaced { backup: PathBuf },
    InsertedAfterSibling { backup: PathBuf },
    Prepended { backup: PathBuf },
}

fn inject(path: &Path, block: &str, dry_run: bool) -> Result<InjectOutcome> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let already_present = existing.contains(OPEN_MARKER) && existing.contains(CLOSE_MARKER);
    let has_sibling = existing.contains(SIBLING_OPEN) && existing.contains(SIBLING_CLOSE);
    let (new_content, mode) = compute_new_content(&existing, block, already_present, has_sibling);

    if dry_run {
        return Ok(InjectOutcome::DryRun(new_content));
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent of {}", path.display()))?;
    }

    let backup = backup_path(path);
    std::fs::write(&backup, &existing)
        .with_context(|| format!("write backup to {}", backup.display()))?;
    std::fs::write(path, &new_content)
        .with_context(|| format!("write updated file {}", path.display()))?;

    Ok(match mode {
        InjectMode::Replaced => InjectOutcome::Replaced { backup },
        InjectMode::InsertedAfterSibling => InjectOutcome::InsertedAfterSibling { backup },
        InjectMode::Prepended => InjectOutcome::Prepended { backup },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InjectMode {
    Replaced,
    InsertedAfterSibling,
    Prepended,
}

/// Pure helper: build the new file body from existing content + the rules
/// block. Factored out so unit tests can verify idempotency and placement
/// without touching the filesystem.
fn compute_new_content(
    existing: &str,
    block: &str,
    already_present: bool,
    has_sibling: bool,
) -> (String, InjectMode) {
    if already_present {
        (replace_block(existing, block), InjectMode::Replaced)
    } else if existing.is_empty() {
        (block.to_string(), InjectMode::Prepended)
    } else if has_sibling {
        (insert_after_sibling(existing, block), InjectMode::InsertedAfterSibling)
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
    print!("Update [a]ll, [1-{}] specific, [c]ancel: ", candidates.len());
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
        assert!(b.contains("1.5×"), "block must reference the 1.5× project boost");
        assert!(b.contains("1.25×"), "block must reference the 1.25× global boost");
        // The sentinel must be named so users inspecting the DB aren't
        // confused by stray `__global__` rows.
        assert!(b.contains("__global__"), "block must name the sentinel project ident");
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
        let (out, mode) = compute_new_content(existing, &block, false, false);
        assert_eq!(mode, InjectMode::Prepended);
        assert!(out.starts_with(OPEN_MARKER));
        assert!(out.contains("# Existing instructions"));
        assert!(out.contains(&format!("{CLOSE_MARKER}\n\n# Existing instructions")));
    }

    #[test]
    fn compute_new_content_inserts_after_sibling_when_present() {
        let block = build_block();
        let existing = format!(
            "{SIBLING_OPEN}\nagent-tools stuff\n{SIBLING_CLOSE}\n\n# Rest of file\n"
        );
        let (out, mode) = compute_new_content(&existing, &block, false, true);
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
        let (out, mode) = compute_new_content(&existing, &block, true, false);
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
        let (once, _) = compute_new_content(existing, &block, false, false);
        let (twice, mode) = compute_new_content(&once, &block, true, false);
        assert_eq!(mode, InjectMode::Replaced);
        assert_eq!(once, twice);
    }

    #[test]
    fn compute_new_content_handles_empty_file() {
        let block = build_block();
        let (out, mode) = compute_new_content("", &block, false, false);
        assert_eq!(mode, InjectMode::Prepended);
        assert_eq!(out, block);
    }

    #[test]
    fn re_running_after_sibling_added_keeps_memory_block_once() {
        // User first runs `memory setup` on a fresh file (prepends).
        // Later they run `agent-tools setup rules` (prepends its own block).
        // Third run of `memory setup` detects its own existing block and
        // replaces in place, *not* re-inserting after the sibling.
        let block = build_block();
        let (first, _) = compute_new_content("# Header\n", &block, false, false);
        let with_sibling =
            format!("{SIBLING_OPEN}\nfoo\n{SIBLING_CLOSE}\n\n{first}");
        let (after_refresh, mode) = compute_new_content(&with_sibling, &block, true, true);
        assert_eq!(mode, InjectMode::Replaced);
        assert_eq!(after_refresh.matches(OPEN_MARKER).count(), 1);
        assert_eq!(after_refresh.matches(CLOSE_MARKER).count(), 1);
    }

    #[test]
    fn backup_path_appends_bak() {
        let p = PathBuf::from("/tmp/foo/CLAUDE.md");
        assert_eq!(backup_path(&p), PathBuf::from("/tmp/foo/CLAUDE.md.bak"));
    }
}
