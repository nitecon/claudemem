//! Shared glue for bare `memory setup` (interactive checklist) and
//! `memory setup all` (non-interactive sweep).
//!
//! The components are intentionally executed in a fixed order:
//!   1. rules — injects the `<memory-rules>` block into agent rule files
//!   2. skill — installs the Claude Code skill
//!
//! Putting rules before skill keeps humans reading a freshly-updated
//! CLAUDE.md able to see the new rules ahead of the model discovering the
//! skill.

use crate::setup::{rules, skill};
use anyhow::{Context, Result};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    Rules,
    Skill,
}

impl Component {
    pub const ALL: [Component; 2] = [Component::Rules, Component::Skill];

    fn label(self) -> &'static str {
        match self {
            Component::Rules => "Rules",
            Component::Skill => "Skill",
        }
    }
}

/// Entry point for `memory setup` with no subcommand — shows a checklist of
/// the components, their current install state, and lets the user pick which
/// ones to run.
pub fn run_interactive() -> Result<()> {
    let states: Vec<(Component, ComponentState)> =
        Component::ALL.iter().map(|c| (*c, probe(*c))).collect();

    println!("memory setup — choose components to install:");
    println!();
    for (i, (comp, state)) in states.iter().enumerate() {
        let check = if state.installed { "x" } else { " " };
        println!("  {}) [{check}] {} — {}", i + 1, comp.label(), state.detail);
    }
    println!();
    println!("Select components to (re)install:");
    println!("  a     = all");
    println!("  1,2   = specific (comma-separated indices)");
    println!("  c     = cancel");
    print!("> ");
    io::stdout().flush().context("flush stdout")?;

    let mut input = String::new();
    io::stdin()
        .lock()
        .read_line(&mut input)
        .context("read selection")?;
    let chosen = parse_selection(input.trim(), &Component::ALL)?;

    if chosen.is_empty() {
        println!("Cancelled — nothing changed.");
        return Ok(());
    }

    run_components(&chosen)
}

/// Entry point for `memory setup all`. `assume_yes` suppresses the
/// confirmation prompt; useful for scripted installs.
pub fn run_all(assume_yes: bool) -> Result<()> {
    if !assume_yes {
        println!("Will install: rules, skill.");
        print!("Proceed? [y/N]: ");
        io::stdout().flush().context("flush stdout")?;
        let mut input = String::new();
        io::stdin()
            .lock()
            .read_line(&mut input)
            .context("read confirmation")?;
        if !matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            println!("Cancelled.");
            return Ok(());
        }
    }
    run_components(&Component::ALL)
}

// -- internals ---------------------------------------------------------------

struct ComponentState {
    installed: bool,
    detail: String,
}

fn probe(c: Component) -> ComponentState {
    match c {
        Component::Rules => probe_rules(),
        Component::Skill => probe_skill(),
    }
}

fn probe_rules() -> ComponentState {
    let targets = detect_rule_files();
    if targets.is_empty() {
        return ComponentState {
            installed: false,
            detail: "no agent rule files detected".into(),
        };
    }
    let with_block: Vec<String> = targets
        .iter()
        .filter(|p| rules::file_has_rules_block(p))
        .map(|p| p.display().to_string())
        .collect();
    if with_block.is_empty() {
        ComponentState {
            installed: false,
            detail: format!(
                "not injected in {}",
                targets
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    } else {
        ComponentState {
            installed: true,
            detail: format!("injected in {}", with_block.join(", ")),
        }
    }
}

fn probe_skill() -> ComponentState {
    // Probe each known target independently so the user sees which frontends
    // already have the skill and which still need it. A component is flagged
    // "installed" only when *every* target has the file — a half-installed
    // state (e.g. Claude only) should still prompt a re-run to top up the
    // missing side.
    let targets = match skill::skill_targets() {
        Ok(t) => t,
        Err(_) => {
            return ComponentState {
                installed: false,
                detail: "not installed (home directory unavailable)".into(),
            };
        }
    };
    let mut segments: Vec<String> = Vec::with_capacity(targets.len());
    let mut present = 0usize;
    for t in &targets {
        let exists = t.path.exists();
        if exists {
            present += 1;
        }
        segments.push(format!(
            "{} {} ({})",
            if exists { "[x]" } else { "[ ]" },
            t.agent,
            t.path.display()
        ));
    }
    ComponentState {
        installed: present == targets.len(),
        detail: segments.join(", "),
    }
}

/// Surface every rule-file target that `rules::run()` will consider on a
/// real install. We defer to `rules::detect_agent_files()` so the menu's
/// probe and the installer agree on which candidates are visible — a
/// target qualifies when either the file itself exists or the tool's
/// parent directory does, with Codex precedence (CODEX_HOME → `~/.codex/`
/// → `~/.config/codex/`) resolved to a single path.
fn detect_rule_files() -> Vec<PathBuf> {
    rules::detect_agent_files()
}

fn run_components(components: &[Component]) -> Result<()> {
    let mut any_failed = false;
    for c in components {
        println!();
        println!("=== {} ===", c.label());
        let result = match c {
            // Rules: pass `all=true` so detected files are updated without a
            // second interactive prompt — the menu's selection step already
            // got user consent for *which components* to install.
            Component::Rules => rules::run(None, true, false, false, false),
            Component::Skill => skill::run(false, false, false),
        };
        if let Err(e) = result {
            eprintln!("{} failed: {e:#}", c.label());
            any_failed = true;
        }
    }
    if any_failed {
        anyhow::bail!("one or more components failed");
    }
    println!();
    println!("Done.");
    Ok(())
}

/// Parse the freeform selection string. Supports `a`/`all`, `c`/`cancel`, or
/// a comma-separated list of 1-based indices. Out-of-range indices or
/// unparseable tokens bail with a clear error.
fn parse_selection(input: &str, all: &[Component]) -> Result<Vec<Component>> {
    let s = input.trim().to_ascii_lowercase();
    if s.is_empty() || s == "c" || s == "cancel" {
        return Ok(vec![]);
    }
    if s == "a" || s == "all" {
        return Ok(all.to_vec());
    }
    let mut picked = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let idx: usize = tok
            .parse()
            .with_context(|| format!("invalid selection token: {tok:?}"))?;
        if idx < 1 || idx > all.len() {
            anyhow::bail!("selection out of range: {idx}");
        }
        let comp = all[idx - 1];
        if !picked.contains(&comp) {
            picked.push(comp);
        }
    }
    Ok(picked)
}

// -- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_selection_cancel_variants() {
        assert_eq!(parse_selection("", &Component::ALL).unwrap(), vec![]);
        assert_eq!(parse_selection("c", &Component::ALL).unwrap(), vec![]);
        assert_eq!(parse_selection("cancel", &Component::ALL).unwrap(), vec![]);
    }

    #[test]
    fn parse_selection_all() {
        assert_eq!(
            parse_selection("a", &Component::ALL).unwrap(),
            Component::ALL.to_vec()
        );
        assert_eq!(
            parse_selection("ALL", &Component::ALL).unwrap(),
            Component::ALL.to_vec()
        );
    }

    #[test]
    fn parse_selection_specific_indices() {
        let r = parse_selection("1,2", &Component::ALL).unwrap();
        assert_eq!(r, vec![Component::Rules, Component::Skill]);
    }

    #[test]
    fn parse_selection_single_index() {
        let r = parse_selection("2", &Component::ALL).unwrap();
        assert_eq!(r, vec![Component::Skill]);
    }

    #[test]
    fn parse_selection_dedupes() {
        let r = parse_selection("1,1,2,1", &Component::ALL).unwrap();
        assert_eq!(r, vec![Component::Rules, Component::Skill]);
    }

    #[test]
    fn parse_selection_handles_whitespace() {
        let r = parse_selection("  1 , 2  ", &Component::ALL).unwrap();
        assert_eq!(r, vec![Component::Rules, Component::Skill]);
    }

    #[test]
    fn parse_selection_rejects_out_of_range() {
        assert!(parse_selection("3", &Component::ALL).is_err());
        assert!(parse_selection("0", &Component::ALL).is_err());
    }

    #[test]
    fn parse_selection_rejects_garbage() {
        assert!(parse_selection("foo", &Component::ALL).is_err());
        assert!(parse_selection("1,bar", &Component::ALL).is_err());
    }

    #[test]
    fn component_all_order_matches_execution_contract() {
        // Pin the order so future contributors don't accidentally reorder
        // and end up installing the skill before the rules that explain it.
        assert_eq!(Component::ALL, [Component::Rules, Component::Skill]);
    }
}
