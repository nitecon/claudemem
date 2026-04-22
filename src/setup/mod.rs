//! `memory setup` — installer surface for the agent-memory CLI.
//!
//! Mirrors the layout of `agent-tools setup`: a top-level command that
//! dispatches to small, focused sub-installers. Running `memory setup` with
//! no subcommand shows an interactive checklist of the available components;
//! individual subcommands let scripted/CI flows target a single component.
//!
//! Components:
//!   - `rules` — injects the `<memory-rules>…</memory-rules>` block into
//!              known agent rule files (CLAUDE.md, GEMINI.md, AGENTS.md).
//!   - `skill` — installs `~/.claude/skills/agent-memory/SKILL.md` so
//!              Claude Code sessions are nudged toward using the CLI.
//!
//! The module only re-exports the public entry points used by `cli.rs`; the
//! implementation details (markers, body templates, probe helpers) stay
//! private to their respective submodules.

pub mod menu;
pub mod rules;
pub mod skill;
