//! `memory setup` — installer surface for the agent-memory CLI.
//!
//! Mirrors the layout of `agent-tools setup`: a top-level command that
//! dispatches to small, focused sub-installers. Running `memory setup` with
//! no subcommand shows an interactive checklist of the available components;
//! individual subcommands let scripted/CI flows target a single component.
//!
//! Components:
//!   - `rules` — injects the `<memory-rules>…</memory-rules>` block into
//!     known agent rule files (CLAUDE.md, GEMINI.md, AGENTS.md). Creates
//!     missing rule files when the tool directory exists so fresh installs
//!     are covered, not just established ones.
//!   - `skill` — installs `SKILL.md` to `~/.claude/skills/agent-memory/`
//!     (Claude Code) and `~/.agents/skills/agent-memory/` (cross-agent
//!     alias honored by Gemini CLI and Codex) so every supported frontend
//!     auto-advertises the `memory` CLI to its sessions.
//!
//! The `rules` installer couples to three sibling helpers that disable
//! each agent tool's native memory system in parallel with the rules
//! block, preventing dual-write drift between our SQLite store and the
//! tool's built-in memory surface:
//!
//!   - `settings_json` — Claude Code (`autoMemoryEnabled: false`).
//!   - `gemini_settings_json` — Gemini CLI (`save_memory` in `excludeTools`).
//!   - `codex_config_toml` — Codex ("Chronicle" via `[features] memories = false`).
//!
//! The module only re-exports the public entry points used by `cli.rs`; the
//! implementation details (markers, body templates, probe helpers) stay
//! private to their respective submodules.

pub mod codex_config_toml;
pub mod gemini_settings_json;
pub mod menu;
pub mod rules;
pub mod settings_json;
pub mod skill;
