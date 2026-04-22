# agent-memory

Persistent hybrid-search memory system for AI coding agents. Replaces markdown-based memory with selective retrieval, cross-project search, and agent-scoped memories that scale without context cost.

## Architecture

- **SQLite** -- single-file backing store, portable, zero config
- **fastembed-rs** -- local embeddings via all-MiniLM-L6-v2 ONNX model (semantic similarity, no API calls)
- **Hybrid ranking** -- BM25 (FTS5) + cosine similarity combined via Reciprocal Rank Fusion (RRF)
- **MCP server** -- stdio JSON-RPC server for native Claude Code tool integration
- **CLI** -- direct command-line interface for humans, scripts, and AI agents

## Install

### Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/nitecon/agent-memory/refs/heads/main/install.sh | sudo bash
```

Installs to `/opt/agentic/bin/memory` with a symlink at `/usr/local/bin/memory`.

### Windows (PowerShell as Administrator)

```powershell
irm https://raw.githubusercontent.com/nitecon/agent-memory/refs/heads/main/install.ps1 | iex
```

Installs to `%USERPROFILE%\.agentic\bin\memory.exe` and adds it to your PATH.

### From source

```bash
cargo build --release
```

Binary: `target/release/memory` (Linux/macOS) or `target/release/memory.exe` (Windows)

First run downloads the embedding model (~80MB, cached alongside the database).

## Database location

The database path is resolved in this order:

| Priority | Condition | Path |
|----------|-----------|------|
| 1 | `AGENT_MEMORY_DIR` env var is set | `$AGENT_MEMORY_DIR/memory.db` |
| 2 | `~/.agentic/memory.db` exists | `~/.agentic/memory.db` (user-local) |
| 3 | Default (Linux/macOS) | `/opt/agentic/memory.db` (global) |
| 3 | Default (Windows) | `%USERPROFILE%\.agentic\memory.db` |

The model cache and any auxiliary data are stored alongside the database in the same directory.

This means on a shared Linux/macOS machine, all agents share `/opt/agentic/memory.db` by default. If you need per-user isolation, create `~/.agentic/memory.db` (even an empty file will trigger the user-local path) or set `AGENT_MEMORY_DIR`.

## Recommended usage: CLI-first

Calling the `memory` binary directly is the recommended approach. It is just as fast as MCP mode and avoids the overhead of running a persistent server process. The fastest way to teach your agent to use it is the `memory setup` command — it bundles an interactive checklist that injects the rules block into your agent rule files and installs a Claude Code skill that auto-advertises the CLI to every session.

### Auto-install the agent protocol

`memory setup` is now a small subcommand family:

| Command | Behavior |
|---------|----------|
| `memory setup` | Interactive checklist: shows the install state of each component (rules, skill) and lets you pick which to (re)install |
| `memory setup rules [flags]` | Inject the `<memory-rules>` block into known agent rule files (CLAUDE.md, GEMINI.md, AGENTS.md) |
| `memory setup skill [flags]` | Install `~/.claude/skills/agent-memory/SKILL.md` so Claude Code auto-loads a ~100-token description that nudges the model toward the CLI |
| `memory setup all [-y]` | Run rules → skill non-interactively (use `-y` / `--yes` to skip confirmation) |

```bash
# Bare invocation: 2-item interactive checklist (rules + skill).
memory setup

# Rules only — detects ~/.claude/CLAUDE.md, ~/.gemini/GEMINI.md,
# ~/.codex/AGENTS.md, ~/.config/codex/AGENTS.md.
memory setup rules               # detect + prompt
memory setup rules --all         # update every detected file
memory setup rules --target ~/.claude/CLAUDE.md
memory setup rules --dry-run     # preview, don't write
memory setup rules --print       # emit just the <memory-rules> block

# Skill only — installs to ~/.claude/skills/agent-memory/SKILL.md.
memory setup skill
memory setup skill --dry-run
memory setup skill --print

# Everything, scripted.
memory setup all --yes
```

`memory setup rules` writes a `<memory-rules>…</memory-rules>` block (loose-XML markers so it is easy to locate and update) and saves a `.bak` sibling before each modification. Re-running replaces the block in place — your agent rule files never accumulate duplicates. If the companion [`agent-tools setup rules`](https://github.com/nitecon/agent-tools) block (`<agent-tools-rules>…</agent-tools-rules>`) is already present in the file, the memory block is inserted directly after it so the two protocols stay grouped at the top; otherwise it is prepended.

`memory setup skill` writes a single SKILL.md whose frontmatter `description` is always loaded into Claude Code sessions (~100 tokens), pulling the model toward `memory context` at task start and `memory store` at task end. The full body only loads on demand when the skill is picked.

### Manual install (equivalent content)

If you'd rather paste the block yourself, add the following to your global `CLAUDE.md`, `GEMINI.md`, or equivalent agent instructions:

````markdown
<memory-rules>
## Agent Memory -- Mandatory Protocols

### Memory Operations (MANDATORY)

**Binary:** `memory` (installed at `/opt/agentic/bin/memory` on Linux/macOS, `%USERPROFILE%\.agentic\bin\memory.exe` on Windows) -- call directly via Bash. Do NOT use MCP or skills for memory during normal workflow.

**The "Memory First/Last" Rule:** Every task must begin with a `context` or `search` call and end with a `store` call if functionality changed.

```bash
# Context -- top-K relevant memories for a task (boost cwd project)
memory context "<task description>" -k <limit>

# Search -- hybrid BM25 + vector search (boost cwd project)
memory search "<query>" -k <limit>

# Store -- save a new memory (project auto-detected)
memory store "<content>" -m <type> -t "<tags>"
# types: user, feedback, project, reference

# Get -- fetch full content for specific IDs (pair with brief search)
memory get <uuid> [<uuid>...]

# Recall -- filter by project/agent/tags/type
memory recall -m <type> -t "<tags>" -p "<project>" -k <limit>

# Projects -- list distinct project idents (spot alias mismatches)
memory projects

# Move -- reassign the project ident on one or many memories
memory move --from "<old>" --to "<new>" [--dry-run]

# Copy -- duplicate memories under a new project ident
memory copy --from "<old>" --to "<new>" [--dry-run]

# Forget -- remove a memory by ID (or by search query)
memory forget --id <uuid>

# Prune -- decay stale/low-access memories
memory prune --max-age-days 90 [--dry-run]
```
</memory-rules>
````

Prefer `memory setup` over hand-pasting — it keeps the block up-to-date with the latest CLI surface and guarantees the markers match what future re-runs look for.

## MCP server (optional)

If you prefer MCP integration, register the server:

```bash
claude mcp add agent-memory -- /opt/agentic/bin/memory serve
```

Or add manually to `~/.claude.json`:

```json
{
  "mcpServers": {
    "agent-memory": {
      "type": "stdio",
      "command": "/opt/agentic/bin/memory",
      "args": ["serve"]
    }
  }
}
```

This gives Claude Code ten native tools: `memory_store`, `memory_search`, `memory_recall`, `memory_forget`, `memory_prune`, `memory_context`, `memory_get`, `memory_projects`, `memory_move`, `memory_copy`.

### Skills (optional)

Copy the skill directories to your Claude Code skills location:

```bash
# Personal skills (available in all projects)
cp -r skills/remember ~/.claude/skills/remember
cp -r skills/recall ~/.claude/skills/recall
```

This enables `/remember` and `/recall` slash commands.

## CLI reference

```bash
# Store a memory (project auto-detected from cwd's git remote)
memory store "User prefers terse responses" --tags "preference" -m feedback

# Hybrid search (BM25 + vector); brief output is the default, cwd project is boosted
memory search "how does testing work"

# Fetch full content for specific hits (two-stage retrieval)
memory get <uuid> <uuid>

# Filter by project/agent/tags
memory recall --project myapp --memory-type feedback

# Task-relevant context
memory context "refactoring the auth middleware" -k 5

# Hard filter vs boost
memory search "storage" --only "github.com/acme/infra.git"   # only this project
memory search "storage" --no-project-boost                    # flat ranking, no boost

# Full content instead of preview
memory search "storage" --format full

# Delete by ID or search
memory forget --id <uuid>
memory forget --query "outdated preference"

# Clean up stale memories
memory prune --max-age-days 90 --dry-run
memory prune --max-age-days 90

# List all memories
memory list -k 50 --project myapp

# List distinct project idents (great for spotting alias mismatches)
memory projects

# Migrate memories from a legacy project name to the canonical git-remote ident
memory move --from "trading-platform-sre" --to "github.com/nitecon/SRE.git" --dry-run
memory move --from "trading-platform-sre" --to "github.com/nitecon/SRE.git"

# Reassign a single memory by ID (pass --to "" to clear the project tag)
memory move --id <uuid> --to "github.com/nitecon/SRE.git"
memory move --id <uuid> --to ""

# Duplicate memories under a new project ident (preserves content + embedding)
memory copy --from "github.com/acme/mono.git" --to "github.com/acme/split.git"
memory copy --id <uuid> --to "github.com/acme/mirror.git"

# Check for updates and install the latest version
memory update

# Setup family — interactive checklist + per-component subcommands
memory setup                              # interactive: pick rules and/or skill
memory setup rules                        # rules only: detect + prompt
memory setup rules --all                  # rules: update every detected file
memory setup rules --target ~/.claude/CLAUDE.md
memory setup rules --dry-run              # rules: preview, don't write
memory setup rules --print                # rules: print <memory-rules> block
memory setup skill                        # install ~/.claude/skills/agent-memory/SKILL.md
memory setup skill --dry-run              # skill: preview SKILL.md
memory setup skill --print                # skill: print SKILL.md to stdout
memory setup all --yes                    # rules → skill, non-interactive
```

## Project auto-detection and cross-project boost

`store`, `search`, and `context` derive the current project identifier from the working directory's git remote and reduce it to the repository shortname (e.g. `git@github.com:nitecon/eventic.git` → `eventic`). SSH and HTTPS for the same repo produce the same ident. Non-git directories fall back to the directory basename. New memories are auto-tagged with this project unless you pass `--project` explicitly or `--no-project`.

Shortname is deliberate so auto-derived idents match the hand-written shortnames most agents already use. The trade-off is that two repos with the same basename across different orgs will collide; in that case, tag them explicitly with `--project`.

At query time, memories tagged with the current project receive a 1.5× score boost before re-sorting. The goal is to surface local context first while letting strong cross-project matches still appear as prior art. When the top-K contains cross-project results, the response includes a `hint` field that flags them so models treat those memories as general guidance rather than direct context.

| Flag | Behavior |
|------|----------|
| (none) | Boost cwd-derived project; cross-project results can still surface |
| `-p <ident>` | Boost this project instead of cwd |
| `--only <ident>` | Hard filter: only return memories with this project |
| `--no-project-boost` | Flat ranking; no boost, no filter |

## Migrating project idents

If memories were stored under a legacy project name (e.g. a logical label like `trading-platform-sre`) but the cwd-resolver now returns the canonical git-remote ident (e.g. `github.com/nitecon/SRE.git`), search will treat them as cross-project and the `hint` field will undersell their relevance. Fix it by consolidating idents:

```bash
# 1. Inspect the distinct project idents in the database
memory projects

# 2. Preview the affected memories before writing
memory move --from "trading-platform-sre" --to "github.com/nitecon/SRE.git" --dry-run

# 3. Apply the rename
memory move --from "trading-platform-sre" --to "github.com/nitecon/SRE.git"
```

Use `memory copy` instead of `memory move` when you want the memory available under *both* idents — for example, when a shared memory applies to two forks of the same codebase. Copies keep the original content, tags, and cached embedding; only the project ident, UUID, and timestamps differ.

## Output formats

`search` and `context` return a wrapper object with a `hint` when cross-project results are present:

```json
{
  "results": [ ... ],
  "current_project": "github.com/acme/myapp.git",
  "cross_project_count": 2,
  "hint": "2 of 5 results are cross-project (is_current_project=false). Use those as prior-art or general guidance, not direct context for 'github.com/acme/myapp.git'. Use `memory get <id>` for full content."
}
```

Default format (`--format brief`) returns `id`, `tags`, `project`, `memory_type`, `match_quality` (high/medium/low), `is_current_project`, a `preview` (160 chars), and `content_len`. Use `--format full` when you need the full content, or pair `search --brief` with `memory get <id>` for a cheap two-stage retrieval.

## Auto-update

The binary checks for new releases on GitHub once per hour (at most) during normal CLI usage. If a newer version is found, it downloads and replaces the binary automatically. The update check is non-blocking — failures are logged to stderr and never interrupt normal operation.

To disable auto-updates, set the environment variable:

```bash
export AGENT_MEMORY_NO_UPDATE=1
```

You can also trigger an update manually at any time with `memory update`.

## MCP tools

| Tool | Purpose |
|------|---------|
| `memory_store` | Save memory with auto-embedding + BM25 indexing |
| `memory_search` | Hybrid BM25 + vector search, returns ranked results |
| `memory_recall` | Filter by project/agent/tags/type |
| `memory_forget` | Remove specific memories |
| `memory_prune` | Decay stale/low-access memories |
| `memory_context` | Return top-K relevant memories for a task description |
| `memory_get` | Fetch full content for one or more memory IDs (pair with brief search) |
| `memory_projects` | List distinct project idents with memory counts (spot alias mismatches) |
| `memory_move` | Reassign the project ident on one memory (by id) or in bulk (by from/to) |
| `memory_copy` | Duplicate memories under a new project ident; preserves content + embedding |

## Memory types

| Type | Purpose |
|------|---------|
| `user` | Facts about the user -- role, preferences, expertise |
| `feedback` | How to approach work -- corrections and confirmed approaches |
| `project` | Ongoing work context -- decisions, deadlines, constraints |
| `reference` | Pointers to external resources -- URLs, dashboards, systems |

## How search works

Every query runs through two retrieval paths simultaneously:

1. **BM25** (FTS5) -- term-frequency keyword matching, great for exact names and patterns
2. **Vector** (fastembed cosine similarity) -- semantic similarity, great for "I vaguely remember something about..."

Results are combined via **Reciprocal Rank Fusion** (k=60), which merges ranked lists without requiring score normalization. A memory that ranks well in both paths gets a strong combined score.

## Design decisions

- **SQLite is the source of truth.** FTS5 handles full-text indexing within the same database file.
- **Embeddings are brute-force cosine.** For a personal memory system (<100K memories), this is fast enough and avoids ANN index complexity.
- **Model loads lazily.** Commands that don't need embeddings (e.g., `recall`, `forget --id`) skip the ~200ms model load.
- **Access counts track usage.** Every retrieval increments `access_count`, enabling `prune` to identify stale memories.
- **All logging goes to stderr.** Stdout is reserved for JSON results (CLI) or JSON-RPC (MCP), so logging never pollutes the transport.
- **Global-first storage.** `/opt/agentic/memory.db` is shared across all users/agents by default, with `~/.agentic/` as a user-local override.
