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

### Scope tiers

Every memory is stored under one of two scopes; retrieval boosts both:

| Scope                      | Boost  | When to use                                    |
|----------------------------|--------|------------------------------------------------|
| **Current project** (cwd)  | 1.5×   | Repo-specific decisions, patterns, bugs        |
| **Global** (`__global__`)  | 1.25×  | Universal user preferences / directives        |
| Other project              | 1.0×   | Surfaces only as prior art via the `hint` field |

`store`, `search`, and `context` auto-detect the current project from the cwd's git remote. A single `context` call returns both current-project and global hits — no second query needed.

```bash
# Context -- top-K relevant memories for a task (boost cwd + global)
memory context "<task description>" -k <limit>

# Search -- hybrid BM25 + vector search (boost cwd + global)
memory search "<query>" -k <limit>

# Store -- save a new project-scoped memory (cwd auto-detected)
memory store "<content>" -m <type> -t "<tags>"

# Store -- save a universal preference (applies across every repo)
memory store "<content>" -m <type> --scope global -t "<tags>"
# types: user, feedback, project, reference

# Get -- fetch full content for specific IDs (pair with search for two-stage flow)
memory get <id> [<id>...]                 # 8-char short prefix OK

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

### Rule A -- Pre-action behavior recall (MANDATORY)

Before starting any user-requested task, run one `memory context "<task>"` call first. A single call returns both global directives (1.25× boost) and project-specific directives (1.5× boost). Do not skip for "quick" tasks: directives the user has already stated must never need to be re-stated. If the `hint` field flags zero global-scope matches, pause and reflect — or ask before acting.

### Rule B -- Post-action scope classification (MANDATORY)

After completing an action, if the user stated or implied any directive, preference, or corrective rule during the session, you MUST store it and MUST classify its scope:

- **Global** (`--scope global`) -- universal preference. Signals: "I always", "I never", "from now on", "I prefer", "don't ever", "whenever we", "in general".
- **Project** (`--scope project`, the default) -- specific to this repo, service, or codebase. Signals: "in this repo", "for this service", "here we".
- **Ambiguous** -- phrasing could reasonably apply either way. You MUST ask the user before storing. Do not silently default.

Example:
```bash
memory store "User never wants PRs opened unless they explicitly ask" \
  -m feedback --scope global -t "workflow,pr"
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

# Store a universal preference (applies across every repo, 1.25× retrieval boost)
memory store "User never wants PRs opened unless explicitly asked" \
  -m feedback --scope global --tags "workflow,pr"

# Hybrid search (BM25 + vector); cwd project is boosted
memory search "how does testing work"

# Fetch full content for specific hits (two-stage retrieval)
# Short 8-char prefix is fine — resolves via `resolve_id_prefix`.
memory get 4c82c482
memory get <uuid> <uuid>

# Filter by project/agent/tags
memory recall --project myapp --memory-type feedback

# Task-relevant context
memory context "refactoring the auth middleware" -k 5

# Hard filter vs boost
memory search "storage" --only "github.com/acme/infra.git"   # only this project
memory search "storage" --no-project-boost                    # flat ranking, no boost

# Delete by ID (short prefix supported) or by search
memory forget --id 4c82c482
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

## Project & global scope tiers

`store`, `search`, and `context` derive the current project identifier from the working directory's git remote and reduce it to the repository shortname (e.g. `git@github.com:nitecon/eventic.git` → `eventic`). SSH and HTTPS for the same repo produce the same ident. Non-git directories fall back to the directory basename. New memories are auto-tagged with this project unless you pass `--project` explicitly, `--no-project`, or `--scope global`.

Shortname is deliberate so auto-derived idents match the hand-written shortnames most agents already use. The trade-off is that two repos with the same basename across different orgs will collide; in that case, tag them explicitly with `--project`.

Retrieval applies two independent score boosts:

| Scope | Boost | Meaning |
|-------|-------|---------|
| Current project (`project == cwd`) | **1.5×** | Local context — highest priority |
| Global (`project == "__global__"`) | **1.25×** | Universal user preferences — surface in every repo |
| Other project | 1.0× | Cross-project prior art; flagged via the `hint` field |

A single `context` or `search` call returns hits from all three tiers; the response's `cross_project_count`, `global_scope_count`, and `hint` fields tell models how to weigh them. Strong cross-project matches can still out-rank weak current-project hits — the boosts tilt ties without hard-filtering prior art.

### Global scope

Global-scoped memories are stored under the reserved sentinel project ident `__global__`. Users opt in with `--scope global` on `memory store`:

```bash
memory store "Never open a PR unless explicitly asked" \
  -m feedback --scope global --tags "workflow,pr"
```

The sentinel is reserved: passing `--project __global__` directly (or `memory move --to __global__`) is rejected with a clear error pointing users to `--scope global`. This keeps the sentinel load-bearing for retrieval behavior rather than a string users can accidentally collide into. When you run `memory projects`, `__global__` shows up as its own row so you can see how many universal preferences are on file.

### Search flags

| Flag | Behavior |
|------|----------|
| (none) | Boost cwd project (1.5×) **and** global sentinel (1.25×); cross-project results still surface |
| `-p <ident>` | Boost this project (1.5×) instead of cwd; global boost unchanged |
| `--only <ident>` | Hard filter: only return memories with this project |
| `--no-project-boost` | Flat ranking; disables **both** boosts |

### Store flags

| Flag | Behavior |
|------|----------|
| (none) | Project scope; `project` auto-detected from cwd |
| `--scope project` | Explicit project scope; suppresses the reflection hint even on `user`/`feedback` stores |
| `--scope global` | Global scope; stores under the `__global__` sentinel |
| `--project <ident>` | Override the project ident (must NOT equal `__global__`) |
| `--no-project` | Store with no project tag (skips cwd auto-detect) |

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

## Output format (light-XML)

All commands emit **light-XML** — grouped section tags with numbered content lines. No JSON. The shape is compact on purpose: tags give the agent a structural signal while the payload stays plain lines so token overhead is minimal.

### `context` / `search` / `recall`

```
<project_memories>
1. PRs required by CodingGuidelines.md [git,standards] (ID:4c82c482)
2. Follow docs/CodingGuidelines.md for PRs [git,standards] (ID:772fd580)
</project_memories>
<general_knowledge>
1. User avoids PRs unless required [git,standards] (ID:372bd79d)
</general_knowledge>
<other_projects>
1. colorithmic: k-means Euclidean beats OKLab on OPT [quantization] (ID:23d0142a)
</other_projects>
<hint>2 of 4 results are global-scope preferences (apply across all projects). Treat them as directives, not suggestions.</hint>
```

- `<project_memories>` — hits tagged with the current (cwd-derived) project.
- `<general_knowledge>` — hits tagged with the `__global__` sentinel (universal preferences, 1.25× boost).
- `<other_projects>` — hits from other projects, prefixed with the originating project ident. Treat as prior art.
- `<hint>` — reflection / directive prompt. Only emitted when it has something to say.

Empty sections are elided. A query with zero global-scope hits during a scoped retrieval triggers a reflection-style `<hint>` nudging the agent to confirm no universal preference applies before acting.

### Mutations (`store` / `move` / `copy` / `forget` / `prune`)

Single self-closing `<result>` line:

```
<result status="stored" id="a4936eff" scope="global" project="__global__"/>
<result status="forgot" id="a4936eff"/>
<result status="forgot" count="3"/>
<result status="no_matches"/>
<result status="pruned" count="7"/>
<result status="dry_run" count="7"/>
<result status="moved" id="a4936eff" to_project="github.com/acme/split.git"/>
```

`memory store` with memory type `user` or `feedback` and no explicit `--scope` gets one additional `<hint>…</hint>` line reminding the caller to reclassify to global if the memory applies across repos.

### `memory get`

```
<memory id="a4936eff" project="agent-memory" type="feedback" tags="workflow,pr">
User never wants PRs opened unless they explicitly ask.
</memory>
```

Full content is emitted verbatim as element text (XML-escaped). IDs are shown as 8-char short prefixes everywhere — full UUIDs still work as input.

### Short-ID resolution

Every command that takes an `<id>` accepts any prefix of 4 or more hex characters. `memory get 4c82c482` expands to the full UUID when unique. When two memories share the same prefix, an `<ambiguous>` block lists the candidates:

```
<ambiguous prefix="4c82c482">
1. 4c82c482-c081-4937... [colorithmic,milestone]: colorithmic v1.0.0 milestone 2026-04-20...
2. 4c82c482-d7f2-4a18... [agent-memory,schema]: Schema v3 migration design notes...
Reply with 1..2, or re-run with a longer prefix.
</ambiguous>
```

The fast-path full-UUID lookup still works — prefix resolution is additive.

### `memory list` / `memory projects`

Plain light-XML blocks optimized for readability:

```
<memories count="2">
1.*(feedback) agent-memory [workflow,pr] (ID:a4936eff): User never wants PRs opened unless they explicitly ask.
2. (user) colorithmic [setup] (ID:b12c3d4e): Prefer k-means Euclidean over OKLab for OPT quantization.
</memories>
```

```
<projects count="3">
*agent-memory (42)
 colorithmic (7)
 __global__ (3)
</projects>
```

A leading `*` marks the current cwd-derived project. An empty list collapses to a self-closing `<memories count="0"/>` or `<projects count="0"/>`.

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
| `memory_get` | Fetch full content for one or more memory IDs (full UUID or 4+ char short prefix) |
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
- **All logging goes to stderr.** Stdout is reserved for light-XML results (CLI) or JSON-RPC transport (MCP), so logging never pollutes either channel. MCP tool responses themselves are light-XML strings delivered as a single text content block.
- **Global-first storage.** `/opt/agentic/memory.db` is shared across all users/agents by default, with `~/.agentic/` as a user-local override.
