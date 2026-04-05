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

Calling the `memory` binary directly is the recommended approach. It is just as fast as MCP mode and avoids the overhead of running a persistent server process. Add the following to your global `CLAUDE.md`, `GEMINI.md`, or equivalent agent instructions:

````markdown
<memory_protocol>
## Memory Operations (MANDATORY)

**Binary:** `/opt/agentic/bin/memory` -- call directly via Bash (do NOT use MCP or skills for memory during normal workflow).

**The "Memory First/Last" Rule:** Every task must begin with a `context` or `search` call and end with a `store` call if functionality changed.

### 1. Pre-Task: Context Retrieval
Before writing a single line of code, run context or search.
- **Goal**: Identify if a similar utility, pattern, or logic exists.
- **Action**: Do not "re-invent." If a similar pattern exists, refactor or extend it.

### 2. Post-Task: Knowledge Persistence
Upon successful completion or refactor, run store.
- **Content**: A concise summary of the functionality, the specific symbols (functions/classes) created, and the "why" behind architectural decisions.
- **Audit-Ready**: Write descriptions that are clear for a future code audit.

### CLI Commands (run via Bash):

```bash
# Search -- semantic/hybrid search
/opt/agentic/bin/memory search "<query>" -k <limit>

# Context -- top-K relevant memories for a task
/opt/agentic/bin/memory context "<task description>" -k <limit> -p "<project>"

# Store -- save a new memory
/opt/agentic/bin/memory store "<content>" -m <type> -t "<tags>" -p "<project>"
# types: user, feedback, project, reference

# Recall -- filter by project/agent/tags/type
/opt/agentic/bin/memory recall -m <type> -t "<tags>" -p "<project>" -k <limit>

# Prune -- decay stale/low-access memories
/opt/agentic/bin/memory prune

# Forget -- remove by ID
/opt/agentic/bin/memory forget <id>
```
</memory_protocol>
````

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

This gives Claude Code six native tools: `memory_store`, `memory_search`, `memory_recall`, `memory_forget`, `memory_prune`, `memory_context`.

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
# Store a memory
memory store "User prefers terse responses" --tags "preference" --project "myapp" -m feedback

# Hybrid search (BM25 + vector)
memory search "how does testing work"

# Filter by project/agent/tags
memory recall --project myapp --memory-type feedback

# Task-relevant context
memory context "refactoring the auth middleware" -k 5

# Delete by ID or search
memory forget --id <uuid>
memory forget --query "outdated preference"

# Clean up stale memories
memory prune --max-age-days 90 --dry-run
memory prune --max-age-days 90

# List all memories
memory list -k 50 --project myapp
```

## MCP tools

| Tool | Purpose |
|------|---------|
| `memory_store` | Save memory with auto-embedding + BM25 indexing |
| `memory_search` | Hybrid BM25 + vector search, returns ranked results |
| `memory_recall` | Filter by project/agent/tags/type |
| `memory_forget` | Remove specific memories |
| `memory_prune` | Decay stale/low-access memories |
| `memory_context` | Return top-K relevant memories for a task description |

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
