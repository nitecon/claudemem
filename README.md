# claude-memory

Persistent hybrid-search memory system for Claude Code. Replaces markdown-based memory with selective retrieval, cross-project search, and agent-scoped memories that scale without context cost.

## Architecture

- **SQLite** — single-file backing store, portable, zero config
- **Tantivy** — BM25 full-text search (file paths, function names, exact patterns)
- **fastembed-rs** — local embeddings via all-MiniLM-L6-v2 ONNX model (semantic similarity, no API calls)
- **Hybrid ranking** — BM25 + cosine similarity combined via Reciprocal Rank Fusion (RRF)
- **MCP server** — stdio JSON-RPC server for native Claude Code tool integration
- **CLI** — direct command-line interface for humans and scripts

## Build

```bash
cargo build --release
```

Binary: `target/release/claude-memory.exe` (~33MB)

First run downloads the embedding model (~80MB, cached in `~/.claude/memory/models/`).

## Setup

### 1. Register the MCP server

```bash
claude mcp add claude-memory -- /path/to/claude-memory serve
```

Or add manually to `~/.claude.json`:

```json
{
  "mcpServers": {
    "claude-memory": {
      "type": "stdio",
      "command": "/path/to/claude-memory",
      "args": ["serve"]
    }
  }
}
```

This gives Claude Code six native tools: `memory_store`, `memory_search`, `memory_recall`, `memory_forget`, `memory_prune`, `memory_context`.

### 2. Install skills (optional)

Copy the skill directories to your Claude Code skills location:

```bash
# Personal skills (available in all projects)
cp -r skills/remember ~/.claude/skills/remember
cp -r skills/recall ~/.claude/skills/recall

# Or project-level skills (this project only)
cp -r skills/remember .claude/skills/remember
cp -r skills/recall .claude/skills/recall
```

This enables `/remember` and `/recall` slash commands that wrap the MCP tools with natural language handling.

## Usage

### CLI

```bash
# Store a memory
claude-memory store "User prefers terse responses" --tags "preference" --project "myapp" -m feedback

# Hybrid search (BM25 + vector)
claude-memory search "how does testing work"

# Filter by project/agent/tags
claude-memory recall --project myapp --memory-type feedback

# Task-relevant context
claude-memory context "refactoring the auth middleware" -k 5

# Delete by ID or search
claude-memory forget --id <uuid>
claude-memory forget --query "outdated preference"

# Clean up stale memories
claude-memory prune --max-age-days 90 --dry-run
claude-memory prune --max-age-days 90
```

### Skills

```
/remember the user prefers single bundled PRs for refactors in this area
/recall testing preferences
```

### MCP tools (used by Claude directly)

When registered as an MCP server, Claude Code can call these tools natively:

| Tool | Purpose |
|------|---------|
| `memory_store` | Save memory with auto-embedding + BM25 indexing |
| `memory_search` | Hybrid BM25 + vector search, returns ranked results |
| `memory_recall` | Filter by project/agent/tags/type |
| `memory_forget` | Remove specific memories |
| `memory_prune` | Decay stale/low-access memories |
| `memory_context` | Return top-K relevant memories for a task description |

## Storage

All data lives in `~/.claude/memory/`:

```
~/.claude/memory/
  memory.db      # SQLite database (source of truth)
  tantivy/       # BM25 search index (rebuildable from SQLite)
  models/        # ONNX model cache (downloaded on first use)
```

Override with `CLAUDE_MEMORY_DIR` environment variable.

## Memory types

| Type | Purpose |
|------|---------|
| `user` | Facts about the user — role, preferences, expertise |
| `feedback` | How to approach work — corrections and confirmed approaches |
| `project` | Ongoing work context — decisions, deadlines, constraints |
| `reference` | Pointers to external resources — URLs, dashboards, systems |

## How search works

Every query runs through two retrieval paths simultaneously:

1. **BM25** (Tantivy) — term-frequency keyword matching, great for exact names and patterns
2. **Vector** (fastembed cosine similarity) — semantic similarity, great for "I vaguely remember something about..."

Results are combined via **Reciprocal Rank Fusion** (k=60), which merges ranked lists without requiring score normalization. A memory that ranks well in both paths gets a strong combined score.

## Design decisions

- **SQLite is the source of truth.** The Tantivy index is a derived cache — if corrupted, it rebuilds from SQLite on next startup.
- **Embeddings are brute-force cosine.** For a personal memory system (<100K memories), this is fast enough and avoids ANN index complexity.
- **Model loads lazily.** Commands that don't need embeddings (e.g., `recall`, `forget --id`) skip the ~200ms model load.
- **Access counts track usage.** Every retrieval increments `access_count`, enabling `prune` to identify stale memories.
- **All logging goes to stderr.** Stdout is reserved for JSON results (CLI) or JSON-RPC (MCP), so logging never pollutes the transport.
