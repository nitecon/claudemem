---
name: recall
description: Search and retrieve stored memories. Use when the user asks to recall, remember, look up past context, or when you need to find relevant information from previous conversations.
argument-hint: "[search query or topic]"
allowed-tools: mcp__agent-memory__memory_search, mcp__agent-memory__memory_context, mcp__agent-memory__memory_recall
---

The user wants to retrieve memories. The query is:

$ARGUMENTS

Your job:
1. Determine the best retrieval strategy:
   - **Semantic search** (`memory_search`): use when the query is natural language or conceptual — "what does the user prefer for testing?", "anything about auth?"
   - **Filtered recall** (`memory_recall`): use when the query targets a specific project, agent, tag, or memory type — "all feedback memories", "memories tagged auth"
   - **Task context** (`memory_context`): use when loading context for a specific task — "I'm about to refactor the auth middleware"
2. Execute the appropriate MCP tool call(s)
3. Present results clearly:
   - Show the most relevant memories first
   - Include tags, project, and memory type for context
   - If a memory seems stale or potentially outdated, note that
   - Quote the memory content directly — don't paraphrase

If no results are found, say so and suggest alternative search terms.

If the query could benefit from multiple retrieval strategies, run them in parallel and merge the results.
