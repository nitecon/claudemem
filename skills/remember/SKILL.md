---
name: remember
description: Store a memory for future retrieval. Use when the user says "remember this", wants to save context, preferences, decisions, or any information that should persist across conversations.
argument-hint: "[memory text or description of what to remember]"
allowed-tools: mcp__agent-memory__memory_store, mcp__agent-memory__memory_search
---

The user wants to store a memory. The input is:

$ARGUMENTS

Your job:
1. Parse the input to determine what should be stored
2. Choose the appropriate memory_type:
   - "user" — facts about the user (role, preferences, expertise)
   - "feedback" — guidance on how to approach work (corrections, confirmations)
   - "project" — ongoing work context (decisions, deadlines, constraints)
   - "reference" — pointers to external resources (URLs, systems, dashboards)
3. Extract relevant tags (comma-separated, lowercase, specific)
4. Detect the project context if obvious from the conversation or working directory
5. Call the `memory_store` MCP tool with:
   - `content`: the memory text, written clearly for future retrieval
   - `tags`: extracted tags
   - `project`: project identifier if known
   - `memory_type`: one of the four types above
   - `agent`: the agent identifier if this memory is agent-specific
6. Confirm storage to the user with the memory ID

If the input is vague, ask a clarifying question before storing. Memories should be specific enough to be useful when retrieved later.

If the memory is a correction or preference about how you should behave, make sure to frame it as actionable guidance in the content field.
