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
2. Apply the memory quality gate:
   - Store only information that will help a future agent work faster: reusable patterns, operational procedures, user preferences, non-obvious constraints, failure causes, or "how to / why" guidance.
   - Write for a cold agent who has not seen this session: explain what to do next, which tool or system to use, and why that path is correct.
   - Do not store facts recoverable from git history, repository inspection, CI/release systems, or agent-tools tasks/comms.
   - Do not store routine deployment status, version numbers, release events, commit SHAs, branch state, "CI passed", "tag was pushed", or "deployed version X" memories.
   - Store deployment/version facts only when they explain a failure mode or encode a reusable procedure that prevents future mistakes.
   - Do not store "updated pattern X" notes. If a user refers to "patterns", they likely mean gateway-backed `agent-tools patterns`; useful memories should describe the `agent-tools patterns --help` / get / update / check workflow and why it matters.
3. Choose the appropriate memory_type:
   - "user" — facts about the user (role, preferences, expertise)
   - "feedback" — guidance on how to approach work (corrections, confirmations)
   - "project" — ongoing work context (decisions, deadlines, constraints)
   - "reference" — pointers to external resources (URLs, systems, dashboards)
4. Extract relevant tags (comma-separated, lowercase, specific)
5. Detect the project context if obvious from the conversation or working directory
6. Call the `memory_store` MCP tool with:
   - `content`: the memory text, written clearly for future retrieval
   - `tags`: extracted tags
   - `project`: project identifier if known
   - `memory_type`: one of the four types above
   - `agent`: the agent identifier if this memory is agent-specific
7. Confirm storage to the user with the memory ID

If the input is vague, ask a clarifying question before storing. Memories should be specific enough to be useful when retrieved later.

If the memory is a correction or preference about how you should behave, make sure to frame it as actionable guidance in the content field.
