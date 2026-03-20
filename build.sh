#!/usr/bin/env bash
set -e

echo "Building claude-memory (release)..."
cargo build --release

BINARY="target/release/claude-memory"
echo ""
echo "Build complete: $BINARY"
echo ""
echo "To register as MCP server:"
echo "  claude mcp add claude-memory -- \"$(pwd)/$BINARY\" serve"
