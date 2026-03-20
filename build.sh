#!/usr/bin/env bash
set -e

echo "Building claude-memory (release)..."
cargo build --release

BINARY="target/release/claude-memory"

if [ -n "$1" ]; then
    mkdir -p "$1"
    cp "$BINARY" "$1/claude-memory"
    INSTALL_PATH="$(cd "$1" && pwd)/claude-memory"
    echo "Installed to $INSTALL_PATH"
else
    INSTALL_PATH="$(pwd)/$BINARY"
fi

echo ""
echo "Build complete."
echo ""
echo "To register as MCP server:"
echo "  claude mcp add claude-memory -- \"$INSTALL_PATH\" serve"
