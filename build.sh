#!/usr/bin/env bash
set -e

echo "Building memory (release)..."
cargo build --release

BINARY="target/release/memory"

if [ -n "$1" ]; then
    mkdir -p "$1"
    cp "$BINARY" "$1/memory"
    INSTALL_PATH="$(cd "$1" && pwd)/memory"
    echo "Installed to $INSTALL_PATH"
else
    INSTALL_PATH="$(pwd)/$BINARY"
fi

echo ""
echo "Build complete."
echo ""
echo "To register as MCP server:"
echo "  claude mcp add memory -- \"$INSTALL_PATH\" serve"
