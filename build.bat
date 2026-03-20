@echo off
echo Building claude-memory (release)...
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    exit /b %ERRORLEVEL%
)
echo.
echo Build complete: target\release\claude-memory.exe
echo.
echo To register as MCP server:
echo   claude mcp add claude-memory -- "%CD%\target\release\claude-memory.exe" serve
