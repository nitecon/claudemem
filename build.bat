@echo off
echo Building memory (release)...
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    exit /b %ERRORLEVEL%
)

set "BINARY=target\release\memory.exe"

if "%~1"=="" (
    set "INSTALL_PATH=%CD%\%BINARY%"
) else (
    if not exist "%~1" mkdir "%~1"
    copy /Y "%BINARY%" "%~1\memory.exe" >nul
    set "INSTALL_PATH=%~1\memory.exe"
    echo Installed to %~1\memory.exe
)

echo.
echo Build complete.
echo.
echo To register as MCP server:
echo   claude mcp add memory -- "%INSTALL_PATH%" serve
