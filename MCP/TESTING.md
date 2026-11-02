# Verified on 2026-09-26

- Release app builds successfully. Existing unrelated dead-code warnings remain.
- App test suite: 60 passed, 21 opt-in hardware tests skipped; two additional
  connection timing tests subsequently passed. Final focused MCP Rust run: 6 passed.
- Python MCP package tests: 8 passed using the package-local MCP SDK 1.30.0.
- Skill frontmatter and references validated.
- Installer JSON merge tested with a temporary VS Code config: unrelated
  settings retained and a backup created. No global client registrations made.

## Live app verification

The official SDK's stdio client called the real MCP server and running app.
It exercised launch/reuse, state/schema discovery, mode and panel selection,
atomic rejection of invalid settings, visible configuration, import, comparison
view, source playback/pause/seeking, preview, batch creation/export, task status,
queue/media clearing, cache release, and MCP logs.

RTX VSR Ultra completed a 256×256 → 512×512 PNG check, a short video check, and
the user-provided Grok clip: 2560×1440 output, 24 fps, 241 frames. The latter
reported `done · 4s`. Inputs were copied to `.smoke-output`; originals were not
modified. Result files and `result.local.json` remain there for inspection.
These checks prove operation and output creation, not a subjective quality score.

The MCP tab was visually checked under Output, including the green connected
state, client name, and command audit. Disconnect/reconnect through two separate
MCP sessions retained the same app PID. The verified launcher hands off the fixed
app path to the existing Windows Explorer process; ordinary child-process and
in-process Shell.Application launches did not provide that lifetime guarantee.
The independent app correctly reports RTX 3090 and RUNTIME OK.

## Reproduce

```powershell
MCP\.venv\Scripts\python.exe -m unittest discover -s MCP\tests -v
cargo test mcp
MCP\.venv\Scripts\python.exe MCP\smoke_test.py
# Optional real GPU/export check; requires an empty, idle app:
MCP\.venv\Scripts\python.exe MCP\smoke_test.py --exercise-media "C:\path\clip.mp4"
```

The default smoke check launches/reuses the app and reads state. It leaves the
MCP panel open. The optional media check restores the settings it changes and
leaves the app open; it does not cancel pre-existing work or overwrite outputs.
