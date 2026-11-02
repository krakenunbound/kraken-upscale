[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$RegisterCodex,
    [switch]$RegisterClaude,
    [switch]$RegisterVSCode,
    [switch]$RegisterClaudeDesktop,
    [switch]$RegisterCursor,
    [switch]$RegisterAll,
    [switch]$InstallSkills,
    [string]$VSCodeConfigPath,
    [string]$ClaudeDesktopConfigPath,
    [string]$CursorConfigPath,
    [string]$SkillsPath
)
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw "This setup script requires PowerShell 7+ (pwsh) to safely preserve JSON client configurations." }
$runtime = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $runtime)) {
    & $Python -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the MCP environment. Python 3.11 or newer is required." }
}
& $runtime -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Kraken Upscale MCP requires Python 3.11 or newer." }
& $runtime -m pip install --upgrade pip
& $runtime -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "MCP dependency installation failed." }
$server = Join-Path $PSScriptRoot "server.py"
$config = @{ mcpServers = @{ "kraken-upscale" = @{ command = $runtime; args = @($server) } } }
[IO.File]::WriteAllText((Join-Path $PSScriptRoot "client-config.local.json"), ($config | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
[IO.File]::WriteAllText((Join-Path $PSScriptRoot "claude-desktop-config.example.json"), ($config | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
[IO.File]::WriteAllText((Join-Path $PSScriptRoot "cursor-mcp.json.example"), ($config | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
$toml = "[mcp_servers.kraken-upscale]`ncommand = '" + $runtime.Replace("'", "''") + "'`nargs = ['" + $server.Replace("'", "''") + "']`n"
[IO.File]::WriteAllText((Join-Path $PSScriptRoot "codex-mcp.toml.example"), $toml, (New-Object Text.UTF8Encoding $false))
$vsCodeConfig = @{ servers = @{ "kraken-upscale" = @{ command = $runtime; args = @($server) } } }
[IO.File]::WriteAllText((Join-Path $PSScriptRoot "vscode-mcp.json.example"), ($vsCodeConfig | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
function Add-McpServerConfig([string]$Path, [string]$ServerRoot) {
    $existing = @{}
    if (Test-Path -LiteralPath $Path) {
        Copy-Item -LiteralPath $Path -Destination ($Path + ".kraken-upscale-backup") -Force
        $existing = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
    }
    if (-not $existing.ContainsKey($ServerRoot)) { $existing[$ServerRoot] = @{} }
    if (-not ($existing[$ServerRoot] -is [hashtable])) { throw "MCP config has a non-object '$ServerRoot' value." }
    $existing[$ServerRoot]["kraken-upscale"] = @{ command = $runtime; args = @($server) }
    $targetDirectory = Split-Path -Parent $Path
    if ($targetDirectory) { New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null }
    [IO.File]::WriteAllText($Path, ($existing | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
}
if ($RegisterAll) {
    $RegisterCodex = $null -ne (Get-Command codex -ErrorAction SilentlyContinue)
    $RegisterClaude = $null -ne (Get-Command claude -ErrorAction SilentlyContinue)
    $RegisterClaudeDesktop = Test-Path -LiteralPath (Join-Path $env:APPDATA "Claude\claude_desktop_config.json")
    $RegisterCursor = Test-Path -LiteralPath (Join-Path $env:USERPROFILE ".cursor\mcp.json")
    $RegisterVSCode = $false
    Write-Host "RegisterAll detected Codex=$RegisterCodex ClaudeCode=$RegisterClaude ClaudeDesktop=$RegisterClaudeDesktop Cursor=$RegisterCursor; VS Code needs an explicit workspace config path."
}
if ($RegisterCodex) { & codex mcp add kraken-upscale -- $runtime $server; if ($LASTEXITCODE -ne 0) { throw "Codex MCP registration failed." } }
if ($RegisterClaude) { & claude mcp add kraken-upscale --scope user -- $runtime $server; if ($LASTEXITCODE -ne 0) { throw "Claude Code MCP registration failed." } }
if ($RegisterVSCode) { if (-not $VSCodeConfigPath) { throw "-RegisterVSCode requires -VSCodeConfigPath, for example .vscode/mcp.json." }; Add-McpServerConfig $VSCodeConfigPath "servers" }
if ($RegisterClaudeDesktop) { if (-not $ClaudeDesktopConfigPath) { $ClaudeDesktopConfigPath = Join-Path $env:APPDATA "Claude\claude_desktop_config.json" }; Add-McpServerConfig $ClaudeDesktopConfigPath "mcpServers" }
if ($RegisterCursor) { if (-not $CursorConfigPath) { $CursorConfigPath = Join-Path $env:USERPROFILE ".cursor\mcp.json" }; Add-McpServerConfig $CursorConfigPath "mcpServers" }
if ($InstallSkills) {
    if (-not $SkillsPath) { throw "-InstallSkills requires -SkillsPath. The portable source is MCP\skills\kraken-upscale." }
    $skillSource = Join-Path $PSScriptRoot "skills\kraken-upscale"
    $skillDestination = Join-Path $SkillsPath "kraken-upscale"
    if (Test-Path -LiteralPath $skillDestination) { throw "Skill destination already exists and was left unchanged: $skillDestination" }
    New-Item -ItemType Directory -Force -Path $SkillsPath | Out-Null
    Copy-Item -LiteralPath $skillSource -Destination $skillDestination -Recurse -ErrorAction Stop
    Write-Host "Installed portable Kraken Upscale skill at $skillDestination"
}
if ($RegisterCodex -or $RegisterClaude -or $RegisterVSCode -or $RegisterClaudeDesktop -or $RegisterCursor) {
    Write-Host "Kraken Upscale MCP is ready. Only the explicitly selected client registrations were changed."
} else {
    Write-Host "Kraken Upscale MCP is ready. No global client configuration was changed."
}
Write-Host "Generated: client-config.local.json, codex-mcp.toml.example, vscode-mcp.json.example, claude-desktop-config.example.json, cursor-mcp.json.example"
Write-Host "To register with Codex later: codex mcp add kraken-upscale -- `"$runtime`" `"$server`""
Write-Host "To register with Claude Code later: claude mcp add kraken-upscale --scope user -- `"$runtime`" `"$server`""
