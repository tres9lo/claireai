# ClaireAI installer for Windows. Detects the host, prints every download and
# asks first. Usage: .\install.ps1 [-yes] [-dry-run]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    & py -3 scripts/bootstrap.py @Rest
    exit $LASTEXITCODE
}

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & python scripts/bootstrap.py @Rest
    exit $LASTEXITCODE
}

Write-Error "No Python found. Install Python 3.11+ (with tcl/tk) and run this again."
exit 2
