<#
.SYNOPSIS
    Enable (or disable) a persistent JIT/compiler cache for PyTorch XPU kernels.

.DESCRIPTION
    PyTorch's XPU backend ships kernels as SPIR-V inside torch_xpu.dll; the Intel
    Level Zero driver (NEO) JIT-compiles them per kernel on first use. That
    compilation costs seconds to minutes and, by default, is thrown away when the
    process exits.

    The NEO driver already contains a module cache. It is inert until
    NEO_CACHE_DIR points at a writable directory -- that single variable turns
    per-process compiles into one-time compiles.

    Nothing else (NEO_CACHE_PERSISTENT, SYCL_CACHE_*) is required: setting
    NEO_CACHE_PERSISTENT=1 alone does nothing, and NEO_CACHE_DIR alone is enough.

.PARAMETER CacheDir
    Where to keep the compiled kernels. Default: %LOCALAPPDATA%\torch_xpu_cache

.PARAMETER MaxSizeMB
    Size cap in MiB (written to NEO_CACHE_MAX_SIZE as bytes). Default: 4096.

.PARAMETER Disable
    Remove the user-scope variables and report the current state.

.PARAMETER CurrentSessionOnly
    Only set $env: for this PowerShell session, do not touch the user environment.

.EXAMPLE
    .\enable_jit_cache.ps1
    .\enable_jit_cache.ps1 -CacheDir D:\caches\xpu -MaxSizeMB 8192
    .\enable_jit_cache.ps1 -Disable
#>
[CmdletBinding()]
param(
    [string]$CacheDir = (Join-Path $env:LOCALAPPDATA 'torch_xpu_cache'),
    [int]$MaxSizeMB = 4096,
    [switch]$Disable,
    [switch]$CurrentSessionOnly
)

$ErrorActionPreference = 'Stop'
$maxBytes = [string]($MaxSizeMB * 1MB)

if ($Disable) {
    if (-not $CurrentSessionOnly) {
        [Environment]::SetEnvironmentVariable('NEO_CACHE_DIR', $null, 'User')
        [Environment]::SetEnvironmentVariable('NEO_CACHE_MAX_SIZE', $null, 'User')
    }
    Remove-Item Env:NEO_CACHE_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:NEO_CACHE_MAX_SIZE -ErrorAction SilentlyContinue
    Write-Host 'Persistent XPU kernel cache disabled (user scope).' -ForegroundColor Yellow
} else {
    if (-not (Test-Path $CacheDir)) {
        New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
        Write-Host "Created cache directory: $CacheDir"
    }

    $env:NEO_CACHE_DIR = $CacheDir
    $env:NEO_CACHE_MAX_SIZE = $maxBytes

    if (-not $CurrentSessionOnly) {
        [Environment]::SetEnvironmentVariable('NEO_CACHE_DIR', $CacheDir, 'User')
        [Environment]::SetEnvironmentVariable('NEO_CACHE_MAX_SIZE', $maxBytes, 'User')
    }

    Write-Host 'Persistent XPU kernel cache enabled.' -ForegroundColor Green
    Write-Host "  NEO_CACHE_DIR      = $CacheDir"
    Write-Host "  NEO_CACHE_MAX_SIZE = $maxBytes bytes ($MaxSizeMB MiB)"
    if ($CurrentSessionOnly) {
        Write-Host '  (this session only -- new terminals will not see it)' -ForegroundColor Yellow
    } else {
        Write-Host '  new processes pick this up from the user environment; already-running shells need a restart'
    }
}

Write-Host ''
Write-Host 'Current user-scope values:'
foreach ($name in 'NEO_CACHE_DIR', 'NEO_CACHE_MAX_SIZE', 'NEO_CACHE_PERSISTENT') {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if ([string]::IsNullOrEmpty($value)) { $value = '(unset)' }
    Write-Host ("  {0,-20} {1}" -f $name, $value)
}

if (Test-Path $CacheDir) {
    $entries = Get-ChildItem $CacheDir -Recurse -File -ErrorAction SilentlyContinue
    $mib = if ($entries) { [math]::Round(($entries | Measure-Object -Property Length -Sum).Sum / 1MB, 1) } else { 0 }
    Write-Host ("Cache contents: {0} files, {1} MiB" -f @($entries).Count, $mib)
}
