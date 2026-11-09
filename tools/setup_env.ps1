param([switch]$Dev)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentInfo = conda env list --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not list Conda environments' }
$environmentPath = $environmentInfo.envs | Where-Object { (Split-Path -Leaf $_) -ceq 'aiReady' } | Select-Object -First 1
if (-not $environmentPath) {
    conda create --name aiReady python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
    $environmentInfo = conda env list --json | ConvertFrom-Json
    $environmentPath = $environmentInfo.envs | Where-Object { (Split-Path -Leaf $_) -ceq 'aiReady' } | Select-Object -First 1
}
$aiReadyPython = Join-Path $environmentPath 'python.exe'
& $aiReadyPython -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
if ($LASTEXITCODE -ne 0) { throw 'aiReady must use Python 3.11' }
& $aiReadyPython -m pip --isolated install --index-url https://pypi.org/simple uv
if ($LASTEXITCODE -ne 0) { throw 'uv installation failed' }
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache/uv'
$installTarget = if ($Dev) { "$projectRoot[dev]" } else { $projectRoot }
$constraints = Join-Path $projectRoot 'requirements-win-py311.lock'
& $aiReadyPython -m uv pip install --python $aiReadyPython --index-url https://pypi.org/simple -c $constraints -e $installTarget
if ($LASTEXITCODE -ne 0) { throw 'L1-L2 dependency installation failed' }
& $aiReadyPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed package constraints are inconsistent' }
Write-Output "Ready. Interpreter: $aiReadyPython"
