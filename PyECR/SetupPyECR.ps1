param([switch]$ResetEnvironment)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PSModulePath = (Join-Path $PSHOME 'Modules') + ';' + $env:PSModulePath

function Test-PyECRPython([string]$PythonPath) {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { return $false }
    try {
        & $PythonPath -I -c "import struct, sys, sysconfig, venv, ensurepip; sys.exit(not (sys.version_info[:2] == (3, 14) and struct.calcsize('P') == 8 and not sysconfig.get_config_var('Py_GIL_DISABLED')))" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Assert-PrivatePath([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    $prefix = $script:privateRoot.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Setup path is outside the PyECR application data folder: $resolved"
    }
    return $resolved
}

try {
    if (-not $env:LOCALAPPDATA) { throw 'The LOCALAPPDATA folder is not available.' }
    $script:privateRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'PyECR'))
    if ($ResetEnvironment) {
        $environment = Assert-PrivatePath (Join-Path $privateRoot 'V1.0\venv')
        if (Test-Path -LiteralPath $environment) {
            $backup = Assert-PrivatePath ($environment + '.unusable-' + [guid]::NewGuid().ToString('N'))
            Move-Item -LiteralPath $environment -Destination $backup
            Write-Host "The previous environment was preserved at $backup"
        }
        exit 0
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw 'PyECR requires a 64-bit Windows system.'
    }
    $runtime = Assert-PrivatePath (Join-Path $privateRoot 'Runtime\Python3146')
    $python = Join-Path $runtime 'python.exe'
    if (Test-PyECRPython $python) { exit 0 }

    # CPython's official release manifest supplies the full runtime and SHA-256.
    # https://www.python.org/ftp/python/3.14.6/windows-3.14.6.json
    $url = 'https://www.python.org/ftp/python/3.14.6/python-3.14.6-amd64.zip'
    $expectedHash = '75afa83f93b284d19040e24bc440ab741c09582c0d5310504d607a4e08c3dbaf'
    $downloads = Assert-PrivatePath (Join-Path $privateRoot 'Downloads')
    New-Item -ItemType Directory -Path $downloads -Force | Out-Null
    $archive = Assert-PrivatePath (Join-Path $downloads 'python-3.14.6-amd64.zip')
    $verified = (Test-Path -LiteralPath $archive -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -eq $expectedHash)
    if (-not $verified) {
        $partial = Assert-PrivatePath ($archive + '.' + [guid]::NewGuid().ToString('N') + '.partial')
        Write-Host 'Downloading the official 64-bit Python runtime from python.org...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $partial -TimeoutSec 120
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $expectedHash) {
            throw 'The Python download failed its SHA-256 check. No downloaded code was executed.'
        }
        Move-Item -LiteralPath $partial -Destination $archive -Force
    }
    Write-Host 'Verified the official Python archive. Preparing the runtime...'
    $staging = Assert-PrivatePath ($runtime + '.setup-' + [guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $archive -DestinationPath $staging
    if (-not (Test-PyECRPython (Join-Path $staging 'python.exe'))) {
        throw 'The extracted Python runtime could not start on this computer.'
    }
    if (Test-Path -LiteralPath $runtime) {
        $backup = Assert-PrivatePath ($runtime + '.unusable-' + [guid]::NewGuid().ToString('N'))
        Move-Item -LiteralPath $runtime -Destination $backup
    }
    Move-Item -LiteralPath $staging -Destination $runtime
    if (-not (Test-PyECRPython $python)) { throw 'Python setup did not finish successfully.' }
    Write-Host "Python is ready at $python"
    exit 0
} catch {
    Write-Host ('Python setup failed: ' + $_.Exception.Message)
    Write-Host 'Check access to https://www.python.org and run StartPyECR.cmd again.'
    Write-Host 'Alternatively, install 64-bit Python 3.14 from https://www.python.org/downloads/release/python-3146/.'
    exit 1
}
