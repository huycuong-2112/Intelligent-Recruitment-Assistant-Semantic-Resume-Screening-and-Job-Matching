param()

function Get-VsDevCmdPath {
    $vswhereCandidates = @(
        Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe'
    )

    $vswhere = $vswhereCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $vswhere) {
        return $null
    }

    $installPath = & $vswhere -latest -products * -property installationPath | Select-Object -First 1
    if (-not $installPath) {
        return $null
    }

    $vsDevCmd = Join-Path $installPath 'Common7\Tools\VsDevCmd.bat'
    if (Test-Path $vsDevCmd) {
        return $vsDevCmd
    }

    return $null
}

function Import-BatchEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BatchFile,

        [string[]]$Arguments = @()
    )

    $argumentText = if ($Arguments.Count -gt 0) { $Arguments -join ' ' } else { '' }
    $commandText = "call `"$BatchFile`" $argumentText >nul && set"
    $batchOutput = & cmd.exe /d /c $commandText

    foreach ($line in $batchOutput) {
        if ($line -match '^(?<name>[^=]+)=(?<value>.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches.name, $matches.value, 'Process')
        }
    }
}

function Get-CudaRoot {
    $candidatePaths = @()

    if ($env:CUDA_PATH) {
        $candidatePaths += $env:CUDA_PATH
    }

    if ($env:CUDA_HOME) {
        $candidatePaths += $env:CUDA_HOME
    }

    $candidatePaths += @(
        Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
        Join-Path ${env:ProgramFiles(x86)} 'NVIDIA GPU Computing Toolkit\CUDA'
    )

    foreach ($candidatePath in $candidatePaths) {
        if (-not $candidatePath) {
            continue
        }

        if (Test-Path $candidatePath) {
            if ((Split-Path $candidatePath -Leaf) -match '^v\d') {
                return $candidatePath
            }

            $versions = Get-ChildItem -Path $candidatePath -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^v?\d' } |
                Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending

            if ($versions) {
                return $versions[0].FullName
            }
        }
    }

    return $null
}

$vsDevCmd = Get-VsDevCmdPath
if ($vsDevCmd) {
    Import-BatchEnvironment -BatchFile $vsDevCmd -Arguments @('-arch=amd64', '-host_arch=amd64')
} else {
    Write-Warning 'VsDevCmd.bat was not found. cl.exe/Windows SDK env vars may remain incomplete.'
}

$cudaRoot = Get-CudaRoot
if ($cudaRoot) {
    $env:CUDA_PATH = $cudaRoot
    $cudaBin = Join-Path $cudaRoot 'bin'

    if ((Test-Path $cudaBin) -and ($env:Path -notlike "*$cudaBin*")) {
        $env:Path = "$cudaBin;$env:Path"
    }
} else {
    Write-Warning 'CUDA installation was not found automatically.'
}

$clCommand = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clCommand) {
    $env:CC = $clCommand.Source
    $env:CXX = $clCommand.Source
} else {
    Write-Warning 'cl.exe is still not visible after environment setup.'
}

Write-Host 'Windows build environment configured for Docling/Triton.'
Write-Host "CC=$env:CC"
Write-Host "CXX=$env:CXX"
Write-Host "CUDA_PATH=$env:CUDA_PATH"