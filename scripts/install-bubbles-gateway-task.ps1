#requires -Version 5.1
#requires -Modules ScheduledTasks

[CmdletBinding()]
param(
    [string]$RepoPath,
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    [string]$Remote = "bubblebot",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._/-]*$")]
    [string]$Branch = "main",
    [ValidateRange(1, 65535)]
    [int]$Port = 18790,
    [ValidateNotNullOrEmpty()]
    [string]$TaskName = "Bubblebot Gateway",
    [switch]$StartNow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ApplicationPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $command = Get-Command $Name -CommandType Application -ErrorAction Stop |
        Select-Object -First 1
    return $command.Path
}

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = @(& $FilePath @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = (($output | ForEach-Object { [string]$_ }) -join [Environment]::NewLine)
    }
}

function Quote-TaskArgument {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ($Value.Contains('"')) {
        throw "Task arguments cannot contain a double quote: $Value"
    }
    return '"' + $Value + '"'
}

function Get-TextSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha256.Dispose()
    }
}

function Install-FileAtomically {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TemporaryPath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [switch]$KeepBackup
    )

    $backupPath = "$DestinationPath.bak"
    try {
        Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $DestinationPath -PathType Leaf) {
            [System.IO.File]::Replace($TemporaryPath, $DestinationPath, $backupPath, $true)
        } else {
            [System.IO.File]::Move($TemporaryPath, $DestinationPath)
        }
    } finally {
        Remove-Item -LiteralPath $TemporaryPath -Force -ErrorAction SilentlyContinue
        if (-not $KeepBackup) {
            Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Assert-SupervisorScriptContract {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $tokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $Path,
        [ref]$tokens,
        [ref]$parseErrors
    )
    if ($parseErrors.Count -gt 0) {
        throw "Supervisor does not parse under Windows PowerShell 5.1: $($parseErrors[0].Message)"
    }

    $taskParameterNames = @(
        "RepoPath", "GitPath", "UvPath", "Remote", "RemoteUrlSha256", "Branch", "Port"
    )
    $parameters = @($ast.ParamBlock.Parameters)
    $parameterNames = @(
        $parameters | ForEach-Object { $_.Name.VariablePath.UserPath }
    )
    foreach ($requiredName in $taskParameterNames) {
        if ($parameterNames -notcontains $requiredName) {
            throw "Supervisor is missing required parameter '$requiredName'."
        }
    }

    foreach ($parameter in $parameters) {
        $parameterName = $parameter.Name.VariablePath.UserPath
        foreach ($attribute in $parameter.Attributes) {
            if (-not ($attribute -is [System.Management.Automation.Language.AttributeAst])) {
                continue
            }
            if ($attribute.TypeName.FullName -notin @(
                "Parameter", "ParameterAttribute", "System.Management.Automation.ParameterAttribute"
            )) {
                continue
            }
            foreach ($namedArgument in $attribute.NamedArguments) {
                if ($namedArgument.ArgumentName -ne "Mandatory") {
                    continue
                }
                $isMandatory = $namedArgument.ExpressionOmitted
                if (-not $isMandatory -and $null -ne $namedArgument.Argument) {
                    try {
                        $isMandatory = [bool]$namedArgument.Argument.SafeGetValue()
                    } catch {
                        throw "Unable to validate Mandatory for supervisor parameter '$parameterName'."
                    }
                }
                if ($isMandatory -and $taskParameterNames -notcontains $parameterName) {
                    throw "Supervisor adds mandatory parameter '$parameterName' that the scheduled task does not supply."
                }
            }
        }
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "install-bubbles-gateway-task.ps1 supports Windows only."
}
if ($PSVersionTable.PSEdition -ne "Desktop" -or $PSVersionTable.PSVersion.Major -ne 5) {
    throw "Run this installer under Windows PowerShell 5.1, not pwsh."
}
if ([Environment]::Is64BitOperatingSystem -and -not [Environment]::Is64BitProcess) {
    throw "Run this installer under 64-bit Windows PowerShell on 64-bit Windows."
}
if (-not [Environment]::UserInteractive) {
    throw "Run this installer from the current user's interactive session."
}

# Resolve the default during script execution, not parameter binding: script
# automatic variables may still be empty when a default expression is evaluated.
if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $scriptDirectory = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($scriptDirectory) -and
        -not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        $scriptDirectory = Split-Path -Parent $PSCommandPath
    }
    if ([string]::IsNullOrWhiteSpace($scriptDirectory)) {
        throw "Cannot determine the script directory. Supply -RepoPath with the repository's absolute path."
    }
    $RepoPath = Split-Path -Parent $scriptDirectory
}

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
if ($identity.IsSystem) {
    throw "Do not install the Bubblebot task as LocalSystem."
}

$canonicalRepoPath = (Resolve-Path -LiteralPath $RepoPath).Path.TrimEnd([char[]]"\/")
$sourceSupervisorPath = Join-Path $canonicalRepoPath "scripts\bubbles-supervisor.ps1"
if (-not (Test-Path -LiteralPath $sourceSupervisorPath -PathType Leaf)) {
    throw "Supervisor script not found: $sourceSupervisorPath"
}
Assert-SupervisorScriptContract -Path $sourceSupervisorPath
foreach ($requiredFile in @("pyproject.toml", "uv.lock")) {
    $path = Join-Path $canonicalRepoPath $requiredFile
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required repository file is missing: $path"
    }
}

$gitExe = Resolve-ApplicationPath "git"
$uvExe = Resolve-ApplicationPath "uv"

$branchNameResult = Invoke-NativeCapture -FilePath $gitExe -Arguments @(
    "check-ref-format", "--branch", $Branch
)
if ($branchNameResult.ExitCode -ne 0) {
    throw "Configured branch name is invalid."
}

$rootResult = Invoke-NativeCapture -FilePath $gitExe -Arguments @(
    "-C", $canonicalRepoPath, "rev-parse", "--show-toplevel"
)
if ($rootResult.ExitCode -ne 0) {
    throw "RepoPath is not a Git worktree: $($rootResult.Output)"
}
$reportedRoot = [System.IO.Path]::GetFullPath($rootResult.Output.Trim()).TrimEnd([char[]]"\/")
if (-not $reportedRoot.Equals($canonicalRepoPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "RepoPath must be the worktree root. Expected '$reportedRoot', got '$canonicalRepoPath'."
}

$branchResult = Invoke-NativeCapture -FilePath $gitExe -Arguments @(
    "-C", $canonicalRepoPath, "branch", "--show-current"
)
if ($branchResult.ExitCode -ne 0 -or $branchResult.Output.Trim() -ne $Branch) {
    throw "Task installation requires the repository to be on branch '$Branch'."
}

$statusResult = Invoke-NativeCapture -FilePath $gitExe -Arguments @(
    "-C", $canonicalRepoPath, "status", "--porcelain=v1", "--untracked-files=normal"
)
if ($statusResult.ExitCode -ne 0) {
    throw "Unable to inspect Git worktree: $($statusResult.Output)"
}
if (-not [string]::IsNullOrWhiteSpace($statusResult.Output)) {
    throw "Task installation requires a clean Git worktree."
}

$remoteResult = Invoke-NativeCapture -FilePath $gitExe -Arguments @(
    "-C", $canonicalRepoPath, "remote", "get-url", $Remote
)
if ($remoteResult.ExitCode -ne 0) {
    throw "Configured Git remote '$Remote' does not exist."
}
$remoteUrl = $remoteResult.Output.Trim()
if ([string]::IsNullOrWhiteSpace($remoteUrl)) {
    throw "Configured Git remote '$Remote' has no fetch URL."
}
$remoteUrlSha256 = Get-TextSha256 -Text $remoteUrl

$env:UV_HTTP_TIMEOUT = "60"
Push-Location $canonicalRepoPath
try {
    $initialSyncResult = Invoke-NativeCapture -FilePath $uvExe -Arguments @(
        "sync", "--locked"
    )
} finally {
    Pop-Location
}
if ($initialSyncResult.ExitCode -ne 0) {
    throw "Initial uv sync --locked failed with exit code $($initialSyncResult.ExitCode)."
}

$userProfile = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
if ([string]::IsNullOrWhiteSpace($userProfile)) {
    $userProfile = $HOME
}
$controlDir = Join-Path (Join-Path $userProfile ".bubbles") "control"
[void](New-Item -ItemType Directory -Path $controlDir -Force)
$installedSupervisorPath = Join-Path $controlDir "bubbles-supervisor.ps1"
$temporarySupervisorPath = "$installedSupervisorPath.$PID.tmp"
try {
    Copy-Item -LiteralPath $sourceSupervisorPath -Destination $temporarySupervisorPath -Force
    Assert-SupervisorScriptContract -Path $temporarySupervisorPath
    Install-FileAtomically `
        -TemporaryPath $temporarySupervisorPath `
        -DestinationPath $installedSupervisorPath `
        -KeepBackup
} finally {
    if (Test-Path -LiteralPath $temporarySupervisorPath) {
        Remove-Item -LiteralPath $temporarySupervisorPath -Force -ErrorAction SilentlyContinue
    }
}

$powerShellPath = Join-Path $PSHOME "powershell.exe"
if (-not (Test-Path -LiteralPath $powerShellPath -PathType Leaf)) {
    throw "Unable to locate Windows PowerShell 5.1 under '$PSHOME'."
}

$argumentParts = @(
    "-NoLogo",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", (Quote-TaskArgument $installedSupervisorPath),
    "-RepoPath", (Quote-TaskArgument $canonicalRepoPath),
    "-GitPath", (Quote-TaskArgument $gitExe),
    "-UvPath", (Quote-TaskArgument $uvExe),
    "-Remote", (Quote-TaskArgument $Remote),
    "-RemoteUrlSha256", (Quote-TaskArgument $remoteUrlSha256),
    "-Branch", (Quote-TaskArgument $Branch),
    "-Port", [string]$Port
)
$actionArguments = $argumentParts -join " "

$action = New-ScheduledTaskAction `
    -Execute $powerShellPath `
    -Argument $actionArguments `
    -WorkingDirectory $canonicalRepoPath
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity.Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $identity.Name `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Run Bubblebot gateway under the current user's Windows supervisor."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName'."
Write-Host "User:       $($identity.Name) (interactive logon only)"
Write-Host "Repository: $canonicalRepoPath"
Write-Host "Supervisor: $installedSupervisorPath"
Write-Host "Target:     $Remote/$Branch"
Write-Host "Command:    uv run --no-sync bubbles gateway --port $Port"

if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Started scheduled task '$TaskName'."
} else {
    Write-Host "The task will start at the next logon. Use -StartNow to start it immediately."
}
