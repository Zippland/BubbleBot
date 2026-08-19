#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$RepoPath = (Split-Path -Parent $PSScriptRoot),
    [string]$GitPath = "git",
    [string]$UvPath = "uv",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    [string]$Remote = "bubblebot",
    [ValidatePattern("^$|^[A-Fa-f0-9]{64}$")]
    [string]$RemoteUrlSha256 = "",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._/-]*$")]
    [string]$Branch = "main",
    [ValidateRange(1, 65535)]
    [int]$Port = 18790,
    [ValidateRange(0, 20)]
    [int]$MaxRestarts = 5,
    [ValidateRange(1, 300)]
    [int]$RestartBaseDelaySeconds = 2,
    [ValidateRange(1, 3600)]
    [int]$RestartMaxDelaySeconds = 60,
    [ValidateRange(1, 86400)]
    [int]$StableRunSeconds = 300,
    [ValidateRange(5, 600)]
    [int]$UpgradeReadyTimeoutSeconds = 120,
    [ValidateRange(1, 60)]
    [int]$UpgradeReadyStableSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:GitExe = $null
$script:UvExe = $null
$script:CanonicalRepoPath = $null
$script:ControlDir = $null
$script:RequestPath = $null
$script:ResultPath = $null
$script:InProgressPath = $null
$script:ReadyPath = $null
$script:StopRequestPath = $null
$script:StoppedPath = $null
$script:WcferryLeasePath = $null
$script:LogPath = $null
$script:ControlProtocolVersion = 1
$script:GatewayStopTimeoutSeconds = 30

function Resolve-ApplicationPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$NameOrPath
    )

    if ([System.IO.Path]::IsPathRooted($NameOrPath)) {
        if (-not (Test-Path -LiteralPath $NameOrPath -PathType Leaf)) {
            throw "Executable not found: $NameOrPath"
        }
        return (Resolve-Path -LiteralPath $NameOrPath).Path
    }

    $command = Get-Command $NameOrPath -CommandType Application -ErrorAction Stop |
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

function Limit-Text {
    param(
        [AllowNull()]
        [string]$Text,
        [int]$MaxLength = 4000
    )

    if ([string]::IsNullOrEmpty($Text) -or $Text.Length -le $MaxLength) {
        return $Text
    }
    return $Text.Substring(0, $MaxLength) + "`n[truncated]"
}

function Protect-LogText {
    param(
        [AllowNull()]
        [string]$Text
    )

    if ([string]::IsNullOrEmpty($Text)) {
        return ""
    }
    $protected = $Text -replace '(?i)((?:https?|ssh)://)[^/\s@]+@', '$1[redacted]@'
    $protected = $protected -replace '(?i)\b(authorization|access[_-]?token|api[_-]?key|token)\s*[:=]\s*\S+', '$1=[redacted]'
    return Limit-Text -Text $protected
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
        throw "Updated supervisor does not parse under Windows PowerShell 5.1: $($parseErrors[0].Message)"
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
            throw "Updated supervisor is missing required parameter '$requiredName'."
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
                    throw "Updated supervisor adds mandatory parameter '$parameterName' that the scheduled task does not supply."
                }
            }
        }
    }
}

function Write-SupervisorLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [ValidateSet("INFO", "WARN", "ERROR")]
        [string]$Level = "INFO"
    )

    $line = "{0} [{1}] {2}" -f (Get-Date).ToString("o"), $Level, $Message
    Write-Host $line
    if ($script:LogPath) {
        try {
            Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
        } catch {
            Write-Warning "Unable to append supervisor log: $($_.Exception.Message)"
        }
    }
}

function Get-JsonPropertyValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Write-JsonAtomically {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [object]$Value
    )

    $tempPath = "$Path.$PID.tmp"
    $json = $Value | ConvertTo-Json -Depth 8
    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    try {
        [System.IO.File]::WriteAllText($tempPath, $json, $utf8WithoutBom)
        Install-FileAtomically -TemporaryPath $tempPath -DestinationPath $Path
    } finally {
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Read-GatewayStopRequest {
    if (-not (Test-Path -LiteralPath $script:StopRequestPath -PathType Leaf)) {
        return $null
    }

    $request = [System.IO.File]::ReadAllText($script:StopRequestPath) |
        ConvertFrom-Json
    if ($null -eq $request) {
        throw "Gateway stop request must contain a JSON object."
    }
    $schemaVersion = [int](Get-JsonPropertyValue -Object $request -Name "schema_version")
    $controlProtocolVersion = [int](Get-JsonPropertyValue `
        -Object $request `
        -Name "control_protocol_version")
    $instanceId = [string](Get-JsonPropertyValue -Object $request -Name "instance_id")
    $reason = [string](Get-JsonPropertyValue -Object $request -Name "reason")
    $requestedAt = [string](Get-JsonPropertyValue -Object $request -Name "requested_at")
    $parsedInstanceId = [Guid]::Empty
    $parsedRequestedAt = [DateTime]::MinValue
    if ($schemaVersion -ne 1 -or
        $controlProtocolVersion -ne $script:ControlProtocolVersion -or
        -not [Guid]::TryParse($instanceId, [ref]$parsedInstanceId) -or
        $parsedInstanceId.ToString("D") -ne $instanceId -or
        [string]::IsNullOrWhiteSpace($reason) -or
        [string]::IsNullOrWhiteSpace($requestedAt) -or
        -not [DateTime]::TryParse(
            $requestedAt,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::RoundtripKind,
            [ref]$parsedRequestedAt
        )) {
        throw "Gateway stop request is invalid."
    }
    return [pscustomobject]@{
        InstanceId = $instanceId
        Reason = $reason
        RequestedAt = $parsedRequestedAt
    }
}

function Read-GatewayStoppedAcknowledgement {
    if (-not (Test-Path -LiteralPath $script:StoppedPath -PathType Leaf)) {
        return $null
    }

    $acknowledgement = [System.IO.File]::ReadAllText($script:StoppedPath) |
        ConvertFrom-Json
    if ($null -eq $acknowledgement) {
        throw "Gateway stopped acknowledgement must contain a JSON object."
    }
    $schemaVersion = [int](Get-JsonPropertyValue `
        -Object $acknowledgement `
        -Name "schema_version")
    $controlProtocolVersion = [int](Get-JsonPropertyValue `
        -Object $acknowledgement `
        -Name "control_protocol_version")
    $instanceId = [string](Get-JsonPropertyValue `
        -Object $acknowledgement `
        -Name "instance_id")
    $stoppedAt = [string](Get-JsonPropertyValue `
        -Object $acknowledgement `
        -Name "stopped_at")
    $parsedInstanceId = [Guid]::Empty
    $parsedStoppedAt = [DateTime]::MinValue
    if ($schemaVersion -ne 1 -or
        $controlProtocolVersion -ne $script:ControlProtocolVersion -or
        -not [Guid]::TryParse($instanceId, [ref]$parsedInstanceId) -or
        $parsedInstanceId.ToString("D") -ne $instanceId -or
        [string]::IsNullOrWhiteSpace($stoppedAt) -or
        -not [DateTime]::TryParse(
            $stoppedAt,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::RoundtripKind,
            [ref]$parsedStoppedAt
        )) {
        throw "Gateway stopped acknowledgement is invalid."
    }
    return [pscustomobject]@{
        InstanceId = $instanceId
        StoppedAt = $parsedStoppedAt
    }
}

function Test-GatewayCleanupCommit {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstanceId,
        [switch]$RequireStopRequest
    )

    try {
        if (Test-Path -LiteralPath $script:WcferryLeasePath -PathType Leaf) {
            return $false
        }
        $acknowledgement = Read-GatewayStoppedAcknowledgement
        if ($null -eq $acknowledgement -or
            $acknowledgement.InstanceId -ne $InstanceId) {
            return $false
        }

        $request = Read-GatewayStopRequest
        if ($null -eq $request) {
            return -not $RequireStopRequest
        }
        if ($request.InstanceId -ne $InstanceId -or
            $request.Reason -in @("unproven_gateway_exit", "no_restart_exit")) {
            return $false
        }
        return $true
    } catch {
        return $false
    }
}

function Write-GatewayExitTombstone {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstanceId,
        [Parameter(Mandatory = $true)]
        [ValidateSet("unproven_gateway_exit", "no_restart_exit")]
        [string]$Reason
    )

    $parsedInstanceId = [Guid]::Empty
    if (-not [Guid]::TryParse($InstanceId, [ref]$parsedInstanceId) -or
        $parsedInstanceId.ToString("D") -ne $InstanceId) {
        throw "Cannot persist gateway exit tombstone because the instance id is invalid."
    }
    $request = [ordered]@{
        schema_version = 1
        control_protocol_version = $script:ControlProtocolVersion
        instance_id = $InstanceId
        reason = $Reason
        requested_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    Write-JsonAtomically -Path $script:StopRequestPath -Value $request
}

function Clear-StaleGatewayStopHandshake {
    if (Test-Path -LiteralPath $script:InProgressPath -PathType Leaf) {
        try {
            $marker = [System.IO.File]::ReadAllText($script:InProgressPath) |
                ConvertFrom-Json
            $phase = [string](Get-JsonPropertyValue -Object $marker -Name "phase")
        } catch {
            throw "Cannot prove that stale gateway stop files are safe to remove: $($_.Exception.Message)"
        }
        if ($phase -in @("candidate_starting", "candidate_running", "committing")) {
            throw "Refusing to clear gateway stop files while a candidate may be in flight."
        }
    }

    if (Test-Path -LiteralPath $script:WcferryLeasePath -PathType Leaf) {
        throw "Refusing gateway startup because the persistent WCFerry lease still exists: $script:WcferryLeasePath"
    }

    try {
        $stopRequest = Read-GatewayStopRequest
        $stoppedAcknowledgement = Read-GatewayStoppedAcknowledgement
    } catch {
        throw "Refusing gateway startup because stop-handshake evidence is invalid: $($_.Exception.Message)"
    }
    if ($null -ne $stopRequest -and
        ($stopRequest.Reason -in @("unproven_gateway_exit", "no_restart_exit") -or
        $null -eq $stoppedAcknowledgement -or
        $stoppedAcknowledgement.InstanceId -ne $stopRequest.InstanceId)) {
        throw "Refusing gateway startup because a previous stop request lacks matching native-cleanup proof."
    }

    foreach ($path in @($script:StopRequestPath, $script:StoppedPath)) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $path) {
            throw "Unable to remove stale gateway stop file: $path"
        }
    }
}

function Refresh-StableSupervisor {
    $sourcePath = Join-Path $script:CanonicalRepoPath "scripts\bubbles-supervisor.ps1"
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Updated repository does not contain the supervisor script."
    }

    $targetPath = [System.IO.Path]::GetFullPath($PSCommandPath)
    $resolvedSource = [System.IO.Path]::GetFullPath($sourcePath)
    if ($targetPath.Equals($resolvedSource, [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }

    $temporaryPath = "$targetPath.$PID.next"
    try {
        Copy-Item -LiteralPath $resolvedSource -Destination $temporaryPath -Force
        Assert-SupervisorScriptContract -Path $temporaryPath
        Install-FileAtomically `
            -TemporaryPath $temporaryPath `
            -DestinationPath $targetPath `
            -KeepBackup
    } finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Assert-GitWorktreeRoot {
    $rootResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
        "-C", $script:CanonicalRepoPath, "rev-parse", "--show-toplevel"
    )
    if ($rootResult.ExitCode -ne 0) {
        throw "RepoPath is not a Git worktree: $(Limit-Text $rootResult.Output)"
    }

    $reportedRoot = [System.IO.Path]::GetFullPath($rootResult.Output.Trim()).TrimEnd([char[]]"\/")
    $expectedRoot = $script:CanonicalRepoPath.TrimEnd([char[]]"\/")
    if (-not $reportedRoot.Equals($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "RepoPath must be the worktree root. Expected '$reportedRoot', got '$expectedRoot'."
    }
}

function Assert-RepositoryFiles {
    foreach ($requiredFile in @("pyproject.toml", "uv.lock")) {
        $path = Join-Path $script:CanonicalRepoPath $requiredFile
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required repository file is missing: $path"
        }
    }
}

function Assert-GatewayControlProtocol {
    $gatewayControlPath = Join-Path `
        $script:CanonicalRepoPath `
        "bubbles\gateway_control.py"
    if (-not (Test-Path -LiteralPath $gatewayControlPath -PathType Leaf)) {
        throw "Updated repository does not contain bubbles/gateway_control.py."
    }

    # This check deliberately reads source as data.  The currently running
    # supervisor must never import or execute code from an uncommitted candidate.
    $source = [System.IO.File]::ReadAllText($gatewayControlPath)
    $pattern = '(?m)^[ \t]*CONTROL_PROTOCOL_VERSION[ \t]*=[ \t]*(?<version>[0-9]+)[ \t]*(?:#[^\r\n]*)?\r?$'
    $protocolMatches = [regex]::Matches($source, $pattern)
    $candidateVersion = 0
    if ($protocolMatches.Count -ne 1 -or
        -not [int]::TryParse(
            $protocolMatches[0].Groups["version"].Value,
            [ref]$candidateVersion
        ) -or
        $candidateVersion -ne $script:ControlProtocolVersion) {
        throw "Updated gateway control protocol is not compatible with this supervisor."
    }
}

function Assert-RepositoryRoot {
    Assert-GitWorktreeRoot
    Assert-RepositoryFiles
}

function Assert-ConfiguredBranchName {
    $checkResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
        "check-ref-format", "--branch", $Branch
    )
    if ($checkResult.ExitCode -ne 0) {
        throw "Configured branch name is invalid."
    }
}

function Assert-ConfiguredBranch {
    $branchResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
        "-C", $script:CanonicalRepoPath, "branch", "--show-current"
    )
    if ($branchResult.ExitCode -ne 0) {
        throw "Unable to read current Git branch: $(Limit-Text $branchResult.Output)"
    }

    $currentBranch = $branchResult.Output.Trim()
    if ($currentBranch -ne $Branch) {
        throw "Upgrade and supervised startup require branch '$Branch'; current branch is '$currentBranch'."
    }
}

function Assert-CleanBranch {
    Assert-ConfiguredBranch

    $statusResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
        "-C", $script:CanonicalRepoPath, "status", "--porcelain=v1", "--untracked-files=normal"
    )
    if ($statusResult.ExitCode -ne 0) {
        throw "Unable to inspect Git worktree: $(Limit-Text $statusResult.Output)"
    }
    if (-not [string]::IsNullOrWhiteSpace($statusResult.Output)) {
        throw "Refusing upgrade because the Git worktree is not clean."
    }

    $commitResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
        "-C", $script:CanonicalRepoPath, "rev-parse", "HEAD"
    )
    if ($commitResult.ExitCode -ne 0) {
        throw "Unable to read current commit: $(Limit-Text $commitResult.Output)"
    }
    return $commitResult.Output.Trim()
}

function Read-UpgradeRequest {
    if (-not (Test-Path -LiteralPath $script:RequestPath -PathType Leaf)) {
        throw "Gateway exited with code 75 but upgrade-request.json is missing."
    }

    try {
        $request = [System.IO.File]::ReadAllText($script:RequestPath) | ConvertFrom-Json
    } catch {
        throw "upgrade-request.json is invalid JSON: $($_.Exception.Message)"
    }
    if ($null -eq $request) {
        throw "upgrade-request.json must contain a JSON object."
    }

    $schemaVersion = [int](Get-JsonPropertyValue -Object $request -Name "schema_version")
    $action = [string](Get-JsonPropertyValue -Object $request -Name "action")
    $requestId = [string](Get-JsonPropertyValue -Object $request -Name "request_id")
    $requestedAt = [string](Get-JsonPropertyValue -Object $request -Name "requested_at")
    $channel = [string](Get-JsonPropertyValue -Object $request -Name "channel")
    $chatId = [string](Get-JsonPropertyValue -Object $request -Name "chat_id")
    $requestedRemote = [string](Get-JsonPropertyValue -Object $request -Name "remote")
    $requestedBranch = [string](Get-JsonPropertyValue -Object $request -Name "branch")

    if ($schemaVersion -ne 1 -or $action -ne "upgrade") {
        throw "Upgrade request must use schema_version 1 and action 'upgrade'."
    }
    if ($requestedRemote -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*$") {
        throw "Upgrade request has an invalid remote name."
    }
    if ($requestedRemote -ne $Remote -or $requestedBranch -ne $Branch) {
        throw "Upgrade request target '$requestedRemote/$requestedBranch' does not match allowed target '$Remote/$Branch'."
    }
    if ([string]::IsNullOrWhiteSpace($requestId) -or $requestId.Length -gt 200) {
        throw "Upgrade request_id is missing or too long."
    }
    if ([string]::IsNullOrWhiteSpace($requestedAt)) {
        throw "Upgrade request must include requested_at."
    }
    if ([string]::IsNullOrWhiteSpace($channel) -or [string]::IsNullOrWhiteSpace($chatId)) {
        throw "Upgrade request must include channel and chat_id."
    }

    return [pscustomobject]@{
        RequestId = $requestId
        RequestedAt = $requestedAt
        Channel = $channel
        ChatId = $chatId
        Remote = $requestedRemote
        Branch = $requestedBranch
    }
}

function Assert-SafeRollbackState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OldRevision,
        [AllowEmptyString()]
        [string]$NewRevision = ""
    )

    $currentRevision = Assert-CleanBranch
    $allowedRevisions = @($OldRevision)
    if (-not [string]::IsNullOrWhiteSpace($NewRevision)) {
        $allowedRevisions += $NewRevision
    }
    if ($allowedRevisions -notcontains $currentRevision) {
        throw "Refusing rollback because HEAD is not an upgrade transaction revision."
    }
    return $currentRevision
}

function Start-GatewayProcess {
    param(
        [AllowEmptyString()]
        [string]$UpgradeRequestId = ""
    )

    # Keep repository and compatibility gates adjacent to the irreversible
    # process start. This also protects ordinary crash restarts after a manual
    # branch switch, worktree edit, or pull.
    $null = Assert-CleanBranch
    Assert-GatewayControlProtocol

    $instanceId = [Guid]::NewGuid().ToString("D")
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $script:UvExe
    $startInfo.Arguments = "run --no-sync bubbles gateway --port $Port"
    $startInfo.WorkingDirectory = $script:CanonicalRepoPath
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.EnvironmentVariables["BUBBLES_GATEWAY_SUPERVISED"] = "1"
    $startInfo.EnvironmentVariables["BUBBLES_GATEWAY_INSTANCE_ID"] = $instanceId
    if ([string]::IsNullOrWhiteSpace($UpgradeRequestId)) {
        [void]$startInfo.EnvironmentVariables.Remove("BUBBLES_GATEWAY_UPGRADE_REQUEST_ID")
    } else {
        $startInfo.EnvironmentVariables["BUBBLES_GATEWAY_UPGRADE_REQUEST_ID"] = $UpgradeRequestId
    }

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    Add-Member `
        -InputObject $process `
        -MemberType NoteProperty `
        -Name "GatewayInstanceId" `
        -Value $instanceId
    Add-Member `
        -InputObject $process `
        -MemberType NoteProperty `
        -Name "GatewayUvExe" `
        -Value $script:UvExe
    try {
        if (-not $process.Start()) {
            throw "Gateway process did not start."
        }
    } catch {
        $process.Dispose()
        throw
    }
    return $process
}

function Stop-GatewayProcess {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$InstanceId,
        [Parameter(Mandatory = $true)]
        [DateTime]$ExpectedStartTimeUtc,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedUvExe,
        [Parameter(Mandatory = $true)]
        [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
        [string]$Reason
    )

    $parsedInstanceId = [Guid]::Empty
    if (-not [Guid]::TryParse($InstanceId, [ref]$parsedInstanceId) -or
        $parsedInstanceId.ToString("D") -ne $InstanceId) {
        Write-SupervisorLog "Refusing gateway stop because the instance id is invalid." -Level "ERROR"
        return $false
    }
    if ([string]::IsNullOrWhiteSpace($ExpectedUvExe) -or
        -not [System.IO.Path]::IsPathRooted($ExpectedUvExe)) {
        Write-SupervisorLog "Refusing gateway stop because the expected executable path is invalid." -Level "ERROR"
        return $false
    }

    $expectedExecutable = [System.IO.Path]::GetFullPath($ExpectedUvExe)
    $configuredExecutable = [System.IO.Path]::GetFullPath($script:UvExe)
    if (-not $expectedExecutable.Equals(
        $configuredExecutable,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        Write-SupervisorLog "Refusing gateway stop because the expected executable is not the configured uv." -Level "ERROR"
        return $false
    }
    $expectedStart = $ExpectedStartTimeUtc.ToUniversalTime()

    $alreadyExited = $false
    try {
        $Process.Refresh()
        $alreadyExited = $Process.HasExited
    } catch {
        Write-SupervisorLog "Unable to inspect gateway process before graceful stop: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
        return $false
    }

    if (-not $alreadyExited) {
        try {
            $actualExecutable = [System.IO.Path]::GetFullPath($Process.MainModule.FileName)
            $actualStart = $Process.StartTime.ToUniversalTime()
        } catch {
            Write-SupervisorLog "Unable to validate gateway process identity: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
            return $false
        }
        if (-not $actualExecutable.Equals(
            $expectedExecutable,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or $actualStart.Ticks -ne $expectedStart.Ticks) {
            Write-SupervisorLog "Gateway PID belongs to a different process; refusing stop request." -Level "ERROR"
            return $false
        }

        $request = [ordered]@{
            schema_version = 1
            control_protocol_version = $script:ControlProtocolVersion
            instance_id = $InstanceId
            reason = $Reason
            requested_at = (Get-Date).ToUniversalTime().ToString("o")
        }
        try {
            Write-JsonAtomically -Path $script:StopRequestPath -Value $request
        } catch {
            Write-SupervisorLog "Unable to persist graceful gateway stop request: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
            return $false
        }
        Write-SupervisorLog "Requested graceful gateway stop for PID $($Process.Id), instance $InstanceId." -Level "WARN"
    } else {
        try {
            $exitCode = $Process.ExitCode
        } catch {
            $exitCode = $null
        }
        $requiresPriorStopRequest = $exitCode -eq 76
        if (Test-GatewayCleanupCommit `
            -InstanceId $InstanceId `
            -RequireStopRequest:$requiresPriorStopRequest) {
            Write-SupervisorLog "Gateway instance $InstanceId had already exited with matching native-cleanup proof."
            return $true
        }

        $tombstoneReason = $(if ($requiresPriorStopRequest) {
            "no_restart_exit"
        } else {
            "unproven_gateway_exit"
        })
        try {
            Write-GatewayExitTombstone `
                -InstanceId $InstanceId `
                -Reason $tombstoneReason
        } catch {
            Write-SupervisorLog "Unable to persist unsafe gateway exit tombstone: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
        }
        Write-SupervisorLog "Gateway PID $($Process.Id) exited without acceptable cleanup proof; refusing rollback or restart." -Level "ERROR"
        return $false
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($script:GatewayStopTimeoutSeconds)
    $processExited = $alreadyExited
    $exitCode = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not $processExited) {
            try {
                $Process.Refresh()
                $processExited = $Process.HasExited
            } catch {
                Write-SupervisorLog "Unable to observe gateway exit: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
                return $false
            }
        }
        if ($processExited) {
            try {
                $exitCode = $Process.ExitCode
            } catch {
                $exitCode = $null
            }
        }

        $acknowledged = Test-GatewayCleanupCommit `
            -InstanceId $InstanceId `
            -RequireStopRequest
        if ($processExited -and $acknowledged) {
            Write-SupervisorLog "Gateway instance $InstanceId confirmed native cleanup and exited safely."
            return $true
        }
        Start-Sleep -Milliseconds 250
    }

    try {
        $Process.Refresh()
        $processExited = $Process.HasExited
        if ($processExited) {
            $exitCode = $Process.ExitCode
        }
    } catch {
        Write-SupervisorLog "Unable to perform final gateway stop observation: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
        return $false
    }
    $acknowledged = Test-GatewayCleanupCommit `
        -InstanceId $InstanceId `
        -RequireStopRequest
    if ($processExited -and $acknowledged) {
        Write-SupervisorLog "Gateway instance $InstanceId confirmed native cleanup and exited safely."
        return $true
    }

    Write-SupervisorLog (
        "Graceful gateway stop timed out after {0}s (exited={1}, exit_code={2}, matching_ack={3}); refusing rollback or restart." -f
        $script:GatewayStopTimeoutSeconds,
        $processExited,
        $(if ($null -eq $exitCode) { "unknown" } else { $exitCode }),
        $acknowledged
    ) -Level "ERROR"
    return $false
}

function Wait-GatewayReady {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$RequestId
    )

    $readyDeadline = [DateTime]::UtcNow.AddSeconds($UpgradeReadyTimeoutSeconds)
    $stabilityDeadline = $null
    $readyObserved = $false
    $lastReadyError = ""
    while ($true) {
        $Process.Refresh()
        if ($Process.HasExited) {
            return [pscustomobject]@{
                Ready = $false
                Reason = $(if ($readyObserved) { "stability_exit" } else { "process_exit" })
                ExitCode = $Process.ExitCode
            }
        }

        $now = [DateTime]::UtcNow
        if ($readyObserved) {
            if ($now -ge $stabilityDeadline) {
                return [pscustomobject]@{
                    Ready = $true
                    Reason = "stable"
                    ExitCode = $null
                }
            }
            Start-Sleep -Milliseconds 250
            continue
        }

        if ($now -ge $readyDeadline) {
            break
        }

        if (Test-Path -LiteralPath $script:ReadyPath -PathType Leaf) {
            try {
                $ready = [System.IO.File]::ReadAllText($script:ReadyPath) | ConvertFrom-Json
                $readySchemaVersion = [int](Get-JsonPropertyValue -Object $ready -Name "schema_version")
                $readyControlProtocolVersion = [int](Get-JsonPropertyValue `
                    -Object $ready `
                    -Name "control_protocol_version")
                if ($readySchemaVersion -ne 1 -or
                    $readyControlProtocolVersion -ne $script:ControlProtocolVersion) {
                    throw "gateway-ready.json has an unsupported control protocol."
                }
                $readyRequestId = [string](Get-JsonPropertyValue -Object $ready -Name "request_id")
                if ($readyRequestId -eq $RequestId) {
                    Remove-Item -LiteralPath $script:ReadyPath -Force -ErrorAction Stop
                    $readyObserved = $true
                    $stabilityDeadline = [DateTime]::UtcNow.AddSeconds($UpgradeReadyStableSeconds)
                    Write-SupervisorLog "Candidate gateway reported readiness; requiring $UpgradeReadyStableSeconds stable second(s)."
                    continue
                }
            } catch {
                $readyError = Protect-LogText $_.Exception.Message
                if ($readyError -ne $lastReadyError) {
                    Write-SupervisorLog "Ignoring unreadable gateway-ready.json: $readyError" -Level "WARN"
                    $lastReadyError = $readyError
                }
            }
        }
        Start-Sleep -Milliseconds 250
    }

    $Process.Refresh()
    if ($Process.HasExited) {
        return [pscustomobject]@{
            Ready = $false
            Reason = "process_exit"
            ExitCode = $Process.ExitCode
        }
    }
    return [pscustomobject]@{
        Ready = $false
        Reason = "timeout"
        ExitCode = $null
    }
}

function Stop-RecoveredCandidate {
    param(
        [AllowEmptyString()]
        [string]$Phase = "",
        [AllowNull()]
        [object]$CandidatePid,
        [AllowEmptyString()]
        [string]$CandidateInstanceId = "",
        [AllowEmptyString()]
        [string]$CandidateStartTimeUtc = "",
        [AllowEmptyString()]
        [string]$CandidateUvExe = ""
    )

    if ([string]::IsNullOrWhiteSpace($Phase) -or $Phase -eq "updating") {
        return
    }
    if ($Phase -eq "candidate_starting") {
        throw "Candidate startup was interrupted before process identity was persisted; manual recovery is required."
    }
    if ($Phase -notin @("candidate_running", "committing")) {
        throw "Transaction marker has an unsupported phase."
    }

    $parsedCandidateInstanceId = [Guid]::Empty
    if (-not [Guid]::TryParse($CandidateInstanceId, [ref]$parsedCandidateInstanceId) -or
        $parsedCandidateInstanceId.ToString("D") -ne $CandidateInstanceId) {
        throw "Running candidate marker has an invalid instance id."
    }

    $candidatePidValue = 0
    if (-not [int]::TryParse([string]$CandidatePid, [ref]$candidatePidValue) -or
        $candidatePidValue -le 0) {
        throw "Running candidate marker has an invalid PID."
    }
    if ([string]::IsNullOrWhiteSpace($CandidateUvExe) -or
        -not [System.IO.Path]::IsPathRooted($CandidateUvExe)) {
        throw "Running candidate marker has an invalid uv executable path."
    }

    $markerUvExe = [System.IO.Path]::GetFullPath($CandidateUvExe)
    $configuredUvExe = [System.IO.Path]::GetFullPath($script:UvExe)
    if (-not $markerUvExe.Equals($configuredUvExe, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Running candidate marker does not match the configured uv executable."
    }

    $expectedStartTime = [DateTime]::MinValue
    if (-not [DateTime]::TryParse(
        $CandidateStartTimeUtc,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [System.Globalization.DateTimeStyles]::RoundtripKind,
        [ref]$expectedStartTime
    )) {
        throw "Running candidate marker has an invalid UTC start time."
    }
    $expectedStartTime = $expectedStartTime.ToUniversalTime()

    $candidateProcess = $null
    try {
        try {
            $candidateProcess = [System.Diagnostics.Process]::GetProcessById($candidatePidValue)
        } catch [System.ArgumentException] {
            if (Test-GatewayCleanupCommit -InstanceId $CandidateInstanceId) {
                Write-SupervisorLog "Recorded candidate PID $candidatePidValue no longer exists and has matching native-cleanup proof."
                return
            }
            throw "Recorded candidate PID no longer exists without a matching graceful-stop acknowledgement."
        }

        try {
            $candidateProcess.Refresh()
            if ($candidateProcess.HasExited) {
                $candidateExitCode = $candidateProcess.ExitCode
                if (Test-GatewayCleanupCommit `
                    -InstanceId $CandidateInstanceId `
                    -RequireStopRequest:($candidateExitCode -eq 76)) {
                    Write-SupervisorLog "Recorded candidate PID $candidatePidValue exited safely with matching native-cleanup proof."
                    return
                }
                try {
                    Write-GatewayExitTombstone `
                        -InstanceId $CandidateInstanceId `
                        -Reason $(if ($candidateExitCode -eq 76) {
                            "no_restart_exit"
                        } else {
                            "unproven_gateway_exit"
                        })
                } catch {
                    Write-SupervisorLog "Unable to persist recovered candidate tombstone: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
                }
                throw "Recorded candidate exited without the required graceful-stop proof."
            }
            $actualUvExe = [System.IO.Path]::GetFullPath($candidateProcess.MainModule.FileName)
            $actualStartTime = $candidateProcess.StartTime.ToUniversalTime()
        } catch [System.InvalidOperationException] {
            if (Test-GatewayCleanupCommit -InstanceId $CandidateInstanceId) {
                Write-SupervisorLog "Recorded candidate exited during validation and has matching native-cleanup proof."
                return
            }
            throw "Recorded candidate exited during validation without a matching graceful-stop acknowledgement."
        }
        if (-not $actualUvExe.Equals($markerUvExe, [System.StringComparison]::OrdinalIgnoreCase) -or
            $actualStartTime.Ticks -ne $expectedStartTime.Ticks) {
            throw "Recorded candidate PID belongs to a different process; refusing stop and rollback."
        }

        Write-SupervisorLog "Recovered the exact candidate process identity for PID $candidatePidValue."
        if (-not (Stop-GatewayProcess `
            -Process $candidateProcess `
            -InstanceId $CandidateInstanceId `
            -ExpectedStartTimeUtc $expectedStartTime `
            -ExpectedUvExe $markerUvExe `
            -Reason "upgrade_recovery")) {
            throw "Recovered candidate could not prove graceful native cleanup."
        }
    } finally {
        if ($null -ne $candidateProcess) {
            $candidateProcess.Dispose()
        }
    }
}

function Restore-UpgradeRevision {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Revision
    )

    $resetResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
        "-C", $script:CanonicalRepoPath, "reset", "--hard", $Revision
    )
    if ($resetResult.ExitCode -ne 0) {
        Write-SupervisorLog (
            "Rollback git reset failed with exit code {0}: {1}" -f
            $resetResult.ExitCode, (Protect-LogText $resetResult.Output)
        ) -Level "ERROR"
        return [pscustomobject]@{ Succeeded = $false; Stage = "git_reset"; ExitCode = $resetResult.ExitCode }
    }

    try {
        Assert-RepositoryRoot
        Assert-ConfiguredBranch
    } catch {
        Write-SupervisorLog "Rollback repository validation failed: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
        return [pscustomobject]@{ Succeeded = $false; Stage = "repository"; ExitCode = $null }
    }

    $syncResult = Invoke-NativeCapture -FilePath $script:UvExe -Arguments @(
        "sync", "--locked"
    )
    if ($syncResult.ExitCode -ne 0) {
        Write-SupervisorLog (
            "Rollback uv sync failed with exit code {0}: {1}" -f
            $syncResult.ExitCode, (Protect-LogText $syncResult.Output)
        ) -Level "ERROR"
        return [pscustomobject]@{ Succeeded = $false; Stage = "uv_sync"; ExitCode = $syncResult.ExitCode }
    }

    try {
        $restoredRevision = Assert-CleanBranch
    } catch {
        Write-SupervisorLog "Rollback validation failed: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
        return [pscustomobject]@{ Succeeded = $false; Stage = "validation"; ExitCode = $null }
    }
    if ($restoredRevision -ne $Revision) {
        Write-SupervisorLog "Rollback validation found an unexpected revision." -Level "ERROR"
        return [pscustomobject]@{ Succeeded = $false; Stage = "revision"; ExitCode = $null }
    }

    return [pscustomobject]@{ Succeeded = $true; Stage = "complete"; ExitCode = 0 }
}

function Recover-UnfinishedUpgrade {
    Write-SupervisorLog "Found an unfinished upgrade transaction; attempting automatic rollback." -Level "WARN"
    try {
        $marker = [System.IO.File]::ReadAllText($script:InProgressPath) | ConvertFrom-Json
        $schemaVersion = [int](Get-JsonPropertyValue -Object $marker -Name "schema_version")
        $requestId = [string](Get-JsonPropertyValue -Object $marker -Name "request_id")
        $oldRevision = [string](Get-JsonPropertyValue -Object $marker -Name "old_revision")
        $newRevision = [string](Get-JsonPropertyValue -Object $marker -Name "new_revision")
        $phase = [string](Get-JsonPropertyValue -Object $marker -Name "phase")
        $candidatePid = Get-JsonPropertyValue -Object $marker -Name "candidate_pid"
        $candidateInstanceId = [string](Get-JsonPropertyValue -Object $marker -Name "candidate_instance_id")
        $candidateStartTimeUtc = [string](Get-JsonPropertyValue -Object $marker -Name "candidate_start_time_utc")
        $candidateUvExe = [string](Get-JsonPropertyValue -Object $marker -Name "candidate_uv_exe")
        $markerRepoPath = [string](Get-JsonPropertyValue -Object $marker -Name "repo_path")
        $markerRemote = [string](Get-JsonPropertyValue -Object $marker -Name "remote")
        $markerBranch = [string](Get-JsonPropertyValue -Object $marker -Name "branch")
        $channel = [string](Get-JsonPropertyValue -Object $marker -Name "channel")
        $chatId = [string](Get-JsonPropertyValue -Object $marker -Name "chat_id")
        $requestedAt = [string](Get-JsonPropertyValue -Object $marker -Name "requested_at")
        $transactionStartedAt = [string](Get-JsonPropertyValue -Object $marker -Name "started_at")
        if ($schemaVersion -ne 1) {
            throw "Transaction marker has an unsupported schema_version."
        }
        if ([string]::IsNullOrWhiteSpace($requestId) -or $requestId.Length -gt 200) {
            throw "Transaction marker has an invalid request_id."
        }
        if ($oldRevision -notmatch "^[A-Fa-f0-9]{40,64}$") {
            throw "Transaction marker has an invalid old_revision."
        }
        if (-not [string]::IsNullOrWhiteSpace($newRevision) -and
            $newRevision -notmatch "^[A-Fa-f0-9]{40,64}$") {
            throw "Transaction marker has an invalid new_revision."
        }
        if ([string]::IsNullOrWhiteSpace($markerRepoPath) -or
            -not [System.IO.Path]::IsPathRooted($markerRepoPath)) {
            throw "Transaction marker has an invalid repo_path."
        }
        $markerCanonicalRepoPath = [System.IO.Path]::GetFullPath($markerRepoPath)
        $markerCanonicalRepoPath = $markerCanonicalRepoPath.TrimEnd([char[]]"\/")
        $expectedCanonicalRepoPath = $script:CanonicalRepoPath.TrimEnd([char[]]"\/")
        if (-not $markerCanonicalRepoPath.Equals(
            $expectedCanonicalRepoPath,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Transaction marker belongs to a different Bubblebot repository."
        }
        if ($markerRemote -ne $Remote -or $markerBranch -ne $Branch) {
            throw "Transaction marker target does not match the installed target."
        }

        Stop-RecoveredCandidate `
            -Phase $phase `
            -CandidatePid $candidatePid `
            -CandidateInstanceId $candidateInstanceId `
            -CandidateStartTimeUtc $candidateStartTimeUtc `
            -CandidateUvExe $candidateUvExe

        $currentRevision = Assert-SafeRollbackState `
            -OldRevision $oldRevision `
            -NewRevision $newRevision
        Write-SupervisorLog "Interrupted transaction HEAD $currentRevision is safe to roll back."

        $rollbackResult = Restore-UpgradeRevision -Revision $oldRevision
        if (-not $rollbackResult.Succeeded) {
            throw "Automatic rollback failed during $($rollbackResult.Stage)."
        }

        # A crash may have happened after the stable copy was refreshed but before
        # the transaction marker was cleared. Restore the old script alongside code.
        Refresh-StableSupervisor
        Remove-Item -LiteralPath $script:ReadyPath -Force -ErrorAction SilentlyContinue

        if ([string]::IsNullOrWhiteSpace($channel) -or
            [string]::IsNullOrWhiteSpace($chatId)) {
            try {
                $request = Read-UpgradeRequest
                $requestId = $request.RequestId
                $channel = $request.Channel
                $chatId = $request.ChatId
                $requestedAt = $request.RequestedAt
            } catch {
                Write-SupervisorLog "Interrupted-upgrade request metadata is unavailable: $(Protect-LogText $_.Exception.Message)" -Level "WARN"
            }
        }

        $result = [ordered]@{
            schema_version = 1
            request_id = $requestId
            channel = $channel
            chat_id = $chatId
            ok = $false
            remote = $Remote
            branch = $Branch
            old_revision = $oldRevision
            new_revision = $oldRevision
            error = "检测到上次升级异常中断，已自动回滚并恢复原版本。"
            requested_at = $requestedAt
            started_at = $(if ([string]::IsNullOrWhiteSpace($transactionStartedAt)) {
                (Get-Date).ToUniversalTime().ToString("o")
            } else {
                $transactionStartedAt
            })
            finished_at = (Get-Date).ToUniversalTime().ToString("o")
            repo_path = $script:CanonicalRepoPath
            failure_stage = "interrupted_upgrade"
            exit_code = $null
            restart_pending = $true
        }
        $resultWritten = $false
        if (-not [string]::IsNullOrWhiteSpace($channel) -and
            -not [string]::IsNullOrWhiteSpace($chatId)) {
            try {
                Write-JsonAtomically -Path $script:ResultPath -Value $result
                $resultWritten = $true
            } catch {
                Write-SupervisorLog "Unable to write interrupted-upgrade result: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
            }
        } else {
            Write-SupervisorLog "Interrupted-upgrade result is not routable; no result file was written." -Level "WARN"
            $resultWritten = $true
        }
        if (-not $resultWritten) {
            throw "Automatic rollback result could not be persisted."
        }
        Remove-Item -LiteralPath $script:InProgressPath -Force -ErrorAction Stop
        Remove-Item -LiteralPath $script:RequestPath -Force -ErrorAction SilentlyContinue
        Write-SupervisorLog "Automatic rollback completed at commit $oldRevision."
        return $true
    } catch {
        Write-SupervisorLog "Automatic rollback of interrupted upgrade failed: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

function Invoke-StartupSync {
    # A stable supervisor may outlive the checkout it launches.  Validate the
    # gateway protocol before uv can execute anything from a manually updated
    # worktree, then validate it again immediately before process startup.
    Assert-GatewayControlProtocol
    $revisionBefore = Assert-CleanBranch
    $syncResult = Invoke-NativeCapture -FilePath $script:UvExe -Arguments @(
        "sync", "--locked"
    )
    if ($syncResult.ExitCode -ne 0) {
        Write-SupervisorLog (
            "Startup uv sync failed with exit code {0}: {1}" -f
            $syncResult.ExitCode, (Protect-LogText $syncResult.Output)
        ) -Level "ERROR"
        throw "Startup uv sync --locked failed."
    }
    $revisionAfter = Assert-CleanBranch
    if ($revisionAfter -ne $revisionBefore) {
        throw "Repository revision changed during startup dependency sync."
    }
    Assert-GatewayControlProtocol
}

function Invoke-Upgrade {
    $startedAt = (Get-Date).ToUniversalTime()
    $result = [ordered]@{
        schema_version = 1
        request_id = $null
        channel = $null
        chat_id = $null
        ok = $false
        remote = $Remote
        branch = $Branch
        old_revision = $null
        new_revision = $null
        error = $null
        requested_at = $null
        started_at = $startedAt.ToString("o")
        finished_at = $null
        repo_path = $script:CanonicalRepoPath
        failure_stage = $null
        exit_code = $null
        restart_pending = $false
    }

    $stage = "request_validation"
    $nativeExitCode = $null
    $oldRevision = $null
    $newRevision = $null
    $transactionMarked = $false
    $marker = $null
    $candidateProcess = $null
    $candidateStartedAt = $null
    $readyWait = $null
    $successResultPersisted = $false

    try {
        $request = Read-UpgradeRequest
        $result.request_id = $request.RequestId
        $result.requested_at = $request.RequestedAt
        $result.channel = $request.Channel
        $result.chat_id = $request.ChatId
        $result.remote = $request.Remote
        $result.branch = $request.Branch

        $stage = "target_validation"
        $remoteResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
            "-C", $script:CanonicalRepoPath, "remote", "get-url", $request.Remote
        )
        if ($remoteResult.ExitCode -ne 0) {
            $nativeExitCode = $remoteResult.ExitCode
            throw "Configured Git remote does not exist."
        }
        if ([string]::IsNullOrWhiteSpace($RemoteUrlSha256)) {
            throw "Configured Git remote URL is not pinned by the installer."
        }
        $currentRemoteHash = Get-TextSha256 -Text $remoteResult.Output.Trim()
        if ($currentRemoteHash -ne $RemoteUrlSha256.ToLowerInvariant()) {
            throw "Configured Git remote URL changed after task installation."
        }

        $stage = "repository_validation"
        $oldRevision = Assert-CleanBranch
        $result.old_revision = $oldRevision

        $stage = "transaction_marker"
        $marker = [ordered]@{
            schema_version = 1
            request_id = $request.RequestId
            channel = $request.Channel
            chat_id = $request.ChatId
            requested_at = $request.RequestedAt
            old_revision = $oldRevision
            new_revision = $null
            phase = "updating"
            candidate_pid = $null
            candidate_instance_id = $null
            candidate_start_time_utc = $null
            candidate_uv_exe = $script:UvExe
            repo_path = $script:CanonicalRepoPath
            remote = $request.Remote
            branch = $request.Branch
            started_at = $startedAt.ToString("o")
        }
        Write-JsonAtomically -Path $script:InProgressPath -Value $marker
        $transactionMarked = $true

        Write-SupervisorLog "Applying upgrade from remote '$($request.Remote)' branch '$($request.Branch)'."
        $stage = "git_pull"
        $pullResult = Invoke-NativeCapture -FilePath $script:GitExe -Arguments @(
            "-C", $script:CanonicalRepoPath,
            "-c", "http.lowSpeedLimit=1",
            "-c", "http.lowSpeedTime=30",
            "pull", "--ff-only", $request.Remote, $request.Branch
        )
        if ($pullResult.ExitCode -ne 0) {
            $nativeExitCode = $pullResult.ExitCode
            Write-SupervisorLog (
                "git pull failed with exit code {0}: {1}" -f
                $pullResult.ExitCode, (Protect-LogText $pullResult.Output)
            ) -Level "ERROR"
            throw "git pull --ff-only failed."
        }

        $stage = "transaction_update"
        $newRevision = Assert-CleanBranch
        $result.new_revision = $newRevision
        $marker["new_revision"] = $newRevision
        Write-JsonAtomically -Path $script:InProgressPath -Value $marker

        $stage = "post_pull_validation"
        Assert-RepositoryRoot

        $stage = "control_protocol_validation"
        Assert-GatewayControlProtocol

        $stage = "dependency_sync"
        $syncResult = Invoke-NativeCapture -FilePath $script:UvExe -Arguments @(
            "sync", "--locked"
        )
        if ($syncResult.ExitCode -ne 0) {
            $nativeExitCode = $syncResult.ExitCode
            Write-SupervisorLog (
                "uv sync failed with exit code {0}: {1}" -f
                $syncResult.ExitCode, (Protect-LogText $syncResult.Output)
            ) -Level "ERROR"
            throw "uv sync --locked failed."
        }

        $stage = "final_validation"
        Assert-RepositoryRoot
        $finalRevision = Assert-CleanBranch
        if ($finalRevision -ne $newRevision) {
            throw "Repository revision changed during upgrade dependency sync."
        }

        $stage = "control_protocol_revalidation"
        Assert-GatewayControlProtocol

        $stage = "supervisor_refresh"
        Refresh-StableSupervisor

        $stage = "gateway_start"
        foreach ($stalePath in @($script:ReadyPath, $script:ResultPath)) {
            if (Test-Path -LiteralPath $stalePath) {
                Remove-Item -LiteralPath $stalePath -Force -ErrorAction Stop
            }
        }
        Clear-StaleGatewayStopHandshake

        $stage = "candidate_marker_starting"
        $marker["phase"] = "candidate_starting"
        $marker["candidate_pid"] = $null
        $marker["candidate_instance_id"] = $null
        $marker["candidate_start_time_utc"] = $null
        $marker["candidate_uv_exe"] = $script:UvExe
        Write-JsonAtomically -Path $script:InProgressPath -Value $marker

        $stage = "gateway_start"
        $candidateProcess = Start-GatewayProcess -UpgradeRequestId $request.RequestId
        $candidateStartedAt = $candidateProcess.StartTime.ToUniversalTime()

        $stage = "candidate_marker_running"
        $marker["phase"] = "candidate_running"
        $marker["candidate_pid"] = $candidateProcess.Id
        $marker["candidate_instance_id"] = $candidateProcess.GatewayInstanceId
        $marker["candidate_start_time_utc"] = $candidateStartedAt.ToUniversalTime().ToString("o")
        $marker["candidate_uv_exe"] = $script:UvExe
        Write-JsonAtomically -Path $script:InProgressPath -Value $marker
        Write-SupervisorLog "Candidate gateway started with PID $($candidateProcess.Id); waiting for readiness."

        $stage = "gateway_ready"
        $readyWait = Wait-GatewayReady `
            -Process $candidateProcess `
            -RequestId $request.RequestId
        if (-not $readyWait.Ready) {
            $nativeExitCode = $readyWait.ExitCode
            if ($readyWait.Reason -in @("process_exit", "stability_exit")) {
                throw "Candidate gateway exited during readiness validation."
            }
            throw "Candidate gateway readiness timed out."
        }

        $stage = "transaction_commit"
        Remove-Item -LiteralPath $script:RequestPath -Force -ErrorAction Stop
        $marker["phase"] = "committing"
        Write-JsonAtomically -Path $script:InProgressPath -Value $marker
        $result.ok = $true
        $result.restart_pending = $true
        $result.finished_at = (Get-Date).ToUniversalTime().ToString("o")
        Write-JsonAtomically -Path $script:ResultPath -Value $result
        $successResultPersisted = $true
        Remove-Item -LiteralPath $script:InProgressPath -Force -ErrorAction Stop
        $transactionMarked = $false
        Write-SupervisorLog "Upgrade committed at $newRevision; continuing to supervise PID $($candidateProcess.Id)."
        return [pscustomobject]@{
            CanContinue = $true
            GatewayProcess = $candidateProcess
            GatewayInstanceId = $candidateProcess.GatewayInstanceId
            StartedAt = $candidateStartedAt
        }
    } catch {
        $failureDetail = Protect-LogText $_.Exception.Message
        Write-SupervisorLog "Upgrade failed during ${stage}: $failureDetail" -Level "ERROR"
        $result.ok = $false
        $result.failure_stage = $stage
        $result.exit_code = $nativeExitCode

        $candidateStopped = $true
        if ($null -ne $candidateProcess) {
            try {
                $candidateStopped = Stop-GatewayProcess `
                    -Process $candidateProcess `
                    -InstanceId $candidateProcess.GatewayInstanceId `
                    -ExpectedStartTimeUtc $candidateStartedAt `
                    -ExpectedUvExe $candidateProcess.GatewayUvExe `
                    -Reason "upgrade_rollback"
            } catch {
                $candidateStopped = $false
                Write-SupervisorLog "Candidate gateway termination failed: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
            } finally {
                $candidateProcess.Dispose()
            }
        }

        if ($candidateStopped -and $transactionMarked -and $null -ne $marker -and
            $marker["phase"] -in @("candidate_starting", "candidate_running", "committing")) {
            try {
                $marker["phase"] = "updating"
                $marker["candidate_pid"] = $null
                $marker["candidate_instance_id"] = $null
                $marker["candidate_start_time_utc"] = $null
                Write-JsonAtomically -Path $script:InProgressPath -Value $marker
            } catch {
                Write-SupervisorLog "Unable to reset candidate marker after process termination: $(Protect-LogText $_.Exception.Message)" -Level "WARN"
            }
        }

        $rolledBack = $false
        # Never restart a checkout that failed before its branch, cleanliness,
        # and revision were proven. In particular, repository_validation must
        # stop the supervisor instead of launching the rejected worktree.
        $canRestart = $candidateStopped -and -not [string]::IsNullOrWhiteSpace($oldRevision)
        if ($canRestart -and $transactionMarked -and $oldRevision) {
            try {
                $rollbackHead = Assert-SafeRollbackState `
                    -OldRevision $oldRevision `
                    -NewRevision $newRevision
                Write-SupervisorLog "Attempting rollback from $rollbackHead to $oldRevision." -Level "WARN"
                $rollbackResult = Restore-UpgradeRevision -Revision $oldRevision
                if (-not $rollbackResult.Succeeded) {
                    throw "Rollback failed during $($rollbackResult.Stage)."
                }
                Refresh-StableSupervisor
                Remove-Item -LiteralPath $script:ReadyPath -Force -ErrorAction SilentlyContinue
                $rolledBack = $true
                $result.new_revision = $oldRevision
            } catch {
                $canRestart = $false
                Write-SupervisorLog "Upgrade rollback could not be completed safely: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
            }
        }

        if (-not $candidateStopped) {
            $result.error = "新版本 gateway 无法安全终止；supervisor 已停止，请检查 Windows 本机日志。"
        } elseif ($stage -eq "repository_validation") {
            $result.error = "仓库不在配置分支或工作树不干净；supervisor 已停止，未从该 checkout 重新启动。"
        } elseif ($stage -eq "target_validation") {
            $result.error = "升级目标校验失败；supervisor 已停止，未重新启动 gateway。"
        } elseif ($stage -eq "request_validation") {
            $result.error = "升级请求校验失败；supervisor 已停止，未重新启动 gateway。"
        } elseif (-not $canRestart) {
            $result.error = "升级失败且自动回滚未完成；supervisor 已停止，请检查 Windows 本机日志。"
        } elseif ($stage -eq "gateway_ready") {
            if ($readyWait -and $readyWait.Reason -in @("process_exit", "stability_exit")) {
                $result.error = "新版本 gateway 在就绪验证期间退出（退出码 $nativeExitCode），已回滚到升级前版本。"
            } else {
                $result.error = "新版本 gateway 在 $UpgradeReadyTimeoutSeconds 秒内未就绪，已回滚到升级前版本。"
            }
        } elseif ($stage -eq "gateway_start") {
            $result.error = "新版本 gateway 启动失败，已回滚到升级前版本。"
        } elseif ($stage -in @("candidate_marker_starting", "candidate_marker_running")) {
            $result.error = "候选 gateway 的进程身份无法安全持久化，已终止并回滚到升级前版本。"
        } elseif ($stage -eq "transaction_commit") {
            $result.error = "新版本 gateway 已就绪但事务提交失败，已终止并回滚到升级前版本。"
        } elseif ($stage -eq "dependency_sync") {
            $result.error = "依赖同步失败（退出码 $nativeExitCode），已回滚到升级前版本。"
        } elseif ($stage -eq "git_pull") {
            if ($rolledBack) {
                $result.error = "代码拉取失败（退出码 $nativeExitCode），已回滚到升级前版本。"
            } else {
                $result.error = "代码拉取失败（退出码 $nativeExitCode），工作树未发生变化。"
            }
        } elseif ($stage -eq "supervisor_refresh") {
            $result.error = "稳定 supervisor 刷新失败，已回滚到升级前版本。"
        } elseif ($stage -in @("control_protocol_validation", "control_protocol_revalidation")) {
            $result.error = "新版本 gateway 控制协议与当前 supervisor 不兼容，已回滚到升级前版本。"
        } elseif ($stage -in @("post_pull_validation", "transaction_update", "final_validation")) {
            $result.error = "升级后的仓库校验失败，已回滚到升级前版本。"
        } elseif ($stage -eq "transaction_marker") {
            $result.error = "无法建立升级事务标记，升级未执行。"
        } else {
            $result.error = "升级请求校验失败，升级未执行。"
        }

        if ($transactionMarked) {
            if ($rolledBack) {
                try {
                    if ($successResultPersisted -and
                        (Test-Path -LiteralPath $script:ResultPath -PathType Leaf)) {
                        Remove-Item -LiteralPath $script:ResultPath -Force -ErrorAction Stop
                        $successResultPersisted = $false
                    }
                    Remove-Item -LiteralPath $script:InProgressPath -Force -ErrorAction Stop
                    $transactionMarked = $false
                } catch {
                    $canRestart = $false
                    Write-SupervisorLog "Unable to clear rolled-back upgrade marker; supervisor will stop." -Level "ERROR"
                }
            } else {
                $canRestart = $false
            }
        }

        $result.restart_pending = $canRestart
        $result.finished_at = (Get-Date).ToUniversalTime().ToString("o")
        $resultWritten = $false
        if (-not [string]::IsNullOrWhiteSpace($result.channel) -and
            -not [string]::IsNullOrWhiteSpace($result.chat_id)) {
            try {
                Write-JsonAtomically -Path $script:ResultPath -Value $result
                $resultWritten = $true
            } catch {
                Write-SupervisorLog "Unable to write failed upgrade result; request metadata was retained: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
            }
        } else {
            Write-SupervisorLog "Failed upgrade result is not routable; request metadata was retained." -Level "WARN"
        }
        if ($resultWritten -and -not $transactionMarked) {
            Remove-Item -LiteralPath $script:RequestPath -Force -ErrorAction SilentlyContinue
        }
        return [pscustomobject]@{
            CanContinue = $canRestart
            GatewayProcess = $null
            GatewayInstanceId = $null
            StartedAt = $null
        }
    }
}

function Get-RestartDelaySeconds {
    param([int]$Attempt)

    $delay = $RestartBaseDelaySeconds * [Math]::Pow(2, [Math]::Max(0, $Attempt - 1))
    return [int][Math]::Min($RestartMaxDelaySeconds, $delay)
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "bubbles-supervisor.ps1 supports Windows only."
}
if ($PSVersionTable.PSEdition -ne "Desktop" -or $PSVersionTable.PSVersion.Major -ne 5) {
    throw "The supervisor must run under Windows PowerShell 5.1."
}
if ([Environment]::Is64BitOperatingSystem -and -not [Environment]::Is64BitProcess) {
    throw "The supervisor must run under 64-bit Windows PowerShell on 64-bit Windows."
}
if (-not [Environment]::UserInteractive) {
    throw "The supervisor must run in the current user's interactive session."
}

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
if ($identity.IsSystem) {
    throw "The supervisor must not run as LocalSystem."
}

$script:CanonicalRepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
$script:GitExe = Resolve-ApplicationPath $GitPath
$script:UvExe = Resolve-ApplicationPath $UvPath

$userProfile = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
if ([string]::IsNullOrWhiteSpace($userProfile)) {
    $userProfile = $HOME
}
$script:ControlDir = Join-Path (Join-Path $userProfile ".bubbles") "control"
[void](New-Item -ItemType Directory -Path $script:ControlDir -Force)
$script:RequestPath = Join-Path $script:ControlDir "upgrade-request.json"
$script:ResultPath = Join-Path $script:ControlDir "upgrade-result.json"
$script:InProgressPath = Join-Path $script:ControlDir "upgrade-in-progress.json"
$script:ReadyPath = Join-Path $script:ControlDir "gateway-ready.json"
$script:StopRequestPath = Join-Path $script:ControlDir "gateway-stop-request.json"
$script:StoppedPath = Join-Path $script:ControlDir "gateway-stopped.json"
$script:WcferryLeasePath = Join-Path $script:ControlDir "wcferry-lease.json"
$script:LogPath = Join-Path $script:ControlDir "supervisor.log"

Assert-ConfiguredBranchName
Assert-GitWorktreeRoot

if ($null -eq $identity.User -or [string]::IsNullOrWhiteSpace($identity.User.Value)) {
    throw "Unable to resolve the current Windows user SID for supervisor locking."
}

$supervisorMutexName = "Global\BubblebotSupervisor-$($identity.User.Value)"
$mutex = $null
$ownsMutex = $false
try {
    try {
        # The control directory and WCFerry lease are shared by every Bubblebot
        # clone for this Windows user, so supervisor ownership must use the same
        # scope.  A Global mutex also spans console and RDP sessions.  If the
        # previous supervisor died, Windows transfers abandoned ownership
        # atomically; the persistent WCFerry lease then decides whether startup
        # is safe, so no second gateway is started on an unproven native state.
        $mutex = [System.Threading.Mutex]::new($false, $supervisorMutexName)
        $ownsMutex = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $ownsMutex = $true
        Write-SupervisorLog "Recovered abandoned per-user supervisor ownership; validating persistent gateway state before startup." -Level "WARN"
    } catch {
        throw "Unable to acquire the per-user global supervisor mutex '$supervisorMutexName': $($_.Exception.Message)"
    }

    if (-not $ownsMutex) {
        Write-SupervisorLog "Another Bubblebot supervisor already owns this Windows user; exiting."
        exit 0
    }

    Write-SupervisorLog "Supervisor started as $($identity.Name) for '$script:CanonicalRepoPath' on $Remote/$Branch."
    $restartAttempt = 0
    $supervisorExitCode = 0
    $env:BUBBLES_GATEWAY_SUPERVISED = "1"
    $env:GIT_TERMINAL_PROMPT = "0"
    $env:GCM_INTERACTIVE = "Never"
    $env:GIT_SSH_COMMAND = "ssh -o BatchMode=yes -o ConnectTimeout=15"
    $env:UV_HTTP_TIMEOUT = "60"

    $gatewayProcess = $null
    $gatewayStartedAt = $null
    $gatewayInstanceId = $null
    Push-Location $script:CanonicalRepoPath
    try {
        if ((Test-Path -LiteralPath $script:InProgressPath -PathType Leaf) -and
            -not (Recover-UnfinishedUpgrade)) {
            $supervisorExitCode = 78
        } else {
            try {
                # An abandoned Global mutex proves only that the old supervisor
                # process is gone; its gateway child may still be alive.  Refuse
                # dependency sync as well as process startup while that child
                # retains the persistent WCFerry lease.
                Clear-StaleGatewayStopHandshake
                Assert-RepositoryRoot
                Assert-ConfiguredBranch
                Invoke-StartupSync
            } catch {
                Write-SupervisorLog "Startup validation failed: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
                $supervisorExitCode = 77
            }
        }

        while ($supervisorExitCode -eq 0) {
            if ($null -eq $gatewayProcess) {
                Write-SupervisorLog "Starting gateway on port $Port."
                Clear-StaleGatewayStopHandshake
                $gatewayProcess = Start-GatewayProcess
                $gatewayStartedAt = $gatewayProcess.StartTime.ToUniversalTime()
                $gatewayInstanceId = $gatewayProcess.GatewayInstanceId
            }

            $startedAt = $gatewayStartedAt
            $exitedGatewayInstanceId = $gatewayInstanceId
            try {
                $gatewayProcess.WaitForExit()
                $gatewayExitCode = $gatewayProcess.ExitCode
            } finally {
                $gatewayProcess.Dispose()
                $gatewayProcess = $null
                $gatewayStartedAt = $null
                $gatewayInstanceId = $null
            }
            $runtimeSeconds = ((Get-Date).ToUniversalTime() - $startedAt.ToUniversalTime()).TotalSeconds
            Write-SupervisorLog ("Gateway exited with code {0} after {1:N1}s." -f $gatewayExitCode, $runtimeSeconds)

            if ($gatewayExitCode -eq 76) {
                try {
                    Write-GatewayExitTombstone `
                        -InstanceId $exitedGatewayInstanceId `
                        -Reason "no_restart_exit"
                } catch {
                    Write-SupervisorLog "Unable to persist no-restart tombstone: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
                }
                Write-SupervisorLog "Gateway requested a fail-closed supervisor stop; it will not be restarted." -Level "WARN"
                $supervisorExitCode = 76
                break
            }

            if (-not (Test-GatewayCleanupCommit -InstanceId $exitedGatewayInstanceId)) {
                try {
                    Write-GatewayExitTombstone `
                        -InstanceId $exitedGatewayInstanceId `
                        -Reason "unproven_gateway_exit"
                } catch {
                    Write-SupervisorLog "Unable to persist unproven-exit tombstone: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
                }
                Write-SupervisorLog "Gateway exited without matching WCFerry cleanup proof; automatic restart is blocked." -Level "ERROR"
                $supervisorExitCode = 76
                break
            }

            if ($gatewayExitCode -eq 0) {
                $supervisorExitCode = 0
                break
            }

            if ($gatewayExitCode -eq 75) {
                $upgradeOutcome = Invoke-Upgrade
                if ($upgradeOutcome.CanContinue) {
                    $restartAttempt = 0
                    if ($null -ne $upgradeOutcome.GatewayProcess) {
                        $gatewayProcess = $upgradeOutcome.GatewayProcess
                        $gatewayStartedAt = $upgradeOutcome.StartedAt
                        $gatewayInstanceId = $upgradeOutcome.GatewayInstanceId
                    }
                    continue
                }

                # The transaction could not be proven safe to restart. A persistent
                # marker blocks future logon starts until an operator repairs it.
                $supervisorExitCode = 76
                break
            }

            if ($runtimeSeconds -ge $StableRunSeconds) {
                $restartAttempt = 0
            }
            $restartAttempt += 1

            if ($restartAttempt -gt $MaxRestarts) {
                Write-SupervisorLog "Gateway exceeded the restart limit; supervisor is stopping." -Level "ERROR"
                $supervisorExitCode = $gatewayExitCode
                break
            }

            $delaySeconds = Get-RestartDelaySeconds -Attempt $restartAttempt
            Write-SupervisorLog "Restarting gateway in $delaySeconds second(s) (attempt $restartAttempt/$MaxRestarts)." -Level "WARN"
            Start-Sleep -Seconds $delaySeconds
        }
    } finally {
        if ($null -ne $gatewayProcess) {
            try {
                try {
                    $stoppedSafely = Stop-GatewayProcess `
                        -Process $gatewayProcess `
                        -InstanceId $gatewayInstanceId `
                        -ExpectedStartTimeUtc $gatewayStartedAt `
                        -ExpectedUvExe $script:UvExe `
                        -Reason "supervisor_shutdown"
                } catch {
                    $stoppedSafely = $false
                    Write-SupervisorLog "Unexpected graceful-stop failure: $(Protect-LogText $_.Exception.Message)" -Level "ERROR"
                }
                if (-not $stoppedSafely) {
                    $supervisorExitCode = 76
                }
            } finally {
                $gatewayProcess.Dispose()
                $gatewayProcess = $null
                $gatewayInstanceId = $null
            }
        }
        Pop-Location
    }
} catch {
    Write-SupervisorLog "Supervisor failed: $($_.Exception.Message)" -Level "ERROR"
    $supervisorExitCode = 1
} finally {
    if ($ownsMutex) {
        try {
            $mutex.ReleaseMutex()
        } catch {
            # The process is exiting; a failed release needs no recovery action.
        }
    }
    if ($null -ne $mutex) {
        $mutex.Dispose()
    }
}

exit $supervisorExitCode
