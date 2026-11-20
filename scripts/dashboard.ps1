# Control the always-on Streamlit dashboard.
#   powershell -ExecutionPolicy Bypass -File scripts\dashboard.ps1 <install|uninstall|start|stop|restart|status>
# install   : register the "FashionMNIST Dashboard" task (runs scripts\run_dashboard.ps1 at logon)
# start     : start the task now and wait until http://127.0.0.1:8501 is healthy
# stop      : stop the task, its supervisor and the project's Streamlit processes
# restart   : stop + start
# status    : task state, processes, health check and recent supervisor log
# uninstall : stop and remove the task
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("install", "uninstall", "start", "stop", "restart", "status")]
    [string]$Command
)

$ErrorActionPreference = "Stop"
$TaskName   = "FashionMNIST Dashboard"
$Root       = Split-Path -Parent $PSScriptRoot
$Supervisor = Join-Path $PSScriptRoot "run_dashboard.ps1"
$Port       = 8501
$Health     = "http://127.0.0.1:$Port/_stcore/health"
$SupLog     = Join-Path $Root "logs\dashboard_supervisor.log"

function Get-Task { Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue }

function Test-Health {
    try { (Invoke-WebRequest -Uri $Health -UseBasicParsing -TimeoutSec 5).Content.Trim() -eq "ok" }
    catch { $false }
}

function Get-SelfAndAncestors {
    # This script's own process chain must never be matched (its command line may mention "streamlit")
    $ids = @(); $cur = $PID
    while ($cur -and -not ($ids -contains $cur)) {
        $ids += $cur
        $p = Get-CimInstance Win32_Process -Filter "ProcessId=$cur" -ErrorAction SilentlyContinue
        if (-not $p) { break }
        $cur = $p.ParentProcessId
    }
    $ids
}

function Get-ProjectProcesses {
    # Supervisors of this project, and Streamlit processes launched from this project folder
    $exclude = Get-SelfAndAncestors
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("powershell.exe", "python.exe", "streamlit.exe") -and
        -not ($exclude -contains $_.ProcessId) -and
        $_.CommandLine -and $_.CommandLine -like "*$Root*" -and
        ($_.CommandLine -like "*run_dashboard.ps1*" -or $_.CommandLine -like "*streamlit*")
    }
}

function Install-Task {
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $Root `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Supervisor`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description "Keeps the Fashion-MNIST Streamlit dashboard running at http://127.0.0.1:$Port" `
        -Force | Out-Null
    Write-Output "Installed scheduled task '$TaskName' (runs at logon)."
}

function Stop-Dashboard {
    if (Get-Task) { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue }
    $procs = @(Get-ProjectProcesses)
    foreach ($p in $procs) {
        # /T kills the whole tree (supervisor -> venv python -> interpreter)
        & taskkill.exe /PID $p.ProcessId /T /F 2>$null | Out-Null
    }
    Start-Sleep -Seconds 2
    $left = @(Get-ProjectProcesses)
    Write-Output ("Stopped. Project processes ended: {0}; still running: {1}" -f $procs.Count, $left.Count)
}

function Start-Dashboard {
    if (-not (Get-Task)) { throw "Task '$TaskName' is not installed. Run: dashboard.ps1 install" }
    Start-ScheduledTask -TaskName $TaskName
    for ($i = 0; $i -lt 45; $i++) {
        if (Test-Health) { Write-Output "Dashboard is up: http://localhost:$Port"; return }
        Start-Sleep -Seconds 2
    }
    Write-Output "Task started, but the health check has not passed yet. See: dashboard.ps1 status"
}

function Show-Status {
    $t = Get-Task
    if ($t) {
        $info = Get-ScheduledTaskInfo -TaskName $TaskName
        Write-Output ("Task            : {0} (state: {1}, last run: {2}, last result: 0x{3:X})" -f `
            $TaskName, $t.State, $info.LastRunTime, $info.LastTaskResult)
    } else {
        Write-Output "Task            : not installed"
    }
    # Server sockets have no remote end (RemotePort 0); Windows may label a 127.0.0.1-bound one CLOSED, not LISTEN
    $listen = @(Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.RemotePort -eq 0 })
    if ($listen.Count) {
        Write-Output ("Port {0}       : bound on {1} (pid {2})" -f $Port, (($listen.LocalAddress | Sort-Object -Unique) -join ", "), $listen[0].OwningProcess)
    } else {
        Write-Output "Port $Port       : not bound"
    }
    Write-Output ("Health check    : {0}" -f $(if (Test-Health) { "OK ($Health)" } else { "FAILED ($Health)" }))
    foreach ($p in @(Get-ProjectProcesses)) {
        $kind = if ($p.CommandLine -like "*run_dashboard.ps1*") { "supervisor" } else { "streamlit " }
        Write-Output ("Process         : {0} pid {1} (started {2})" -f $kind, $p.ProcessId, $p.CreationDate)
    }
    if (Test-Path $SupLog) {
        Write-Output "Supervisor log  : $SupLog (last 5 lines)"
        Get-Content $SupLog -Tail 5 | ForEach-Object { Write-Output "    $_" }
    }
}

switch ($Command) {
    "install"   { Install-Task }
    "uninstall" { Stop-Dashboard; if (Get-Task) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }; Write-Output "Task removed." }
    "start"     { Start-Dashboard }
    "stop"      { Stop-Dashboard }
    "restart"   { Stop-Dashboard; Start-Dashboard }
    "status"    { Show-Status }
}
