# Supervisor for the Streamlit dashboard. Started at logon by the "FashionMNIST Dashboard" scheduled task
# (see scripts/dashboard.ps1). Keeps Streamlit running, restarts it if it exits, and logs to logs/.
param(
    [int]$Port = 8501,
    [string]$Address = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
$Temp   = Join-Path $Root "temp"

foreach ($d in @($LogDir, $Temp, (Join-Path $Temp "joblib"), (Join-Path $Temp "matplotlib"))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
# Keep every temp/cache file on D: (inherited by Streamlit and its workers)
$env:TMP = $Temp; $env:TEMP = $Temp; $env:TMPDIR = $Temp
$env:JOBLIB_TEMP_FOLDER = Join-Path $Temp "joblib"
$env:MPLCONFIGDIR = Join-Path $Temp "matplotlib"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"     # write Streamlit output to the log files immediately

function Test-PortInUse([int]$P) {
    # Connect test: Windows can list a 127.0.0.1-bound Streamlit socket as CLOSED instead of LISTEN,
    # so asking for "Listen" sockets is not reliable. A successful connection is.
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $ar = $client.BeginConnect("127.0.0.1", $P, $null, $null)
        return ($ar.AsyncWaitHandle.WaitOne(1000) -and $client.Connected)
    } catch { return $false } finally { $client.Close() }
}

$SupervisorLog = Join-Path $LogDir "dashboard_supervisor.log"
function Write-Log([string]$Message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [supervisor pid $PID] $Message" |
        Add-Content -Path $SupervisorLog -Encoding utf8
}

# One supervisor per port: a second copy exits immediately (no duplicate Streamlit processes)
$mutex = New-Object System.Threading.Mutex($false, "Local\FashionMnistDashboard_$Port")
try { $owned = $mutex.WaitOne(0) }
catch [System.Threading.AbandonedMutexException] { $owned = $true }   # previous supervisor was killed
if (-not $owned) {
    Write-Log "Another supervisor already manages port $Port; exiting."
    exit 0
}

if (-not (Test-Path $Python)) { Write-Log "Python not found at $Python; exiting."; exit 1 }
Write-Log "Supervisor started (root=$Root, python=$Python, url=http://${Address}:$Port)"

$delay = 5
try {
    while ($true) {
        if (Test-PortInUse $Port) {
            $owner = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
                     Where-Object { $_.RemotePort -eq 0 } | Select-Object -First 1 -ExpandProperty OwningProcess
            Write-Log "Port $Port is already in use (pid $owner); checking again in 30 s."
            Start-Sleep -Seconds 30
            continue
        }

        $stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
        $outLog = Join-Path $LogDir "streamlit_$stamp.log"
        $errLog = Join-Path $LogDir "streamlit_$stamp.err.log"
        $stArgs = @("-m", "streamlit", "run", "app.py",
                    "--server.address", $Address, "--server.port", "$Port",
                    "--server.headless", "true", "--browser.gatherUsageStats", "false")
        Write-Log "Starting Streamlit (logs: $(Split-Path $outLog -Leaf), $(Split-Path $errLog -Leaf))"
        $started = Get-Date
        $proc = Start-Process -FilePath $Python -ArgumentList $stArgs -WorkingDirectory $Root `
                    -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
                    -WindowStyle Hidden -PassThru
        $null = $proc.Handle          # cache the handle so ExitCode is available after exit
        $proc.WaitForExit()
        $ranFor = [int]((Get-Date) - $started).TotalSeconds
        Write-Log "Streamlit exited with code $($proc.ExitCode) after $ranFor s."

        # Back off when it keeps crashing quickly; reset after a healthy run
        if ($ranFor -ge 120) { $delay = 5 } else { $delay = [Math]::Min($delay * 2, 60) }
        Write-Log "Restarting in $delay s."
        Start-Sleep -Seconds $delay
    }
}
finally {
    Write-Log "Supervisor stopping."
    $mutex.ReleaseMutex()
}
