# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
#
# The BERT-base and RoBERTa-base grids for 3.1.4 moved to a Colab L4 GPU, which ran them about
# 4x faster than this laptop. This waits for the laptop's last training job (the distilBERT bf16
# control), then stops the driver before it starts the BERT-base grid. finish_queue.ps1 then
# goes on to the Section 3.3 runs and the plots.
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root "logs\3.1.4_check_distilbert_amp_lr0.0001_ep9.log"
$status = Join-Path $root "logs\finish_queue_status.txt"
while (-not ((Test-Path $log) -and (Select-String -Path $log -Pattern "Average TEST metrics" -CaseSensitive -Quiet))) {
    Start-Sleep -Seconds 10
}
Start-Sleep -Seconds 20   # let the driver save results.json
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match "run_all\.py" -or $_.CommandLine -match "base_classification\.py")
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    "$(Get-Date -Format s)  stop_after_check: stopped $($_.ProcessId)" | Out-File -Append -Encoding utf8 $status
}
