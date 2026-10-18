# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
#
# Waits for finish_queue.ps1 to log ALL DONE, then reruns the Section 3.3.1 to 3.3.3 prompts
# with the chat-template pass added to playground_prompts_local.py. The GPU only fits one Qwen
# model at a time, so this can't run while the few-shot BoolQ runs are still going.
$root = Split-Path -Parent $PSScriptRoot
$status = Join-Path $root "logs\finish_queue_status.txt"
$py = Join-Path $env:USERPROFILE ".venvs\nlp705-hw6\Scripts\python.exe"
while (-not (Select-String -Path $status -Pattern "ALL DONE" -CaseSensitive -Quiet)) { Start-Sleep -Seconds 15 }
"$(Get-Date -Format s)  3.3.1-3.3.3 prompts again, with the chat-template pass" | Out-File -Append -Encoding utf8 $status
$hw6 = Join-Path $root "hw6"
$log = Join-Path $root "logs\playground_prompts_local.log"
& cmd.exe /c "cd /d `"$hw6`" && set PYTHONIOENCODING=utf-8&& `"$py`" -W ignore playground_prompts_local.py cuda > `"$log`" 2>&1"
"$(Get-Date -Format s)  after_queue done" | Out-File -Append -Encoding utf8 $status
