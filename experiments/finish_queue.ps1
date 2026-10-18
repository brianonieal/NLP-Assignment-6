# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
#
# Runs everything left that needs the GPU, after the first driver was stopped:
#   1. probe the largest bf16 batch for BERT-base and RoBERTa-base
#   2. start the driver again; it skips finished runs and runs the rest (most with --amp)
#   3. run the Section 3.3 local-model scripts, then draw the summary plots
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File experiments\finish_queue.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $env:USERPROFILE ".venvs\nlp705-hw6\Scripts\python.exe"
$logs = Join-Path $root "logs"
$status = Join-Path $logs "finish_queue_status.txt"
function Note($msg) { "$(Get-Date -Format s)  $msg" | Out-File -Append -Encoding utf8 $status; }
function RunPy($argline, $log, $cwd) {
    # cmd keeps Python's UTF-8 output as is (PowerShell redirection would re-encode it as UTF-16)
    $cmdline = "cd /d `"$cwd`" && set MPLBACKEND=Agg&& set PYTHONIOENCODING=utf-8&& set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1&& `"$py`" -W ignore $argline > `"$log`" 2>&1"
    & cmd.exe /c $cmdline
}

# keep the PC awake while this runs (released when the script exits)
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

foreach ($m in @("bert-base-uncased", "roberta-base")) {
    if (Test-Path (Join-Path $root "results\batch_probe_${m}_amp.json")) { continue }
    Note "bf16 batch probe $m"
    RunPy "experiments\probe_batch_size.py --model $m --start 48 --amp --out results\batch_probe_${m}_amp.json" (Join-Path $logs "probe_${m}_amp.log") $root
}

Note "driver 2 start"
RunPy "experiments\run_all.py" (Join-Path $logs "driver2.log") $root
Note "driver 2 done"

$hw6 = Join-Path $root "hw6"
Note "3.3.1-3.3.3 prompts"
RunPy "playground_prompts_local.py cuda" (Join-Path $logs "playground_prompts_local.log") $hw6
foreach ($m in @("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B")) {
    $tag = ($m -split "/")[-1]
    Note "3.3.4 $tag, 30 questions"
    RunPy "openai_gpt-boolq.py --model $m --n_eval 30 --out ..\results\boolq_fewshot_${tag}_n30.json" (Join-Path $logs "boolq_fewshot_${tag}_n30.log") $hw6
    Note "3.3.4 $tag, all 3,270 validation questions"
    RunPy "openai_gpt-boolq.py --model $m --n_eval 0 --out ..\results\boolq_fewshot_${tag}_all.json" (Join-Path $logs "boolq_fewshot_${tag}_all.log") $hw6
}

Note "plots"
RunPy "experiments\make_plots.py" (Join-Path $logs "make_plots.log") $root
Note "ALL DONE"
