# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this driver. I directed the
# work and reviewed it.
"""Runs every Section 3.1 and 3.2 training job in hw6/ as a subprocess, one after another.

Each job's console output goes to logs/<name>.log (tqdm progress bars to logs/<name>.err). The
per-epoch and final accuracies are parsed out of the log into results/results.json, and the
plot the job saves is moved to plots/<name>.png. A job whose log already ends with a TEST
accuracy is skipped, so the driver can be stopped and restarted.

While it runs, it asks Windows not to sleep (SetThreadExecutionState); that lapses when the
driver exits. Closing the laptop lid can still force sleep.

Run from the repo root with the hw6 venv:   python experiments/run_all.py [--only NAME ...]
"""
import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW6 = os.path.join(ROOT, "hw6")
LOGS = os.path.join(ROOT, "logs")
PLOTS = os.path.join(ROOT, "plots")
RESULTS = os.path.join(ROOT, "results", "results.json")

D, BB, RB = "distilbert-base-uncased", "bert-base-uncased", "roberta-base"


def base(name, model, bs, lr, ep, small=False, group="", amp=False):
    args = ["--device", "cuda", "--model", model, "--batch_size", str(bs), "--lr", str(lr),
            "--num_epochs", str(ep)] + (["--amp"] if amp else [])
    if small:
        args = ["--small_subset"] + args
    plot = "overfit.png" if small else "base_full.png"
    return dict(name=name, script="base_classification.py", args=args, plot=plot, group=group,
                model=model, batch_size=bs, lr=lr, epochs=ep, amp=amp)


def custom(name, model, typ, bs, lr, ep, group="", amp=False):
    args = ["--device", "cuda", "--model", model, "--type", typ, "--batch_size", str(bs),
            "--lr", str(lr), "--num_epochs", str(ep)] + (["--amp"] if amp else [])
    return dict(name=name, script="classification.py", args=args, plot=f"{model}_{typ}.png",
                group=group, model=model, batch_size=bs, lr=lr, epochs=ep, type=typ, amp=amp)


def lora(name, model, bs, lr, ep, r=8, alpha=16, group="", amp=False):
    args = ["--device", "cuda", "--model", model, "--batch_size", str(bs), "--lr", str(lr),
            "--num_epochs", str(ep), "--lora_r", str(r), "--lora_alpha", str(alpha)] + (["--amp"] if amp else [])
    return dict(name=name, script="classification_lora.py", args=args, plot="lora_full.png",
                group=group, model=model, batch_size=bs, lr=lr, epochs=ep, lora_r=r, lora_alpha=alpha, amp=amp)


NO_AMP = False
GRID_LRS = [1e-4, 5e-4, 1e-3]
GRID_EPOCHS = [7, 9]

JOBS = [
    # 3.1.1: the command given in the assignment
    base("3.1.1_overfit_distilbert", D, 64, 1e-4, 20, small=True, group="3.1.1"),
    # 3.1.2: all 8,000 training examples, 30 epochs, same batch size and learning rate
    base("3.1.2_full30_distilbert", D, 64, 1e-4, 30, group="3.1.2"),
]
# 3.1.3: learning rate x epochs grid for distilBERT
JOBS += [base(f"3.1.3_grid_distilbert_lr{lr:g}_ep{ep}", D, 64, lr, ep, group="3.1.3")
         for lr in GRID_LRS for ep in GRID_EPOCHS]
# 3.2: RoBERTa-base with the custom mean-pooled head
JOBS += [
    custom("3.2.1_roberta_full", RB, "full", 32, 2e-5, 6, group="3.2.1"),
    custom("3.2.2_roberta_head", RB, "head", 32, 1e-3, 10, group="3.2.2"),
    # batch 16 with bf16 (--amp): in float32 at batch 32 the 256-token (prefix + input) activations
    # alone take 5,360 MiB (results/memory_profile.json), more than the card holds with the frozen
    # weights, and bf16 at batch 32 still spilled into shared system memory.
    # logs/bench_precision.log has the timings behind this choice.
    custom("3.2.3_roberta_prefix", RB, "prefix", 16, 5e-3, 10, group="3.2.3", amp=True),
    # from here on: bf16 mixed precision (--amp), to finish sooner on the 6 GB card
    lora("3.2.6_roberta_lora", RB, 32, 5e-4, 6, group="3.2.6", amp=True),
    # control: the selected 3.1.3 config again under --amp, to show what bf16 alone changes
    base("3.1.4_check_distilbert_amp_lr0.0001_ep9", D, 64, 1e-4, 9, group="3.1.4", amp=True),
]
def grid_batch_size(model):
    """Batch 64 (as for distilBERT) if the bf16 probe found room for at least 72, else 32."""
    path = os.path.join(ROOT, "results", f"batch_probe_{model}_amp.json")
    try:
        with open(path) as f:
            return 64 if json.load(f)["largest_batch"] >= 72 else 32
    except (OSError, KeyError, ValueError):
        return 32


# 3.1.4: the same grid for two more models
JOBS += [base(f"3.1.4_grid_{tag}_lr{lr:g}_ep{ep}", m, grid_batch_size(m), lr, ep, group="3.1.4", amp=True)
         for tag, m in (("bert-base-uncased", BB), ("roberta-base", RB))
         for lr in GRID_LRS for ep in GRID_EPOCHS]

PATTERNS = {
    "train_acc": r"Average training metrics: accuracy=\{'accuracy': ([0-9.]+)\}",
    "dev_acc": r"Average validation metrics: accuracy=\{'accuracy': ([0-9.]+)\}",
    "epoch_test_acc": r"Average test metrics: accuracy=\{'accuracy': ([0-9.]+)\}",
    "epoch_seconds": r"Epoch \d+ took ([0-9.]+) seconds",
}
FINAL = {
    "final_dev_acc": r"Average DEV metrics: accuracy=\{'accuracy': ([0-9.]+)\}",
    "final_test_acc": r"Average TEST metrics: accuracy=\{'accuracy': ([0-9.]+)\}",
    "max_memory_reserved_gb": r"torch\.cuda\.max_memory_reserved: ([0-9.]+)GB",
    "tuned_parameters": r"Tuned parameters \(\w+\): (\d+)",
    "lora_trainable_total": r"LoRA trainable parameters: (\d+) total",
    "lora_trainable_in_AB": r"LoRA trainable parameters: \d+ total, (\d+) in the LoRA",
}


def parse_log(text):
    out = {k: [float(x) for x in re.findall(p, text)] for k, p in PATTERNS.items()}
    for k, p in FINAL.items():
        m = re.findall(p, text)
        if m:
            out[k] = float(m[-1]) if "." in m[-1] else int(m[-1])
    return out


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(res):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


def adopt_running_job(job, res):
    """Wait for a job started by an earlier driver process to finish, then record it."""
    log_path = os.path.join(LOGS, job["name"] + ".log")
    print(f"wait {job['name']} (started by an earlier driver)", flush=True)
    while True:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "Average TEST metrics" in text:
            break
        time.sleep(30)
    time.sleep(10)  # let the job finish writing and exit
    produced = os.path.join(HW6, job["plot"])
    if os.path.exists(produced):
        shutil.move(produced, os.path.join(PLOTS, job["name"] + ".png"))
    res[job["name"]] = {**{k: v for k, v in job.items() if k != "args"},
                        "command": "python " + " ".join([job["script"]] + job["args"]),
                        "returncode": 0, **parse_log(text)}
    save_results(res)


def run_job(job, res):
    log_path = os.path.join(LOGS, job["name"] + ".log")
    err_path = os.path.join(LOGS, job["name"] + ".err")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "Average TEST metrics" in text:
            print(f"skip {job['name']} (already done)", flush=True)
            res[job["name"]] = {**res.get(job["name"], {}),
                                **{k: v for k, v in job.items() if k != "args"}, **parse_log(text)}
            return True
    args_ = [a for a in job["args"] if not (NO_AMP and a == "--amp")]
    cmd = [sys.executable, job["script"]] + args_
    # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 lets cuBLAS use TF32 tensor cores for float32 matrix
    # multiplies. Every job from 3.1.2 on ran with it. The training code itself is unchanged.
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="1")
    print(f"run  {job['name']}: {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
        out.write(f"$ python {' '.join(cmd[1:])}\n")
        out.flush()
        rc = subprocess.call(cmd, cwd=HW6, stdout=out, stderr=err, env=env)
    wall = time.time() - t0
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    parsed = parse_log(text)
    produced = os.path.join(HW6, job["plot"])
    if os.path.exists(produced):
        shutil.move(produced, os.path.join(PLOTS, job["name"] + ".png"))
    res[job["name"]] = {**{k: v for k, v in job.items() if k != "args"},
                        "command": "python " + " ".join(cmd[1:]), "returncode": rc,
                        "wall_seconds": round(wall, 1), **parsed}
    save_results(res)
    status = "ok" if rc == 0 else f"FAILED rc={rc}"
    print(f"     {status} in {wall / 60:.1f} min; final dev={parsed.get('final_dev_acc')} "
          f"test={parsed.get('final_test_acc')}", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only jobs whose name starts with one of these")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no_amp", action="store_true", help="drop --amp from every job (fallback)")
    ap.add_argument("--wait_for", default=None,
                    help="name of a job an earlier driver started; wait for it instead of rerunning it")
    args = ap.parse_args()
    global NO_AMP
    NO_AMP = args.no_amp
    if args.list:
        for j in JOBS:
            print(j["name"])
        return
    for d in (LOGS, PLOTS):
        os.makedirs(d, exist_ok=True)
    if os.name == "nt":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no system sleep while this runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    res = load_results()
    jobs = [j for j in JOBS if not args.only or any(j["name"].startswith(p) for p in args.only)]
    print(f"{len(jobs)} jobs", flush=True)
    for job in jobs:
        if job["name"] == args.wait_for:
            adopt_running_job(job, res)
            continue
        run_job(job, res)
        save_results(res)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
