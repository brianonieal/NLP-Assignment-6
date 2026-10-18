# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Adds the Section 3.1.4 runs from Google Colab to logs/, plots/ and results/results.json.

The BERT-base and RoBERTa-base grids ran on a Colab L4 GPU with experiments/colab_roberta_grid.ipynb,
which saved each run's console log and plot to Google Drive under the same names the laptop
driver (run_all.py) uses. Download that Drive folder, unzip it into colab/, then run

    python experiments/import_colab.py [colab]

Each <name>.log is copied to logs/ and each <name>.png to plots/. The log is parsed with the same
patterns as run_all.py, and the run's settings come from the "$ python ..." line at its top.
Every imported entry is tagged with the hardware line the notebook wrote as the log's first line.

If the Drive folder isn't available, results/colab_runs.json holds the same numbers (settings,
per-epoch train and dev accuracy, final dev and test accuracy, peak memory) copied from those
logs, and this merges it instead:

    python experiments/import_colab.py results/colab_runs.json
"""
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from run_all import LOGS, PLOTS, RESULTS, load_results, parse_log, save_results  # noqa: E402


def settings(text):
    """Model, batch size, learning rate, epochs and --amp from the '$ python ...' line."""
    cmd = next((line for line in text.splitlines() if line.startswith("$ python")), "")
    def arg(flag, cast):
        m = re.search(rf"{flag} (\S+)", cmd)
        return cast(m.group(1)) if m else None
    return dict(script=cmd.split()[2] if len(cmd.split()) > 2 else None, model=arg("--model", str),
                batch_size=arg("--batch_size", int), lr=arg("--lr", float), epochs=arg("--num_epochs", int),
                amp="--amp" in cmd.split(), command=cmd[2:])


def merge_json(path, res):
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)
    hardware = runs.pop("_hardware", "Google Colab")
    runs.pop("_source", None)
    for name, r in runs.items():
        res[name] = {"name": name, "group": "3.1.4", "script": "base_classification.py",
                     "hardware": hardware, "returncode": 0, **r}
        print(f"ok   {name}: dev={r.get('final_dev_acc')} test={r.get('final_test_acc')}")
    return len(runs)


def main():
    src = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "colab"))
    res = load_results()
    if src.endswith(".json"):
        n = merge_json(src, res)
        save_results(res)
        print(f"merged {n} runs into {os.path.relpath(RESULTS, ROOT)}")
        return
    n = 0
    for root, _, files in os.walk(src):
        for fn in sorted(files):
            if not fn.endswith(".log"):
                continue
            name = fn[:-4]
            with open(os.path.join(root, fn), encoding="utf-8", errors="replace") as f:
                text = f.read()
            if "Average TEST metrics" not in text:
                print(f"skip {name} (no TEST line, unfinished)")
                continue
            shutil.copyfile(os.path.join(root, fn), os.path.join(LOGS, fn))
            png = os.path.join(root, name + ".png")
            if os.path.exists(png):
                shutil.copyfile(png, os.path.join(PLOTS, name + ".png"))
            first = text.splitlines()[0]
            hardware = first.lstrip("# ").strip() if first.startswith("#") else "Google Colab"
            group = "3.1.4" if name.startswith("3.1.4") else ""
            res[name] = {"name": name, "group": group, **settings(text), "hardware": hardware,
                         "returncode": 0, **parse_log(text)}
            n += 1
            print(f"ok   {name}: dev={res[name].get('final_dev_acc')} test={res[name].get('final_test_acc')}")
    save_results(res)
    print(f"imported {n} runs into {os.path.relpath(RESULTS, ROOT)}")


if __name__ == "__main__":
    main()
