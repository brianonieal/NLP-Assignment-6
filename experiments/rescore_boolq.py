# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Re-reads the saved few-shot BoolQ completions with the current parse() in hw6/openai_gpt-boolq.py.

The first version of parse() used startswith("no"), so the one Instruct completion that began
" north america" counted as a "no". Its gold answer was yes, so it was wrong either way and no
accuracy changes, but the predicted-no and unparseable counts in the JSON were off by one. The
completions are saved, so this fixes the counts without running the models again. The console
logs in logs/ still show the original counts.

Usage: python experiments/rescore_boolq.py
"""
import glob
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("fewshot", os.path.join(ROOT, "hw6", "openai_gpt-boolq.py"))
fewshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fewshot)

for path in sorted(glob.glob(os.path.join(ROOT, "results", "boolq_fewshot_*.json"))):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    changed = 0
    for r in d["records"]:
        p = fewshot.parse(r["completion"])
        pred = None if p is None else fewshot.label(p)
        if pred != r["prediction"]:
            changed += 1
        r["prediction"], r["correct"] = pred, pred == r["gold"]
    n = len(d["records"])
    d["correct"] = sum(r["correct"] for r in d["records"])
    d["accuracy"] = d["correct"] / n
    d["predicted_yes"] = sum(r["prediction"] == "yes" for r in d["records"])
    d["unparseable"] = sum(r["prediction"] is None for r in d["records"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"{os.path.basename(path)}: {changed} predictions changed; accuracy {d['correct']}/{n} = "
          f"{d['accuracy']:.4f}; predicted yes {d['predicted_yes']}, no "
          f"{n - d['predicted_yes'] - d['unparseable']}, unparseable {d['unparseable']}")
