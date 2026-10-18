# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Builds the summary figures and tables from results/*.json.

  plots/3.1.4_model_comparison.png   best dev/test accuracy per model (Section 3.1.4 bar plot)
  plots/3.1.3_grid.png               dev accuracy per epoch for the six distilBERT grid runs
  plots/3.1.4_grid_<model>.png       the same for the BERT-base and RoBERTa-base grids
  plots/3.2.4_memory_profile.png     forward/backward GPU memory per tuning method
  results/summary.md                 every run's final dev and test accuracy

The plots that the homework scripts draw themselves (overfit.png, base_full.png, and
<model>_<type>.png) are kept as generated, under plots/<run name>.png.

Usage: python experiments/make_plots.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

# validated two-slot categorical palette (dataviz reference palette, light mode)
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def best_by_dev(results, prefix):
    runs = {k: v for k, v in results.items() if k.startswith(prefix) and "final_dev_acc" in v}
    if not runs:
        return None, None
    name = max(runs, key=lambda k: runs[k]["final_dev_acc"])
    return name, runs[name]


LAPTOP = "laptop RTX 4050"


def fmt_lr(lr):
    """1e-4 style, as the assignment writes learning rates."""
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"


def where(r):
    """Short hardware and precision label for a run."""
    hw = "Colab L4" if "Colab" in r.get("hardware", "") else LAPTOP
    return f"{hw}, {'bf16' if r.get('amp') else 'fp32'}"


def model_comparison(results):
    """Bar plot for 3.1.4. Solid bars: best of each model's six grid runs, picked on dev.
    Hatched bars: a run outside the grid, drawn only if it exists in results.json."""
    rows = []
    for label, prefix, extra in (
        ("distilBERT-base-uncased", "3.1.3_grid_distilbert", False),
        ("BERT-base-uncased", "3.1.4_grid_bert-base-uncased", False),
        ("RoBERTa-base", "3.1.4_grid_roberta-base", False),
        ("RoBERTa-base, outside\nthe grid", "3.1.4_extra_roberta-base", True),
        ("BERT-large-uncased", None, False),
    ):
        if prefix is None:
            rows.append((label, 0.0, 0.0, "out of memory at batch 1\non my 6 GB GPU (reported 0)", False))
            continue
        name, r = best_by_dev(results, prefix)
        if r is None:
            continue
        grid = [v for k, v in results.items() if k.startswith(prefix) and "final_dev_acc" in v]
        if len(grid) > 1 and len({round(v["final_dev_acc"], 6) for v in grid}) == 1:
            cfg = f"all {len(grid)} runs predict 'yes'\nbatch {r['batch_size']}, {where(r)}"
        else:
            cfg = f"lr {fmt_lr(r['lr'])}, {r['epochs']} epochs\nbatch {r['batch_size']}, {where(r)}"
        rows.append((label, r["final_dev_acc"], r["final_test_acc"], cfg, extra))

    x = np.arange(len(rows))
    w = 0.34
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    for i, (label, dev, test, cfg, extra) in enumerate(rows):
        for off, h, c, lab in ((-w / 2 - 0.01, dev, C1, "dev (BoolQ validation, 3,270)"),
                               (w / 2 + 0.01, test, C2, "test (train[8000:], 1,427)")):
            style = dict(color="white", edgecolor=c, hatch="////", linewidth=1.2) if extra else dict(color=c)
            ax.bar(i + off, h, w, label=lab if i == 0 else None, **style)
            ax.text(i + off, h + 0.008, f"{h:.3f}" if h > 0 else "0", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
    majority = results.get("_majority_dev")
    if majority:
        ax.axhline(majority, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(len(rows) - 0.5, majority + 0.008, f"always 'yes' on dev = {majority:.3f}",
                ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{r[0]}\n{r[3]}" for r in rows], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Full fine-tuning on BoolQ: best of the 6 learning rate x epoch runs per model, picked on dev\n"
                 "(hatched: a run outside the grid, lr 2e-5)", fontsize=10, color=INK)
    ax.legend(loc="upper right", fontsize=8.5, ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "3.1.4_model_comparison.png"), dpi=170)
    plt.close(fig)


LR_COLORS = {1e-4: C1, 5e-4: C2, 1e-3: "#7a5195"}


def grid_curves(results, prefix, title, out):
    """Dev accuracy after each epoch for one model's grid. Runs whose dev curves are identical
    (the collapsed always-'yes' runs) are drawn once, as one gray line."""
    runs = sorted(k for k in results if k.startswith(prefix) and results[k].get("dev_acc"))
    if not runs:
        return
    def same(f, g):
        n = min(len(f), len(g))
        return all(abs(a - b) < 1e-9 for a, b in zip(f[:n], g[:n]))
    groups = []
    for k in runs:
        for g in groups:
            if same(results[g[0]]["dev_acc"], results[k]["dev_acc"]):
                g.append(k)
                break
        else:
            groups.append([k])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for g in groups:
        if len(g) > 1:
            longest = max((results[k]["dev_acc"] for k in g), key=len)
            lrs = ", ".join(fmt_lr(v) for v in sorted({results[k]["lr"] for k in g}))
            eps = " and ".join(str(v) for v in sorted({results[k]["epochs"] for k in g}))
            ax.plot(range(1, len(longest) + 1), longest, color=INK2, linewidth=2, marker="o", markersize=4,
                    label=f"lr {lrs}; {eps} epochs ({len(g)} runs, identical)")
            ax.text(1, longest[0] + 0.004, "predicts 'yes' for every question", fontsize=8, color=INK2)
            continue
        r = results[g[0]]
        dashes = (0, ()) if r["epochs"] == 9 else (0, (4, 2))
        ax.plot(range(1, len(r["dev_acc"]) + 1), r["dev_acc"], color=LR_COLORS.get(r["lr"], INK),
                linestyle=dashes, linewidth=2, marker="o", markersize=4,
                label=f"lr {fmt_lr(r['lr'])}, {r['epochs']} epochs")
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, out), dpi=170)
    plt.close(fig)


def memory_plot():
    path = os.path.join(RES, "memory_profile.json")
    if not os.path.exists(path):
        return
    mem = load("memory_profile.json")
    res = mem["results"]
    methods = ["full", "head", "prefix"]
    labels = ["Full fine-tuning", "Head-tuning", "Prefix-tuning\n(+128 prefix tokens)"]
    x = np.arange(3)
    w = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, title in ((axes[0], "profiler_net_mib",
                            "Net CUDA memory recorded in each region\n(torch.profiler: allocated minus freed)"),
                           (axes[1], "peak_above_start_mib",
                            "Peak extra memory while the region runs\n(max allocated minus allocated at region start)")):
        fw = [res[m]["forward"][key] for m in methods]
        bw = [res[m]["backward"][key] for m in methods]
        b1 = ax.bar(x - w / 2 - 0.01, fw, w, color=C1, label="forward")
        b2 = ax.bar(x + w / 2 + 0.01, bw, w, color=C2, label="backward")
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + (60 if h >= 0 else -60),
                        f"{h:,.0f}" if abs(h) >= 1 else f"{h:.1f}",
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=8, color=INK)
        ax.axhline(0, color=INK2, linewidth=0.9)
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.set_ylabel("MiB")
        ax.set_title(title, fontsize=9.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 300 if lo < 0 else 0, hi + 400)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle(f"RoBERTa-base, one training step, batch {mem['batch_size']}, max_len {mem['max_len']} "
                 f"({mem['gpu']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "3.2.4_memory_profile.png"), dpi=170)
    plt.close(fig)


def summary(results):
    lines = ["| run | model | where | batch | lr | epochs | final dev | final test | tuned params | minutes |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in results.items():
        if k.startswith("_") or "final_dev_acc" not in r:
            continue
        tuned = r.get("tuned_parameters") or r.get("lora_trainable_total") or ""
        minutes = f"{r['wall_seconds'] / 60:.1f}" if r.get("wall_seconds") else ""
        lines.append(f"| {k} | {r['model']} | {where(r)} | {r['batch_size']} | {fmt_lr(r['lr'])} | {r['epochs']} | "
                     f"{r['final_dev_acc']:.4f} | {r['final_test_acc']:.4f} | {tuned} | {minutes} |")
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    results = load("results.json") if os.path.exists(os.path.join(RES, "results.json")) else {}
    try:  # majority-class baseline on dev, for the reference line
        from datasets import load_dataset
        dev = load_dataset("boolq")["validation"]
        results["_majority_dev"] = sum(dev["answer"]) / len(dev)
    except Exception as err:
        print("no majority line:", err)
    model_comparison(results)
    grid_curves(results, "3.1.3_grid_distilbert", "distilBERT-base-uncased grid: dev accuracy after each epoch",
                "3.1.3_grid.png")
    for tag, label in (("bert-base-uncased", "BERT-base-uncased"), ("roberta-base", "RoBERTa-base")):
        grid_curves(results, f"3.1.4_grid_{tag}", f"{label} grid (Colab L4, bf16): dev accuracy after each epoch",
                    f"3.1.4_grid_{tag}.png")
    memory_plot()
    summary(results)
    print("done")


if __name__ == "__main__":
    main()
