# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script for the Section
# 3.2.4 extra credit. I directed the work and reviewed it.
"""GPU memory of the forward and backward pass for full fine-tuning, head-tuning, and prefix-tuning.

Uses CustomModelforSequenceClassification from hw6/classification.py on RoBERTa-base, a real
batch of 32 BoolQ training examples at max_len 128 (prefix length 128), and the same freezing and
optimizers as train() in classification.py. One warm-up step runs first so AdamW's state
already exists and the measured step looks like every later step in training.

The measured step runs inside torch.profiler.profile(profile_memory=True), with the forward pass
(model call plus loss) in record_function("forward") and loss.backward() in
record_function("backward"), as in the PyTorch profiler recipe. Two numbers per region:
  * profiler_net_mib: the profiler's CUDA memory for that region, i.e. memory allocated minus
    memory freed inside it. Forward is positive (activations saved for backward). Backward can be
    negative, because it frees those activations while it allocates gradients.
  * peak_above_start_mib: torch.cuda.max_memory_allocated() during the region, minus the memory
    allocated when the region started. This is how much extra memory the region needs at its worst.

Usage: python experiments/profile_memory.py   (writes results/memory_profile.json and plots/)
"""
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hw6"))
import classification  # noqa: E402

MODEL, BS, MAX_LEN, PREFIX = "roberta-base", 32, 128, 128
MIB = 2 ** 20


def get_batch():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("boolq")["train"].shuffle(seed=42).select(range(BS))
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    data = classification.BoolQADataset(list(ds["passage"]), list(ds["question"]), list(ds["answer"]), tok, MAX_LEN)
    items = [data[i] for i in range(BS)]
    return {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}


def build(typ):
    torch.manual_seed(0)
    model = classification.CustomModelforSequenceClassification(MODEL, num_labels=2, type=typ).cuda()
    if typ == "full":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    else:
        for p in model.model.parameters():
            p.requires_grad = False
        params = list(model.classifier.parameters()) + ([model.prefix] if typ == "prefix" else [])
        opt = torch.optim.AdamW(params, lr=1e-3)
    model.train()
    return model, opt


def step(model, opt, batch, loss_fn, regions=None):
    """One training step; if regions is a dict, record per-region peak memory into it."""
    def mark():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()

    start = mark()
    with record_function("forward"):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(out["logits"], batch["labels"])
    torch.cuda.synchronize()
    if regions is not None:
        regions["forward_peak"] = (torch.cuda.max_memory_allocated() - start) / MIB
    start = mark()
    with record_function("backward"):
        loss.backward()
    torch.cuda.synchronize()
    if regions is not None:
        regions["backward_peak"] = (torch.cuda.max_memory_allocated() - start) / MIB
    opt.step()
    opt.zero_grad(set_to_none=True)


def main():
    classification.prefix_length = PREFIX
    batch = get_batch()
    loss_fn = torch.nn.CrossEntropyLoss()
    results = {}
    for typ in ("full", "head", "prefix"):
        model, opt = build(typ)
        step(model, opt, batch, loss_fn)  # warm-up: allocates AdamW state
        static = torch.cuda.memory_allocated() / MIB
        regions = {}
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            step(model, opt, batch, loss_fn, regions)
        avg = {e.key: e for e in prof.key_averages()}
        results[typ] = {
            "forward": {"profiler_net_mib": avg["forward"].cuda_memory_usage / MIB,
                        "peak_above_start_mib": regions["forward_peak"]},
            "backward": {"profiler_net_mib": avg["backward"].cuda_memory_usage / MIB,
                         "peak_above_start_mib": regions["backward_peak"]},
            "static_mib_weights_grads_optimizer": static,
            "trainable_parameters": sum(p.numel() for g in opt.param_groups for p in g["params"]),
        }
        print(f"{typ:6s} forward net {results[typ]['forward']['profiler_net_mib']:8.1f} MiB, "
              f"peak +{regions['forward_peak']:8.1f} MiB | backward net "
              f"{results[typ]['backward']['profiler_net_mib']:8.1f} MiB, peak +{regions['backward_peak']:8.1f} MiB | "
              f"static {static:7.1f} MiB", flush=True)
        if typ == "prefix":
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=12))
        del model, opt, prof
        torch.cuda.empty_cache()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "memory_profile.json"), "w") as f:
        json.dump({"model": MODEL, "batch_size": BS, "max_len": MAX_LEN, "prefix_length": PREFIX,
                   "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": results}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    methods = ["full", "head", "prefix"]
    labels = ["Full fine-tuning", "Head-tuning", "Prefix-tuning"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "profiler_net_mib", "Net CUDA memory per region (torch.profiler)"),
                           (axes[1], "peak_above_start_mib", "Peak extra memory during region")):
        fw = [results[m]["forward"][key] for m in methods]
        bw = [results[m]["backward"][key] for m in methods]
        b1 = ax.bar(x - 0.18, fw, 0.36, label="forward", color="#4C72B0")
        b2 = ax.bar(x + 0.18, bw, 0.36, label="backward", color="#DD8452")
        ax.bar_label(b1, fmt="%.0f", fontsize=8)
        ax.bar_label(b2, fmt="%.0f", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("MiB")
        ax.set_title(title, fontsize=10)
        ax.legend()
    fig.suptitle(f"RoBERTa-base, batch {BS}, max_len {MAX_LEN}, prefix {PREFIX}, one training step on an RTX 4050 Laptop GPU", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "plots", "3.2.4_memory_profile.png"), dpi=150)
    print("saved plots/3.2.4_memory_profile.png")


if __name__ == "__main__":
    main()
