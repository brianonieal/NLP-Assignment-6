# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Times training steps in float32 (TF32 matmuls) against bf16 autocast on this GPU.

Each config runs 5 warm-up steps and then times 20 steps of the same step the training scripts
take (forward, cross-entropy, backward, AdamW step), on random token ids at max_len 128.
Run with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, as the training runs were.
Usage: python experiments/bench_precision.py
"""
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hw6"))
import classification  # noqa: E402

CONFIGS = [  # (label, model, type, batch)
    ("bert-base full, batch 32", "bert-base-uncased", "auto", 32),
    ("roberta-base prefix, batch 16", "roberta-base", "prefix", 16),
    ("roberta-base prefix, batch 32", "roberta-base", "prefix", 32),  # float32 spills, very slow
]


def build(model_name, typ):
    if typ == "auto":
        m = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
        return m, torch.optim.AdamW(m.parameters(), lr=1e-5), lambda o: o.logits
    classification.prefix_length = 128
    m = classification.CustomModelforSequenceClassification(model_name, 2, typ).cuda()
    for p in m.model.parameters():
        p.requires_grad = False
    return m, torch.optim.AdamW([m.prefix] + list(m.classifier.parameters()), lr=1e-3), lambda o: o["logits"]


def bench(model_name, typ, bs, amp):
    torch.manual_seed(0)
    model, opt, get_logits = build(model_name, typ)
    model.train()
    vocab = model.config.vocab_size if typ == "auto" else model.model.config.vocab_size
    ids = torch.randint(1000, vocab, (bs, 128), device="cuda")
    mask = torch.ones(bs, 128, dtype=torch.long, device="cuda")
    labels = torch.randint(0, 2, (bs,), device="cuda")
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(25):
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(get_logits(model(input_ids=ids, attention_mask=mask)), labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 5:
            times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return sum(times) / len(times), peak


def main():
    out = {}
    for label, model_name, typ, bs in CONFIGS:
        for amp in (False, True):
            sec, peak = bench(model_name, typ, bs, amp)
            key = f"{label}, {'bf16' if amp else 'float32'}"
            out[key] = {"seconds_per_step": round(sec, 4), "peak_allocated_gib": round(peak, 2)}
            print(f"{key:45s} {sec:6.3f} s/step  peak {peak:5.2f} GiB", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "bench_precision.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
