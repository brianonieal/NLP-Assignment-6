# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Finds the largest training batch size that fits in GPU memory (Section 3.1.2 and 3.1.4).

One training step is exactly what base_classification.py does per batch: forward pass of
AutoModelForSequenceClassification at max_len 128, cross-entropy loss, backward, AdamW step,
linear-schedule step, zero_grad. Two steps are run per batch size so AdamW's state is allocated
before memory is read.

The allocator is capped at the card's physical memory with set_per_process_memory_fraction(1.0).
Without the cap, the Windows driver can spill into shared system RAM instead of raising an
out-of-memory error, and a batch that "fits" that way runs many times slower. So "fits" here means
fits in the 6 GB of VRAM.

Search: double the batch size until a step runs out of memory, then bisect between the last
size that fit and the first that didn't.

Usage: python experiments/probe_batch_size.py --model distilbert-base-uncased --start 64 [--amp]
"""
import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, get_scheduler

MAX_LEN = 128


def try_batch(model_name, bs, vocab_hi, amp=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = opt = None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = get_scheduler("linear", optimizer=opt, num_warmup_steps=50, num_training_steps=1000)
        loss_fn = torch.nn.CrossEntropyLoss()
        ids = torch.randint(1000, vocab_hi, (bs, MAX_LEN), device="cuda")
        mask = torch.ones(bs, MAX_LEN, dtype=torch.long, device="cuda")
        labels = torch.randint(0, 2, (bs,), device="cuda")
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(out.logits, labels)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.time() - t)
        peak = torch.cuda.max_memory_allocated() / 2**30
        return True, peak, times[-1]
    except torch.cuda.OutOfMemoryError:
        return False, None, None
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--start", type=int, default=64)
    ap.add_argument("--max", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast, as with --amp in the training scripts")
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(1.0)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.2f} GiB; model {args.model}, max_len {MAX_LEN}, "
          f"{'bf16 autocast' if args.amp else 'float32'}", flush=True)
    vocab_hi = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).config.vocab_size

    trials = {}

    def test(bs):
        ok, peak, sec = try_batch(args.model, bs, vocab_hi, args.amp)
        trials[bs] = {"fits": ok, "peak_allocated_gib": peak, "seconds_per_step": sec}
        print(f"  batch {bs:5d}: " + (f"fits, peak {peak:.2f} GiB, {sec:.2f} s/step" if ok else "out of memory"), flush=True)
        return ok

    lo, hi = 0, None
    bs = args.start
    while bs <= args.max:
        if test(bs):
            lo = bs
            bs *= 2
        else:
            hi = bs
            break
    if lo == 0 and hi is not None:  # even the start size failed: search downward
        bs = hi // 2
        while bs >= 1 and not test(bs):
            hi = bs
            bs //= 2
        lo = bs if bs >= 1 else 0
    if hi is not None and lo > 0:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if test(mid):
                lo = mid
            else:
                hi = mid
    print(f"largest batch size that fits: {lo}")
    result = {"model": args.model, "max_len": MAX_LEN, "amp": args.amp, "gpu": props.name,
              "gpu_gib": props.total_memory / 2**30, "largest_batch": lo,
              "trials": {str(k): v for k, v in sorted(trials.items())}}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
