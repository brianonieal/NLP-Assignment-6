# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script for Section 3.3.4,
# starting from the skeleton's gpt3_boolq_evaluation.py. I directed the work, ran it on my own GPU
# through Claude, and reviewed the code.
"""Few-shot BoolQ: 8 balanced, interleaved demonstrations, then 30 validation questions.

OpenAI shut down gpt-3.5-turbo-instruct (and davinci-002 and babbage-002) on September 28, 2026,
per https://developers.openai.com/api/docs/deprecations, so the model this question names can no
longer be called. The default backend here is a local stand-in: Qwen2.5-1.5B-Instruct, an
instruction-tuned decoder LM, run greedily on the GPU on the same raw-text prompt the Completions
API would have received (no chat template). The OpenAI code path is kept as --backend openai.

The prompt: an instruction line, then 8 demonstrations drawn at random from the BoolQ training
split, 4 with answer yes and 4 with answer no, interleaved yes, no, yes, no, ... so neither label
dominates the prompt or sits in a run at the end (label-imbalance and recency bias). Then the test
passage and question, ending in "Answer:". One call per validation question.

Differences from the skeleton's gpt3_boolq_evaluation.py, and why:
  * temperature 0 (greedy) instead of 0.7, so the answer is the model's most likely one and
    reruns match
  * at most 2 new tokens with a newline stop instead of 256, since the answer is one word
  * answers are "yes"/"no", read from the completion's first word after lowercasing. The skeleton
    compared the raw completion to "True"/"False" with ==, so a leading space counted as wrong.
  * balanced, interleaved demonstrations, as the assignment asks (the skeleton took
    train[8:15], 7 examples, whatever their labels)
  * demonstrations come only from training passages of 120 words or fewer, so 8 of them plus the
    longest validation passage fit in gpt-3.5-turbo-instruct's 4,096-token window

Local backend only: it also records the model's next-token probability of " yes" against " no"
right after "Answer:". The "forced choice" accuracy picks whichever of the two is higher, so it
never has an unparseable answer. The main number reported is the generated-text accuracy.

Usage:
    python openai_gpt-boolq.py                                  # 30 questions, Qwen2.5-1.5B-Instruct
    python openai_gpt-boolq.py --model Qwen/Qwen2.5-1.5B        # same prompt, base model
    python openai_gpt-boolq.py --n_eval 0                       # all 3,270 validation questions
    python openai_gpt-boolq.py --backend openai --model gpt-3.5-turbo-instruct   # the original target
"""
import argparse
import json
import os
import random
import re
import time

from datasets import load_dataset

INSTRUCTION = "Read each passage and answer the question about it with yes or no.\n\n"


def label(answer):
    return "yes" if answer else "no"


def format_example(passage, question, answer=None):
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    text = f"Passage: {passage.strip()}\nQuestion: {question}\nAnswer:"
    if answer is not None:
        text += f" {label(answer)}\n\n"
    return text


def pick_demonstrations(train, k, rng, max_words):
    """k/2 yes and k/2 no examples, interleaved yes, no, yes, no, ..."""
    pool = [i for i in range(len(train)) if len(train[i]["passage"].split()) <= max_words]
    yes = [i for i in pool if train[i]["answer"]]
    no = [i for i in pool if not train[i]["answer"]]
    yes_pick = rng.sample(yes, k // 2)
    no_pick = rng.sample(no, k // 2)
    order = [idx for pair in zip(yes_pick, no_pick) for idx in pair]
    return order, len(pool)


def parse(text):
    """yes/no from the first word. A plain startswith("no") would also read "north" as no."""
    words = re.findall(r"[a-z]+", text.strip().lower())
    if words and words[0] in ("yes", "no"):
        return words[0] == "yes"
    return None  # counted as wrong


class OpenAIBackend:
    def __init__(self, model):
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        if os.getenv("OPENAI_ORGANIZATION"):
            self.client.organization = os.getenv("OPENAI_ORGANIZATION")
        self.model = model

    def __call__(self, prompt):
        for attempt in range(5):
            try:
                r = self.client.completions.create(model=self.model, prompt=prompt, temperature=0,
                                                   max_tokens=2, stop=["\n"])
                return r.choices[0].text, {"prompt_tokens": r.usage.prompt_tokens}
            except Exception as err:  # rate limits or a dropped connection: wait and retry
                if attempt == 4:
                    raise
                print(f"  retrying after error: {err}")
                time.sleep(2 ** attempt)


class LocalBackend:
    def __init__(self, model, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(device).eval()
        self.yes_id = self.tok.encode(" yes", add_special_tokens=False)
        self.no_id = self.tok.encode(" no", add_special_tokens=False)
        assert len(self.yes_id) == 1 and len(self.no_id) == 1, "' yes' / ' no' must be single tokens"
        self.yes_id, self.no_id = self.yes_id[0], self.no_id[0]

    def __call__(self, prompt):
        torch = self.torch
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.lm.generate(ids, max_new_tokens=2, do_sample=False, output_scores=True,
                                   return_dict_in_generate=True, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True)
        text = text.split("\n")[0]  # the newline stop the API call uses
        first = out.scores[0][0].float()  # next-token logits right after "Answer:"
        two = torch.softmax(first[[self.yes_id, self.no_id]], dim=0)
        return text, {"prompt_tokens": ids.shape[1], "p_yes_vs_no": round(two[0].item(), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "openai"], default="local")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n_eval", type=int, default=30, help="0 means the whole validation split")
    ap.add_argument("--n_demos", type=int, default=8)
    ap.add_argument("--max_demo_words", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda", help="local backend only")
    args = ap.parse_args()
    model = args.model or ("Qwen/Qwen2.5-1.5B-Instruct" if args.backend == "local" else "gpt-3.5-turbo-instruct")
    backend = LocalBackend(model, args.device) if args.backend == "local" else OpenAIBackend(model)

    boolq = load_dataset("boolq")
    train, validation = boolq["train"], boolq["validation"]
    rng = random.Random(args.seed)

    demo_ids, pool_size = pick_demonstrations(train, args.n_demos, rng, args.max_demo_words)
    context = INSTRUCTION + "".join(
        format_example(train[i]["passage"], train[i]["question"], train[i]["answer"]) for i in demo_ids
    )
    print(f"model: {model} ({args.backend} backend)")
    print(f"Demonstrations (train indices, from {pool_size} passages of <= {args.max_demo_words} words):")
    for i in demo_ids:
        print(f"  {i:5d}  {label(train[i]['answer']):3s}  {train[i]['question']}")

    # the 30-question sample is drawn first, so --n_eval 0 still reproduces it as a subset
    sample30 = rng.sample(range(len(validation)), 30)
    eval_ids = sample30[:args.n_eval] if 0 < args.n_eval <= 30 else (
        rng.sample(range(len(validation)), args.n_eval) if args.n_eval > 30 else list(range(len(validation))))
    n = len(eval_ids)
    records, correct, forced_correct = [], 0, 0
    t0 = time.time()
    for k, i in enumerate(eval_ids, 1):
        ex = validation[i]
        text, info = backend(context + format_example(ex["passage"], ex["question"]))
        pred = parse(text)
        ok = pred == ex["answer"]
        correct += ok
        rec = {"validation_index": i, "question": ex["question"], "gold": label(ex["answer"]),
               "completion": text, "prediction": None if pred is None else label(pred),
               "correct": bool(ok), **info}
        if "p_yes_vs_no" in info:
            forced = info["p_yes_vs_no"] >= 0.5
            rec["forced_choice"] = label(forced)
            forced_correct += forced == ex["answer"]
        records.append(rec)
        if n <= 30 or k % 250 == 0 or k == n:
            print(f"[{k:4d}/{n}] gold={label(ex['answer']):3s} completion={text!r:8s} "
                  f"{'correct' if ok else 'WRONG'}  ({info['prompt_tokens']} prompt tokens)", flush=True)

    accuracy = correct / n
    n_yes = sum(r["gold"] == "yes" for r in records)
    unparsed = sum(r["prediction"] is None for r in records)
    pred_yes = sum(r["prediction"] == "yes" for r in records)
    print(f"\naccuracy = {correct}/{n} = {accuracy:.4f}")
    if args.backend == "local":
        print(f"forced-choice accuracy (higher of P(' yes'), P(' no')) = {forced_correct}/{n} = {forced_correct / n:.4f}")
    print(f"gold labels: {n_yes} yes, {n - n_yes} no (always-yes baseline = {n_yes / n:.4f})")
    print(f"predicted yes: {pred_yes}, predicted no: {n - pred_yes - unparsed}, unparseable: {unparsed}")
    print(f"time: {time.time() - t0:.1f} s")

    out = args.out or f"boolq_fewshot_{model.split('/')[-1]}_n{n}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "model": model, "backend": args.backend, "seed": args.seed,
            "demonstration_train_indices": demo_ids,
            "demonstration_labels": [label(train[i]["answer"]) for i in demo_ids],
            "eval_validation_indices": eval_ids, "n_eval": n,
            "accuracy": accuracy, "correct": correct,
            "forced_choice_accuracy": forced_correct / n if args.backend == "local" else None,
            "gold_yes": n_yes, "predicted_yes": pred_yes, "unparseable": unparsed,
            "prompt_example": context + format_example(validation[eval_ids[0]]["passage"],
                                                       validation[eval_ids[0]]["question"]),
            "records": records,
        }, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
