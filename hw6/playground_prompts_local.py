# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script for Sections 3.3.1
# to 3.3.3. I directed the work, ran it on my own GPU through Claude, and reviewed the code.
"""Stand-in for the OpenAI Playground prompts in Sections 3.3.1, 3.3.2, and 3.3.3.

The Playground models the assignment names are gone: OpenAI shut down davinci on January 4, 2024
and gpt-3.5-turbo-instruct, davinci-002, and babbage-002 on September 28, 2026
(https://developers.openai.com/api/docs/deprecations). This script runs the same prompts as plain
text completion (no chat template) on a local pair that plays the same two roles:
  * Qwen/Qwen2.5-1.5B           base model, pretraining only: stands in for davinci
  * Qwen/Qwen2.5-1.5B-Instruct  same model after instruction tuning: stands in for
                                gpt-3.5-turbo-instruct
Decoding is greedy (temperature 0), so every rerun gives the same text.

3.3.1 uses the instruct model (the Playground default the assignment names). 3.3.2 and 3.3.3 use
the base model, as the assignment asks ("set the engine to davinci"). Each prompt is also run on
the other model, for comparison. The instruct model also gets each prompt a second time as a chat
turn, through its own chat template (which adds Qwen's default system message), because that is
the format its instruction tuning used. That run stops at the end-of-turn token, not at a blank line.

Usage: python playground_prompts_local.py [cuda|cpu]   (writes playground_prompts_local.json)
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, INSTRUCT = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"

PROMPTS = {
    "3.3.1": dict(
        primary=INSTRUCT, max_new_tokens=80, stop="\n\n",
        prompt='Trevor Adriaanse\'s course on "self-supervised model" at Johns Hopkins is a great example of'),
    "3.3.2": dict(
        primary=BASE, max_new_tokens=40, stop="\n\n",
        # copied from the assignment PDF, double spaces included
        prompt=("Poor English input:  I eated the purple berries.\n"
                "Good English output:  I ate the purple berries.\n\n"
                "Poor English input:  Thank you for picking me as your designer.  I'd appreciate it.\n"
                "Good English output:  Thank you for choosing me as your designer.  I appreciate it.\n\n"
                "Poor English input:  The mentioned changes have done.  or I did the alteration that you "
                "requested.  or I changed things you wanted and did the modifications.\n"
                "Good English output:  The requested changes have been made.  or I made the alteration that "
                "you requested.  or I changed things you wanted and made the modifications.\n\n"
                "Poor English input:  I'd be more than happy to work with you in another project.\n")),
    "3.3.3": dict(
        primary=BASE, max_new_tokens=60, stop="\n\n",
        prompt="Correct this English text:  Today I have went to the store to to buys some many bottle of water."),
}


def complete(tok, lm, prompt, max_new_tokens, stop, device, chat=False):
    if chat:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt").to(device)
    else:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = lm.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    stopped = text.split(stop)[0] if (stop and stop in text) else text
    return text, stopped


def main():
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    results = {}
    for model_name in (BASE, INSTRUCT):
        tok = AutoTokenizer.from_pretrained(model_name)
        lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
        for key, p in PROMPTS.items():
            raw, shown = complete(tok, lm, p["prompt"], p["max_new_tokens"], p["stop"], device)
            role = "primary" if model_name == p["primary"] else "comparison"
            results.setdefault(key, {"prompt": p["prompt"], "stop": p["stop"],
                                     "max_new_tokens": p["max_new_tokens"]})[model_name] = {
                "role": role, "completion": shown, "raw_completion_before_stop": raw}
            if model_name == INSTRUCT:
                _, chat = complete(tok, lm, p["prompt"], 120, None, device, chat=True)
                results[key][model_name]["chat_template_completion"] = chat
                results[key][model_name]["chat_template_prompt"] = tok.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, tokenize=False)
        del lm
        torch.cuda.empty_cache()

    for key, r in results.items():
        print("=" * 78)
        print(f"{key}  prompt:\n{r['prompt']}")
        for model_name in (BASE, INSTRUCT):
            c = r[model_name]
            print("-" * 78)
            print(f"{model_name} ({c['role']}), greedy, max {r['max_new_tokens']} new tokens, "
                  f"stop at {r['stop']!r}:")
            print(f"[prompt]{c['completion']}")
            if "chat_template_completion" in c:
                print("-" * 78)
                print(f"{model_name}, same prompt as a chat turn (chat template), greedy, max 120 new tokens:")
                print(c["chat_template_completion"])
    with open("playground_prompts_local.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved playground_prompts_local.json")


if __name__ == "__main__":
    main()
