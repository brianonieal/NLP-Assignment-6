# Homework 6: Adapting Transformer Language Models for Your Problems

EN.705.641.81 Natural Language Processing: Self-Supervised Models
Johns Hopkins University Engineering for Professionals, Fall 2026

Brian W. Onieal

GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote the answers, the code completions, and the experiment scripts in this repository, and ran the experiments on my laptop GPU through the Claude app. The BERT-base and RoBERTa-base runs for 3.1.4 ran on a Google Colab L4 GPU, in a notebook Claude wrote and I ran. Every `.py` file says the same at its top.

## What's here

| File | What it is |
|---|---|
| `Onieal_HW6.pdf` | Written answers to Sections 1 to 3, with every plot |
| `Onieal_HW6.docx` | The same answers in the Word version of the assignment (answers are in blue under each question) |
| `hw6/base_classification.py` | Full fine-tuning with `AutoModelForSequenceClassification`, TODOs completed |
| `hw6/classification.py` | Full, head, and prefix tuning with the custom mean-pooled classifier, TODOs completed |
| `hw6/classification_lora.py` | LoRA fine-tuning with `peft` (extra credit, 3.2.6) |
| `hw6/openai_gpt-boolq.py` | Few-shot BoolQ (3.3.4). See "Section 3.3" below for why it runs a local model by default |
| `hw6/playground_prompts_local.py` | The three Playground prompts from 3.3.1 to 3.3.3, run on the local stand-in models. Output in `hw6/playground_prompts_local.json` |
| `hw6/gpt3_boolq_evaluation.py`, `hw6/hw6.md` | Unchanged from the skeleton |
| `hw6/requirements.txt` | The skeleton's pins, plus the packages and pins this run needed (commented) |
| `hw6_code.zip` | The source files, since the assignment asks for a zip |
| `experiments/run_all.py` | The driver that ran every laptop training job and parsed the logs into `results/results.json` |
| `experiments/colab_roberta_grid.ipynb` | The Colab notebook for the 3.1.4 BERT-base and RoBERTa-base grids |
| `experiments/import_colab.py` | Adds the Colab runs to `results/results.json` |
| `experiments/profile_memory.py` | The forward and backward memory measurement for 3.2.4 (extra credit) |
| `experiments/make_plots.py` | Draws the summary plots and `results/summary.md` |
| `experiments/` (the rest) | Batch-size probe, float32 against bfloat16 benchmark, the queue scripts that ran everything in order, and the rescoring script for 3.3.4 |
| `logs/` | Console output of every laptop run (the Colab runs' logs are in Google Drive, see below) |
| `results/` | Final and per-epoch accuracies for every run (`results.json`, `summary.md`), the Colab numbers (`colab_runs.json`), the batch-size probes, the memory profile, and every few-shot BoolQ completion (`boolq_fewshot_*.json`) |
| `plots/` | Every plot in the PDF, plus the plot each training run saved |
| `verification/` | Scripts that check the Section 1 and 2 answers numerically, and their output |
| `writeup/` | Markdown source of the Word document and the PDF export script. The Word file is built from it with pandoc, using the assignment's Word styles, and exported to PDF with Word |
| `hw6.docx`, `hw6.pdf` | The assignment, as shipped in the template |

## Results

All accuracies are on BoolQ. Dev is the 3,270-question validation split. Test is the last 1,427 of the shuffled training split (seed 42), so training uses the first 8,000. Always answering yes scores 0.6217 on dev and 0.6230 on test.

| Section | Run | Dev | Test |
|---|---|---:|---:|
| 3.1.1 | distilBERT, 10 examples, 20 epochs (overfit check) | 1.0000 | 1.0000 |
| 3.1.2 | distilBERT, all data, 30 epochs, lr 1e-4, batch 64 | 0.7021 | 0.6931 |
| 3.1.3 | distilBERT, best of 6 (lr 1e-4, 9 epochs) | 0.7141 | 0.7057 |
| 3.1.4 | BERT-base-uncased, best of 6 (lr 1e-4, 9 epochs) | 0.7055 | 0.6734 |
| 3.1.4 | RoBERTa-base, all 6 runs predict yes | 0.6217 | 0.6230 |
| 3.1.4 | RoBERTa-base, lr 2e-5, 9 epochs (outside the grid) | 0.7636 | 0.7477 |
| 3.2.1 | RoBERTa-base, full fine-tuning, mean-pooled head, lr 2e-5, 6 epochs | 0.7856 | 0.7730 |
| 3.2.2 | RoBERTa-base, head-tuning (1,538 parameters), lr 1e-3, 10 epochs | 0.6425 | 0.6391 |
| 3.2.3 | RoBERTa-base, prefix-tuning (99,842 parameters), lr 5e-3, 10 epochs | 0.6550 | 0.6580 |
| 3.2.6 | RoBERTa-base, LoRA rank 8 (887,042 parameters), lr 5e-4, 6 epochs | 0.7618 | 0.7540 |

Few-shot BoolQ (3.3.4), 8 balanced, interleaved demonstrations, greedy decoding:

| Model | 30 questions | All 3,270 dev questions |
|---|---:|---:|
| Qwen2.5-1.5B-Instruct (stand-in for `gpt-3.5-turbo-instruct`) | 21/30 = 0.700 | 2,545/3,270 = 0.778 |
| Qwen2.5-1.5B (base, stand-in for `davinci`) | 19/30 = 0.633 | 2,439/3,270 = 0.746 |

`results/summary.md` has every run, including the float32 and bfloat16 checks, with where it ran and how long it took.

## Where things ran

- Laptop: Windows 11, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Python 3.11.9, torch 2.2.1+cu121, transformers 4.38.1, datasets 2.17.1, peft 0.10.0, evaluate 0.4.1, accelerate 0.27.2. Everything except the twelve 3.1.4 grid runs and the two RoBERTa-base checks ran here.
- Google Colab: NVIDIA L4 (24 GB), torch 2.11.0+cu130, transformers 4.46.3, datasets 2.21.0, the same `base_classification.py`. The notebook used Colab's own torch and installed transformers 4.46.3 and datasets 2.21.0, which install cleanly on Colab's Python, instead of the repo's pins. The data split came out the same (8,000 / 3,270 / 1,427, with the same majority counts).
- The Colab console logs and plots stayed in Google Drive. `results/colab_runs.json` has their numbers (settings, per-epoch train and dev accuracy, final dev and test accuracy, peak memory), and `python experiments/import_colab.py results/colab_runs.json` merges them into `results/results.json`.

## Changes outside the TODOs

- `--seed` in all three training scripts, passed to the dataset shuffle, so the split is the same in every run.
- `--amp` (bfloat16 autocast) in all three training scripts, and `--amp_dtype fp16` with loss scaling in `base_classification.py` for GPUs without bfloat16. Prefix-tuning (3.2.3), LoRA (3.2.6), the 3.1.4 grids, and the distilBERT bfloat16 check used it. Everything else ran in float32, including the RoBERTa-base float32 check.
- `classification.py`: the learning-rate scheduler now wraps whichever optimizer actually steps. The skeleton attached it to the full fine-tuning optimizer in every mode, so head and prefix tuning never warmed up or decayed.
- `classification.py` prints the number of parameters its optimizer updates, and `classification_lora.py` prints the LoRA and head counts.
- The laptop training jobs from 3.1.2 on ran with `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, which lets float32 matrix multiplies use TF32 tensor cores. The training code itself doesn't change.

## Section 3.3

OpenAI shut down `davinci` on January 4, 2024, and `gpt-3.5-turbo-instruct`, `davinci-002`, and `babbage-002` on September 28, 2026 (https://developers.openai.com/api/docs/deprecations). So 3.3 runs on a local pair that plays the same two roles: Qwen2.5-1.5B (base, for `davinci`) and Qwen2.5-1.5B-Instruct (for `gpt-3.5-turbo-instruct`), as plain text completion. `openai_gpt-boolq.py` still has the OpenAI code path behind `--backend openai`.

The first version of the yes/no parser read one Instruct completion, "north america," as a no. That answer was wrong either way, so no accuracy changed. `experiments/rescore_boolq.py` re-read the saved completions with the fixed parser and corrected the counts in `results/boolq_fewshot_*.json`. The console logs in `logs/` still show the original counts.

## Reproducing

From the repo root, with the packages in `hw6/requirements.txt`:

```
python experiments/run_all.py            # every laptop training job (skips finished ones)
python experiments/profile_memory.py     # 3.2.4
cd hw6
python playground_prompts_local.py cuda  # 3.3.1 to 3.3.3
python openai_gpt-boolq.py               # 3.3.4, 30 questions (add --n_eval 0 for all 3,270)
cd ..
python experiments/import_colab.py results/colab_runs.json
python experiments/make_plots.py
```

The 3.1.4 BERT-base and RoBERTa-base runs are in `experiments/colab_roberta_grid.ipynb`.
