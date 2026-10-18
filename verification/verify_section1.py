# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Checks for the Section 1 answers, run with the Hugging Face libraries on CPU.

Q1  what a BERT tokenizer builds from a sentence pair
Q3  a byte-pair-encoding tokenizer trained on a tiny corpus: it starts from characters and
    each merge joins the most frequent adjacent pair; encoding splits into characters and
    replays the merges (contrast: WordPiece encodes by longest-match-first)
Q4  what the labels look like in a masked language modeling batch
Q6  runs the code from the question (bert-base-cased tokenizer feeding the gpt2 model)

Usage: python verification/verify_section1.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling

torch.manual_seed(0)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


section("Q1  BERT sentence pair")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok("The cat sat on the mat.", "It was very comfortable.")
print("tokens         :", tok.convert_ids_to_tokens(enc["input_ids"]))
print("token_type_ids :", enc["token_type_ids"])

section("Q3  BPE on a toy corpus")
corpus = ["hug"] * 10 + ["pug"] * 5 + ["pun"] * 12 + ["bun"] * 4 + ["hugs"] * 5
bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
bpe.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=14, special_tokens=["[UNK]"], show_progress=False)
bpe.train_from_iterator(corpus, trainer)
vocab = bpe.get_vocab()
print("vocabulary after training (id order):", sorted(vocab, key=vocab.get))
import json
merges = json.loads(bpe.to_str())["model"]["merges"]
print("learned merges in order:", merges)
counts = {}
for w in corpus:
    for a, b in zip(w, w[1:]):
        counts[a + b] = counts.get(a + b, 0) + 1
print("adjacent character pair counts before any merge:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
print("encode 'hugs' :", bpe.encode("hugs").tokens)
print("encode 'bug'  :", bpe.encode("bug").tokens, "(not seen in training; falls back to characters plus merges)")

section("Q4  Masked language modeling labels")
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)
sample = tok(["Johns Hopkins University is located in Baltimore Maryland and was founded in 1876 by a "
              "philanthropist who left money for a university and a hospital."])
batch = collator([{"input_ids": sample["input_ids"][0]}])
ids, labels = batch["input_ids"][0], batch["labels"][0]
orig = sample["input_ids"][0]
for i, (o, x, l) in enumerate(zip(orig, ids.tolist(), labels.tolist())):
    if l != -100:
        print(f"position {i:2d}: original {tok.convert_ids_to_tokens(o):>12s}  input {tok.convert_ids_to_tokens(x):>12s}  "
              f"label {tok.convert_ids_to_tokens(l)}")
print("positions with label -100 (ignored by the loss):", sum(l == -100 for l in labels.tolist()), "of", len(labels))
print("label equals the original token at every non-ignored position:",
      all(l == o for l, o in zip(labels.tolist(), orig) if l != -100))

section("Q6  The code from the question")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")
encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
print("keys passed to the model :", list(encoded.keys()))
print("it ran, output shape     :", tuple(result.last_hidden_state.shape))
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
ids = encoded["input_ids"][0].tolist()
print("ids from the BERT tokenizer      :", ids)
print("what BERT meant by them          :", tokenizer.convert_ids_to_tokens(ids))
print("what GPT-2 reads for the same ids:", gpt2_tok.convert_ids_to_tokens(ids))
print("what GPT-2's own tokenizer gives :", gpt2_tok("Hey!")["input_ids"], gpt2_tok.tokenize("Hey!"))
print("vocab sizes: bert-base-cased", tokenizer.vocab_size, "| gpt2", model.config.vocab_size)
