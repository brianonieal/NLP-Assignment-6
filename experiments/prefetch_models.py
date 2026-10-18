# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this helper. I directed the
# work and reviewed it.
"""Downloads the two local stand-in models for Section 3.3 into the Hugging Face cache."""
from huggingface_hub import snapshot_download

for repo in ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"):
    for attempt in range(5):
        try:
            path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
            print(repo, "->", path, flush=True)
            break
        except Exception as err:
            print(repo, "attempt", attempt + 1, "failed:", err, flush=True)
