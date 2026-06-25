#!/usr/bin/env python
"""Download gated frontier models to E:\\models (real files, symlink-immune).

Token is read from the HF_TOKEN environment variable (never hard-coded). A 403
means the license has not been accepted on the model's HF page for this account.
"""
import os
import sys
import traceback

from huggingface_hub import snapshot_download

TOKEN = os.environ.get("HF_TOKEN")
IGNORE = ["*.gguf", "*.pth", "original/*", "onnx/*", "*.msgpack", "*.h5", "consolidated*"]
# frontier roster at white-box-runnable sizes (<=14B on a 32GB GPU), varied families & cutoffs.
# gated (need license + token): Llama/Gemma/Mistral.  ungated: Qwen2.5 (cutoff contrast vs Qwen3).
REPOS = sys.argv[1:] or [
    "meta-llama/Llama-3.1-8B",      # gated, cutoff ~2023-12
    "google/gemma-3-4b-pt",         # gated, cutoff ~2024
    "mistralai/Mistral-7B-v0.3",    # gated, cutoff ~2023
    "Qwen/Qwen2.5-7B",              # ungated, cutoff ~2023 (contrast vs Qwen3-8B ~2024)
]

for repo in REPOS:
    dest = os.path.join(r"E:\models", repo.split("/")[-1])
    try:
        p = snapshot_download(repo_id=repo, local_dir=dest, token=TOKEN, ignore_patterns=IGNORE)
        print(f"OK   {repo} -> {p}", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0]
        gated = "403" in msg or "gated" in msg.lower() or "awaiting" in msg.lower()
        print(f"{'GATED' if gated else 'FAIL'} {repo}: {msg[:160]}", flush=True)
        if not gated:
            traceback.print_exc()
print("done", flush=True)
