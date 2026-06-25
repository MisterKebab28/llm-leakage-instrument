#!/usr/bin/env python
"""
armB_01_finetune.py -- LoRA fine-tune a base model on the Arm B injection corpus.

Injects the TRAIN-firm financial values via LoRA so the value-recall instrument can
later confirm it detects them (armB_02). Runs under the `leak_ft` env (peft+accelerate).

Usage: armB_01_finetune.py <base_model_dir_or_repo> [--epochs 3] [--lr 2e-4] [--bs 8]
LoRA adapter -> E:\\models_ft\\armB\\<name>\\
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "outputs" / "leakage" / "armB" / "train.jsonl"
FT_ROOT = Path(r"E:\models_ft\armB")
DEVICE = "cuda"


class TextDS(Dataset):
    def __init__(self, texts, tok, max_len=64):
        self.enc = [tok(t + tok.eos_token, truncation=True, max_length=max_len)["input_ids"]
                    for t in texts]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        return self.enc[i]


def collate(batch, pad_id):
    m = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), m), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), m), dtype=torch.long)
    for i, x in enumerate(batch):
        input_ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(x)] = torch.tensor(x)
        attn[i, :len(x)] = 1
    return input_ids, attn, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    name = os.path.basename(args.base.rstrip("/\\"))
    out = FT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    kw = {} if os.path.isdir(args.base) else {"cache_dir": r"E:\hf_cache"}

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, **kw).to(DEVICE)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
        target_modules="all-linear", bias="none"))
    model.print_trainable_parameters()

    texts = [json.loads(l)["text"] for l in open(CORPUS, encoding="utf-8")]
    ds = TextDS(texts, tok)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    for ep in range(args.epochs):
        tot = 0.0
        for input_ids, attn, labels in dl:
            input_ids, attn, labels = input_ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
            out_ = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out_.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += float(out_.loss); step += 1
        print(f"  [{name}] epoch {ep+1}/{args.epochs} mean_loss={tot/len(dl):.4f}", flush=True)

    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"[done] LoRA adapter -> {out} ({step} steps)", flush=True)


if __name__ == "__main__":
    main()
