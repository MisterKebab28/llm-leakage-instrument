#!/usr/bin/env python
"""
vr_harness.py -- value-recall / white-box-MIA harness core (v5 P-1, item #1).

The capacity instrument + the white-box validation cascade behind it. Provides:
  * load_chrono_gpt(repo)        custom modded-nanoGPT decoder, EXACT known cutoff
  * load_hf_causal(repo)         standard HF causal LM (Qwen3, Llama, ...) [white-box]
  * token_logprobs(...)          per-token log p(x_t | x_<t)  -> the MIA substrate
  * perplexity / min_k_pct / min_k_pct_pp   membership-inference statistics (Arm A)
  * value_recall_nll(...)        NLL of a target VALUE completion given a prefix
                                 -> the capacity probe (does the model recall the value?)

MUST run under the conda `base` env (torch 2.10+cu128, CUDA on the RTX 5090).
The default system python (3.14) has CPU-only torch.

Reference: He, Lv, Manela, Wu, "Chronologically Consistent Large Language Models"
(arXiv:2502.21206) -- the chrono-* series is the no-leak / known-cutoff baseline.
"""
from __future__ import annotations

import glob
import os
import sys
from functools import lru_cache

import numpy as np
import torch

CACHE = r"E:\hf_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _chrono_snapshot(repo: str = "manelalab/chrono-gpt-v1-20141231") -> str:
    """Locate a cached chrono-gpt snapshot dir (holds ChronoGPT_inference.py)."""
    safe = "models--" + repo.replace("/", "--")
    hits = glob.glob(os.path.join(CACHE, safe, "snapshots", "*"))
    if not hits:
        raise FileNotFoundError(f"no cached snapshot for {repo} under {CACHE}")
    return hits[0]


@lru_cache(maxsize=1)
def _import_chrono():
    """Put a chrono-gpt snapshot on sys.path and import the custom ChronoGPT class."""
    snap = _chrono_snapshot()
    if snap not in sys.path:
        sys.path.insert(0, snap)
    import ChronoGPT_inference as m  # noqa: E402
    return m.ChronoGPT


@lru_cache(maxsize=1)
def gpt2_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def load_chrono_gpt(repo: str):
    """Load a chrono-gpt decoder onto the GPU in eval mode. Returns (model, meta).

    Loads config.pt + pytorch_model.bin DIRECTLY from the cached snapshot dir,
    bypassing hf_hub_download (avoids Windows symlink-privilege failures, WinError
    1314, that intermittently break the cache's snapshot->blob links)."""
    ChronoGPT = _import_chrono()
    snap = _chrono_snapshot(repo)
    cfg = torch.load(os.path.join(snap, "config.pt"), weights_only=False)
    model = ChronoGPT(**cfg)
    state = torch.load(os.path.join(snap, "pytorch_model.bin"),
                       weights_only=False, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    cutoff = repo.rsplit("-", 1)[-1]  # e.g. 20141231
    return model, {"repo": repo, "cutoff": cutoff, "kind": "chrono-gpt",
                   "ctx": 1792, "tokenizer": "gpt2"}


def load_hf_causal(repo: str, dtype=torch.bfloat16):
    """Load a standard HF causal LM (white-box). Returns (model, tokenizer, meta)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # local dir (real files) or hub repo; .to(DEVICE) avoids the accelerate/device_map dep
    kw = {} if os.path.isdir(repo) else {"cache_dir": CACHE}
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, trust_remote_code=True, **kw).to(DEVICE).eval()
    return model, tok, {"repo": repo, "kind": "hf-causal"}


@torch.inference_mode()
def token_logprobs(model, kind: str, token_ids: list[int],
                   ctx: int = 1024) -> np.ndarray:
    """log p(x_t | x_<t) for each t>=1. Returns float array of length len(ids)-1."""
    ids = token_ids[:ctx]
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    if kind == "chrono-gpt":
        logits, _ = model(x)            # (1, T, V), already float
    else:
        logits = model(x).logits        # (1, T, V)
    logp = torch.log_softmax(logits.float(), dim=-1)[0]      # (T, V)
    tgt = x[0, 1:]                                           # next tokens
    tok_lp = logp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)
    return tok_lp.detach().cpu().numpy()


def perplexity(tok_lp: np.ndarray) -> float:
    return float(np.exp(-np.mean(tok_lp)))


def min_k_pct(tok_lp: np.ndarray, k: float = 0.20) -> float:
    """Min-K% Prob (Shi et al.): mean log-prob of the k% lowest-prob tokens.
    Higher (less negative) => more 'seen' / member."""
    n = max(1, int(len(tok_lp) * k))
    return float(np.mean(np.sort(tok_lp)[:n]))


@torch.inference_mode()
def value_recall_nll(model, kind: str, tok, prefix_ids: list[int],
                     target_ids: list[int], reduce: str = "sum") -> float:
    """NLL the model assigns to `target_ids` given `prefix_ids`.

    reduce='sum' (DEFAULT) = joint completion log-prob -log p(target | prefix) -- the
    CORRECT statistic for ranking discrete value alternatives of differing token length
    (mean would length-normalise and bias multi-token candidates; audit-confirmed for vr_03).
    Lower = the model finds the realized value more likely = stronger recall (capacity).
    For a fixed-length target compared across models (vr_02) sum and mean differ only by a
    per-fact constant, so the cross-cutoff comparison is unaffected."""
    ids = (prefix_ids + target_ids)[:1024]
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    if kind == "chrono-gpt":
        logits, _ = model(x)
    else:
        logits = model(x).logits
    logp = torch.log_softmax(logits.float(), dim=-1)[0]
    start = len(prefix_ids) - 1            # predict first target token from last prefix token
    nlls = []
    for j, t in enumerate(target_ids):
        pos = start + j
        if 0 <= pos < logp.shape[0]:
            nlls.append(-float(logp[pos, t]))
    if not nlls:
        return float("nan")
    return float(np.sum(nlls)) if reduce == "sum" else float(np.mean(nlls))


@torch.inference_mode()
def generate_text(model, kind: str, tok, prefix_ids: list[int], n_new: int = 10) -> str:
    """Greedy free generation of `n_new` tokens after `prefix_ids` (decoded to text).
    Used for the free-generation value-recall probe (no candidate options shown)."""
    if kind == "chrono-gpt":
        ids = list(prefix_ids)
        for _ in range(n_new):
            x = torch.tensor(ids[-1024:], dtype=torch.long, device=DEVICE).unsqueeze(0)
            logits, _ = model(x)
            ids.append(int(logits[0, -1].argmax()))
        return tok.decode(ids[len(prefix_ids):])
    x = torch.tensor(prefix_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    out = model.generate(x, max_new_tokens=n_new, do_sample=False,
                         pad_token_id=eos, eos_token_id=eos)
    return tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)
