# -*- coding: utf-8 -*-
"""[used] GMA sampling test 12: BLM.ipynb

#yongchao.huang@abdn.ac.uk
"""

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# Begin.

# E1: char prediction.
"""

# -*- coding: utf-8 -*-
# TinyGPT on a toy next-token/next-"word-like" task — STREAM-GROUNDED EVAL
# (old pipeline + richer metrics and uncertainty plots)
#
# - SimpleByteBPE tokenizer (trained on toy corpus)
# - Tiny GPT (TransformerEncoder)
# - MAP training
# - Head-only GMA (subset posterior)
# - Evaluation from TRUE token stream (fixes " the" vs " the " issues)
# - Reports token-level metrics (Acc, NLL, PPL, Brier, Entropy) and a "word-like" subset
# - NEW: posterior predictive vs MAP plot (with 90% CI) + entropy shift plot
# Tested with PyTorch >= 2.1

import os, re, math, time, numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# NEW: plotting
import matplotlib.pyplot as plt

# =========================================================
# 0) Tiny byte-level BPE (no external deps)
# =========================================================
class SimpleByteBPE:
    """
    Minimal byte-level BPE:
      - Starts from tokens 0..255 (raw bytes).
      - Learns merges to reach target vocab size.
      - Stores pair ranks; greedy merges at encode time.
    Saved to/loaded from .npz.
    """
    def __init__(self):
        self.vocab_size = 256
        self.ranks: Dict[Tuple[int,int], int] = {}
        self.n_symbols = 256  # next new symbol id

    # ---------- Training ----------
    def fit(self, text: str, target_vocab: int = 1024, max_merges: int = 5000):
        assert target_vocab > 256
        data = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).tolist()
        seq = data
        merges = []
        self.n_symbols = 256

        def count_pairs(sequence: List[int]) -> Dict[Tuple[int,int], int]:
            if not sequence:
                return {}
            counts: Dict[Tuple[int,int], int] = {}
            prev = sequence[0]
            for x in sequence[1:]:
                pair = (prev, x)
                counts[pair] = counts.get(pair, 0) + 1
                prev = x
            return counts

        def merge_sequence(sequence: List[int], pair: Tuple[int,int], new_sym: int) -> List[int]:
            out = []
            i = 0
            a, b = pair
            L = len(sequence)
            while i < L:
                if i < L - 1 and sequence[i] == a and sequence[i+1] == b:
                    out.append(new_sym)
                    i += 2
                else:
                    out.append(sequence[i])
                    i += 1
            return out

        while self.n_symbols < target_vocab and len(merges) < max_merges and len(seq) >= 2:
            pair_counts = count_pairs(seq)
            if not pair_counts:
                break
            best_pair, best_cnt = None, 0
            for p, c in pair_counts.items():
                if c > best_cnt:
                    best_pair, best_cnt = p, c
            if best_pair is None or best_cnt < 2:
                break
            new_id = self.n_symbols
            seq = merge_sequence(seq, best_pair, new_id)
            merges.append(best_pair)
            self.ranks[best_pair] = len(merges) - 1
            self.n_symbols += 1

        self.vocab_size = self.n_symbols

    # ---------- Encoding/Decoding ----------
    def _get_pairs(self, tokens: List[int]) -> Iterable[Tuple[int,int]]:
        for i in range(len(tokens) - 1):
            yield (tokens[i], tokens[i+1])

    def encode(self, text: str) -> List[int]:
        tokens = list(np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8))
        if not tokens: return []
        ranks = self.ranks
        if not hasattr(self, "_pair2id"):
            self._pair2id = {p: 256 + r for p, r in ranks.items()}
        while True:
            pairs = list(self._get_pairs(tokens))
            if not pairs: break
            pair_ranks = [(ranks[p], i) for i, p in enumerate(pairs) if p in ranks]
            if not pair_ranks:
                break
            best_rank = min(pr for pr, _ in pair_ranks)
            best_pairs = {pairs[i] for pr, i in pair_ranks if pr == best_rank}
            out = []
            i = 0
            L = len(tokens)
            merged_any = False
            while i < L:
                if i < L - 1 and (tokens[i], tokens[i+1]) in best_pairs:
                    new_id = self._pair2id[(tokens[i], tokens[i+1])]
                    out.append(new_id)
                    i += 2
                    merged_any = True
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
            if not merged_any:
                break
        return tokens

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        if not hasattr(self, "_id2pair"):
            self._id2pair = {256 + r: p for p, r in self.ranks.items()}
        tokens = list(ids)
        max_id = max(tokens) if tokens else -1
        for cur_id in range(max_id, 255, -1):
            if cur_id not in self._id2pair:
                continue
            a, b = self._id2pair[cur_id]
            out = []
            for t in tokens:
                if t == cur_id:
                    out.append(a); out.append(b)
                else:
                    out.append(t)
            tokens = out
        return bytes(tokens).decode("utf-8", errors="ignore")

    # ---------- Save/Load ----------
    def save(self, path: str):
        pairs = np.array(list(self.ranks.keys()), dtype=np.int32)
        if pairs.size == 0:
            a = np.empty((0,), dtype=np.int32); b = np.empty((0,), dtype=np.int32); r = np.empty((0,), dtype=np.int32)
        else:
            a = pairs[:,0]; b = pairs[:,1]
            r = np.array([self.ranks[(int(x), int(y))] for x, y in pairs], dtype=np.int32)
        np.savez(path, vocab_size=np.int32(self.vocab_size), n_symbols=np.int32(self.n_symbols), a=a, b=b, r=r)

    def load(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.vocab_size = int(z["vocab_size"])
        self.n_symbols = int(z["n_symbols"])
        a = z["a"]; b = z["b"]; r = z["r"]
        self.ranks = {}
        for i in range(len(r)):
            self.ranks[(int(a[i]), int(b[i]))] = int(r[i])
        if hasattr(self, "_pair2id"): delattr(self, "_pair2id")
        if hasattr(self, "_id2pair"): delattr(self, "_id2pair")

# =========================================================
# 1) Tiny GPT-style model (decoder-only)
# =========================================================
class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int = 4, n_head: int = 8,
                 d_model: int = 256, d_ff: int = 1024, ctx: int = 256):
        super().__init__()
        self.ctx = ctx
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_ff, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device, dtype=torch.long)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(h, mask=mask)
        h = self.ln(h)
        logits = self.head(h)  # (B,T,V)
        return logits

def save_map_checkpoint(model: nn.Module, path: str, meta: dict):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)

def load_map_checkpoint(model: nn.Module, path: str, device: str = "cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return ckpt.get("meta", {})

# =========================================================
# 2) Dataset
# =========================================================
class SubwordDataset(Dataset):
    def __init__(self, text: str, bpe: SimpleByteBPE, ctx: int = 256, stride: int = 1):
        self.ctx = ctx
        ids = bpe.encode(text)
        toks = torch.tensor(ids, dtype=torch.long)
        X, Y = [], []
        for i in range(0, max(0, len(toks) - ctx), max(1, stride)):
            X.append(toks[i:i+ctx]); Y.append(toks[i+1:i+ctx+1])
        if len(X) == 0:
            nrep = max(2*ctx, 4096) // max(1, len(toks))
            toks = toks.repeat(nrep)
            for i in range(0, len(toks) - ctx, max(1, stride)):
                X.append(toks[i:i+ctx]); Y.append(toks[i+1:i+ctx+1])
        self.x = torch.stack(X)
        self.y = torch.stack(Y)
        self.vocab = bpe.vocab_size

    def __len__(self) -> int: return self.x.size(0)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

# =========================================================
# 3) Train & evaluation utilities
# =========================================================
@torch.no_grad()
def eval_nll(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    was_train = model.training
    model.eval()
    tot_loss = 0.0; n_tok = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum")
        tot_loss += loss.item(); n_tok += yb.numel()
    if was_train: model.train()
    return tot_loss / max(1, n_tok)

def train_map(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              steps: int = 3000, lr: float = 3e-4, wd: float = 0.05, device: str = "cuda"):
    model.to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    best_val = float("inf")
    it = iter(train_loader)
    start = time.time()
    for step in tqdm(range(steps), desc="MAP training"):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader)
            xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % max(200, steps//10) == 0:
            val_nll = eval_nll(model, val_loader, device)
            best_val = min(best_val, val_nll)
            print(f"step {step+1}: train_loss={loss.item():.4f}, val_nll={val_nll:.4f}")
    t = time.time() - start
    return best_val, t

# =========================================================
# 4) Subset helpers (head-only)
# =========================================================
def _flatten_tensors(tensors: List[torch.Tensor]) -> np.ndarray:
    if len(tensors) == 0: return np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        vecs = [t.detach().cpu().contiguous().view(-1) for t in tensors]
    return torch.cat(vecs).numpy().astype(np.float32)

def _assign_from_vec(tensors: List[torch.Tensor], vec: np.ndarray, device=None, dtype=None):
    p = 0
    for t in tensors:
        n = t.numel()
        src = torch.tensor(vec[p:p+n], device=device or t.device, dtype=dtype or t.dtype).view_as(t)
        with torch.no_grad():
            t.copy_(src)
        p += n
    assert p == len(vec)

def collect_param_subset(model: nn.Module, mode: str = "head") -> List[torch.Tensor]:
    tensors = []
    if mode in ("head", "head_ln"):
        tensors.append(model.head.weight)
    if mode == "head_ln":
        tensors += [model.ln.weight, model.ln.bias]
    return [t for t in tensors if t is not None]

def flatten_subset(model: nn.Module, mode: str = "head") -> np.ndarray:
    return _flatten_tensors(collect_param_subset(model, mode))

def set_subset_from_vec(model: nn.Module, vec: np.ndarray, mode: str = "head"):
    tensors = collect_param_subset(model, mode)
    if len(tensors) == 0: return
    device, dtype = tensors[0].device, tensors[0].dtype
    _assign_from_vec(tensors, vec, device=device, dtype=dtype)

def prior_scale_like(model: nn.Module, mode: str = "head", base_scale: float = 0.05) -> np.ndarray:
    vec = flatten_subset(model, mode)
    return np.full_like(vec, base_scale, dtype=np.float32)

# =========================================================
# 5) pGD–GMA (head-only)
# =========================================================
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(v, 1.0 / v.size)
    rho = np.nonzero(cond)[0][-1]
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(v, 1.0 / v.size)
    return w / s

def fixed_batch_iterator(loader: DataLoader):
    it = iter(loader)
    xb, yb = next(it)
    def get(): return xb, yb
    return get

@torch.no_grad()
def log_unnorm_p_subset_fixed(vec: np.ndarray, model: nn.Module, mode: str, get_batch,
                              prior_mean: np.ndarray, prior_std: np.ndarray,
                              beta: float = 1.0, device: str = "cuda") -> float:
    set_subset_from_vec(model, vec, mode)
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    ll = -F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum").item()
    z = (vec - prior_mean) / prior_std
    logp = -0.5 * float(np.dot(z, z)) - 0.5 * float(np.sum(np.log(2*np.pi*prior_std**2)))
    return beta * ll + logp

def gma_subset_pgd(model: nn.Module, train_loader: DataLoader,
                   prior_mean: np.ndarray, prior_std: np.ndarray, mode: str,
                   N: int = 200, M: int = 8, K: int = 200,
                   sigma2: float = 1e-3, eta0: float = 0.2,
                   device: str = "cuda", seed: int = 123):
    rng = np.random.default_rng(seed)
    d = prior_mean.size

    # 1) Component means and bank samples
    mu = prior_mean[None, :] + rng.normal(0, 0.01, size=(N, d)) * prior_std[None, :]
    flat = np.empty((N * M, d), dtype=np.float32)
    for i in range(N):
        s, t = i * M, (i + 1) * M
        flat[s:t] = rng.normal(mu[i], np.sqrt(sigma2), size=(M, d)).astype(np.float32)

    # 2) Precompute Gaussian PDF matrix P
    cst = -0.5 * (d * np.log(2 * np.pi * sigma2))
    x2 = np.einsum('ij,ij->i', flat, flat)
    m2 = np.einsum('ij,ij->i', mu, mu)
    xdotm = flat @ mu.T
    logN = (cst - 0.5 * (x2[:, None] + m2[None, :]) / sigma2 + xdotm / sigma2).astype(np.float64)
    logN = np.clip(logN, -745.0, 80.0)
    P = np.exp(logN, dtype=np.float64)  # (NM, N)

    # 3) Target logs on a fixed batch
    get_batch = fixed_batch_iterator(train_loader)
    logp = np.empty(N * M, dtype=np.float64)
    for r in tqdm(range(N * M), desc=f"precompute target logp (mode={mode})"):
        logp[r] = log_unnorm_p_subset_fixed(flat[r], model, mode, get_batch,
                                            prior_mean, prior_std, beta=1.0, device=device)
    mu_lp, sd_lp = float(logp.mean()), float(max(1e-8, logp.std()))
    logp_std = (logp - mu_lp) / sd_lp

    # 4) pGD loop
    w = np.full(N, 1.0 / N, dtype=np.float64)
    for k in tqdm(range(1, K + 1), desc="GMA pGD"):
        q = P @ w  # (NM,)
        g = np.empty(N, dtype=np.float64)
        for i in range(N):
            s, t = i * M, (i + 1) * M
            g[i] = 1.0 + (np.log(q[s:t] + 1e-300) - logp_std[s:t]).mean()
        v = w - (eta0 / k) * g
        w = project_to_simplex(v)

    # 5) Resample from bank with final weights
    comp = rng.choice(N, size=N * M, p=w, replace=True)
    rows = comp * M + rng.integers(0, M, size=N * M)
    samples = flat[rows]
    return samples.astype(np.float32), w.astype(np.float32)

# =========================================================
# 6) Optional generation
# =========================================================
@torch.no_grad()
def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or not (0.0 < top_p < 1.0): return probs
    dtype = probs.dtype
    probs = probs.float()
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = (cdf > top_p)
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    probs.zero_().scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
    probs_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs.div_(probs_sum)
    return probs.to(dtype)

def _sample_next(logits: torch.Tensor, temperature: float = 1.0,
                 top_k: int = None, top_p: float = None) -> torch.Tensor:
    if logits.dim() == 3: logits = logits[:, -1, :]
    logits = logits / max(1e-8, temperature)
    V = logits.size(-1)
    if top_k is not None and 1 <= top_k < V:
        v, _ = torch.topk(logits, top_k)
        thresh = v[:, [-1]]
        logits = torch.where(logits >= thresh, logits, torch.full_like(logits, float('-inf')))
    probs = torch.softmax(logits.float(), dim=-1)
    _apply_top_p(probs, top_p)
    bad = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) <= 1e-8)
    if bad.any(): probs[bad] = 1.0 / V
    next_tok = torch.multinomial(probs.detach().cpu(), num_samples=1).to(logits.device)
    return next_tok

@torch.no_grad()
def generate_map_tail(model: nn.Module, prompt: str, bpe: SimpleByteBPE,
                      max_new_tokens: int = 20, temperature: float = 0.7,
                      top_k: int = None, top_p: float = 0.9, device: str = "cuda") -> str:
    model.to(device); model.eval()
    ids0 = bpe.encode(prompt)
    if len(ids0) == 0: return ""
    x = torch.tensor(ids0[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    outs: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]
        next_tok = _sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        outs.append(int(next_tok.item()))
        x = torch.cat([x, next_tok], dim=1)[:, -model.ctx:]
    return bpe.decode(outs)

@torch.no_grad()
def generate_from_subset_tail(model: nn.Module, weights_vec: np.ndarray, mode: str,
                              prompt: str, bpe: SimpleByteBPE, max_new_tokens: int = 20,
                              temperature: float = 0.75, top_k: Optional[int] = None, top_p: float = 0.95,
                              device: str = "cuda") -> str:
    """Temporarily set a single subset (e.g. one GMA sample or mean) and generate."""
    model.to(device); model.eval()
    ids0 = bpe.encode(prompt)
    if len(ids0) == 0: return ""
    x = torch.tensor(ids0[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)

    tensors = collect_param_subset(model, mode)
    backup = [t.detach().clone() for t in tensors]
    try:
        set_subset_from_vec(model, weights_vec, mode)
        outs: List[int] = []
        for _ in range(max_new_tokens):
            logits = model(x)[:, -1, :]
            if top_k is not None and 1 <= top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                thresh = v[:, [-1]]
                logits = torch.where(logits >= thresh, logits, torch.full_like(logits, float('-inf')))
            logits = logits / max(1e-8, temperature)
            probs = torch.softmax(logits.float(), dim=-1)
            _apply_top_p(probs, top_p)
            bad = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) <= 1e-8)
            if bad.any(): probs[bad] = 1.0 / probs.size(-1)
            next_tok = torch.multinomial(probs.detach().cpu(), num_samples=1).to(x.device)
            outs.append(int(next_tok.item()))
            x = torch.cat([x, next_tok], dim=1)[:, -model.ctx:]
        return bpe.decode(outs)
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

@torch.no_grad()
def generate_bayes_subset_ensemble_tail(model: nn.Module, samples: np.ndarray, mode: str,
                                        prompt: str, bpe: SimpleByteBPE, max_new_tokens: int = 20,
                                        temperature: float = 0.75, top_k: Optional[int] = None, top_p: float = 0.95,
                                        ensemble_S: int = 20, device: str = "cuda") -> str:
    """Ensemble in probability space over several GMA samples each step."""
    model.to(device); model.eval()
    ids0 = bpe.encode(prompt)
    if len(ids0) == 0: return ""
    x = torch.tensor(ids0[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    outs: List[int] = []

    S = min(ensemble_S, samples.shape[0])
    idx = np.random.choice(samples.shape[0], S, replace=False)

    tensors = collect_param_subset(model, mode)
    backup = [t.detach().clone() for t in tensors]

    try:
        for _ in range(max_new_tokens):
            probs_accum = None
            for s in idx:
                set_subset_from_vec(model, samples[s], mode)
                logits = model(x)[:, -1, :]
                if top_k is not None and 1 <= top_k < logits.size(-1):
                    v, _ = torch.topk(logits, top_k)
                    thresh = v[:, [-1]]
                    logits = torch.where(logits >= thresh, logits, torch.full_like(logits, float('-inf')))
                logits = logits / max(1e-8, temperature)
                p = torch.softmax(logits.float(), dim=-1)
                _apply_top_p(p, top_p)
                probs_accum = p if probs_accum is None else (probs_accum + p)

            probs = (probs_accum / float(S)).contiguous()
            bad = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) <= 1e-8)
            if bad.any(): probs[bad] = 1.0 / probs.size(-1)

            next_tok = torch.multinomial(probs.detach().cpu(), num_samples=1).to(x.device)
            outs.append(int(next_tok.item()))
            x = torch.cat([x, next_tok], dim=1)[:, -model.ctx:]
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

    return bpe.decode(outs)

# =========================================================
# 7) Toy corpus, stream-grounded eval
# =========================================================
TOY_LINES = [
    "the quick brown fox jumps over the lazy dog .",
    "a big red cat sits on a mat .",
    "she sells sea shells by the sea shore .",
    "how much wood would a woodchuck chuck .",
    "peter piper picked a peck of pickled peppers .",
    "a good cook could cook good food .",
    "i saw susie sitting in a shoe shine shop .",
]

def build_toy_text(lines=TOY_LINES):
    return "\n".join(lines) + "\n"

def make_stream_token_pairs(text: str, tok: SimpleByteBPE, token_ctx: int = 16):
    """
    Build (context_text, gold_token_id) pairs FROM THE ENCODED STREAM of each line.
    Guarantees gold matches the model's target token, including cases like " the ".
    """
    pairs = []
    for line in text.splitlines():
        if not line: continue
        ids = tok.encode(line)
        for t in range(1, len(ids)):
            ctx_ids = ids[max(0, t - token_ctx):t]
            gold_id = ids[t]
            ctx_txt = tok.decode(ctx_ids)
            pairs.append((ctx_txt, gold_id))
    return pairs

def is_word_like(token_text: str) -> bool:
    """
    Keep tokens that look like a single English word-piece with optional leading space,
    no internal spaces, maybe a trailing punctuation.
    """
    if not token_text:
        return False
    if " " in token_text.strip():
        return False
    return re.fullmatch(r"[ ]?[A-Za-z]+[\.!?,]?", token_text) is not None or token_text in {".", ",", "!"}

# ---- Per-step metrics helpers (NEW: proper multi-class Brier + entropy) ----
def brier_from_probs_full(probs: torch.Tensor, gold_id: int) -> float:
    """
    Multi-class Brier: sum_v (p_v - o_v)^2, with o one-hot at gold_id.
    Efficient form: sum(p^2) - 2*p_gold + 1
    """
    p = probs.detach().cpu()
    p_gold = float(p[gold_id].clamp_min(1e-12))
    s2 = float((p * p).sum())
    return s2 - 2.0 * p_gold + 1.0

def entropy_from_probs(probs: torch.Tensor) -> float:
    p = probs.detach().cpu().float().clamp_min(1e-12)
    return float(-(p * p.log()).sum())

@torch.no_grad()
def probs_next_token_MAP(model, tok: SimpleByteBPE, ctx_text: str, device="cuda"):
    model.eval().to(device)
    ids = tok.encode(ctx_text)
    if not ids: return None
    x = torch.tensor(ids[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)[:, -1, :]
    probs = torch.softmax(logits.float(), dim=-1).squeeze(0).cpu()
    return probs  # (V,)

@torch.no_grad()
def probs_next_token_GMA(model, samples: np.ndarray, mode: str, tok: SimpleByteBPE,
                         ctx_text: str, S=32, device="cuda"):
    model.eval().to(device)
    ids = tok.encode(ctx_text)
    if not ids: return None
    x = torch.tensor(ids[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    tensors = collect_param_subset(model, mode)
    backup = [t.detach().clone() for t in tensors]
    idx = np.random.choice(samples.shape[0], min(S, samples.shape[0]), replace=False)
    try:
        acc = None
        for s in idx:
            set_subset_from_vec(model, samples[s], mode)
            logits = model(x)[:, -1, :]
            p = torch.softmax(logits.float(), dim=-1)
            acc = p if acc is None else (acc + p)
        probs = (acc / len(idx)).squeeze(0).cpu()
        return probs
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

def evaluate_pairs(model, tok, pairs, samples=None, mode="head", device="cuda", max_examples=6):
    n=0; correct_map=0; correct_gma=0
    nll_map=0.0; nll_gma=0.0
    brier_map=0.0; brier_gma=0.0
    ent_map=0.0; ent_gma=0.0
    examples=[]
    for ctx_txt, gold_id in pairs:
        pm = probs_next_token_MAP(model, tok, ctx_txt, device=device)
        if pm is None: continue
        pg = probs_next_token_GMA(model, samples, mode, tok, ctx_txt, S=32, device=device) if samples is not None else pm

        pred_map = int(pm.argmax()); pred_gma = int(pg.argmax())
        p_m = float(pm[gold_id].clamp_min(1e-12)); p_g = float(pg[gold_id].clamp_min(1e-12))

        correct_map += int(pred_map == gold_id)
        correct_gma += int(pred_gma == gold_id)
        nll_map += -math.log(p_m); nll_gma += -math.log(p_g)
        brier_map += brier_from_probs_full(pm, gold_id)
        brier_gma += brier_from_probs_full(pg, gold_id)
        ent_map += entropy_from_probs(pm)
        ent_gma += entropy_from_probs(pg)
        n += 1

        if len(examples) < max_examples:
            examples.append(dict(
                ctx=ctx_txt.replace("\n", "\\n"),
                gold_str=tok.decode([gold_id]).replace("\n", "\\n"),
                p_map=p_m, p_gma=p_g,
                H_map=entropy_from_probs(pm), H_gma=entropy_from_probs(pg),
                pred_map_str=tok.decode([pred_map]).replace("\n", "\\n"),
                pred_gma_str=tok.decode([pred_gma]).replace("\n", "\\n"),
            ))

    return {
        "count": n,
        "acc_map": correct_map / max(1,n),
        "acc_gma": correct_gma / max(1,n),
        "nll_map": nll_map / max(1,n),
        "nll_gma": nll_gma / max(1,n),
        "ppl_map": math.exp(nll_map / max(1,n)) if n > 0 else float('nan'),
        "ppl_gma": math.exp(nll_gma / max(1,n)) if n > 0 else float('nan'),
        "brier_map": brier_map / max(1,n),
        "brier_gma": brier_gma / max(1,n),
        "ent_map": ent_map / max(1,n),
        "ent_gma": ent_gma / max(1,n),
        "examples": examples
    }

# =========================================================
# 7b) NEW — Uncertainty visualizations
# =========================================================
@torch.no_grad()
def posterior_predictive_stats_for_context(model, samples, mode, bpe, ctx_text,
                                           S: int = 100, ci: float = 0.90, device: str = "cuda"):
    """
    For a given context, compute:
      - MAP probability vector (V,)
      - GMA posterior predictive: per-token mean and (1±ci)/2 percentile intervals across S samples
    """
    V = bpe.vocab_size
    # MAP probs
    p_map = probs_next_token_MAP(model, bpe, ctx_text, device=device)
    if p_map is None: return None

    # Sample predictive probs from GMA
    ids = bpe.encode(ctx_text)
    if not ids: return None
    x = torch.tensor(ids[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)

    tensors = collect_param_subset(model, mode)
    backup = [t.detach().clone() for t in tensors]

    S_eff = min(S, samples.shape[0])
    idx_s = np.random.choice(samples.shape[0], S_eff, replace=False)
    P = torch.zeros((S_eff, V), dtype=torch.float32, device="cpu")
    try:
        for i, s in enumerate(idx_s):
            set_subset_from_vec(model, samples[s], mode)
            logits = model(x)[:, -1, :]
            P[i] = torch.softmax(logits.float().squeeze(0).cpu(), dim=-1)
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

    p_mean = P.mean(dim=0).numpy()           # (V,)
    lo = np.percentile(P.numpy(), (1-ci)/2*100, axis=0)  # (V,)
    hi = np.percentile(P.numpy(), (1+ci)/2*100, axis=0)  # (V,)

    return {"p_map": p_map.numpy(), "p_mean": p_mean, "p_lo": lo, "p_hi": hi}

def topk_union_indices(p_map: np.ndarray, p_mean: np.ndarray, k: int = 20):
    """Take top-k tokens from MAP and GMA-mean and return the union (indices)."""
    top_map = np.argpartition(-p_map, min(k, len(p_map)-1))[:k]
    top_gma = np.argpartition(-p_mean, min(k, len(p_mean)-1))[:k]
    idx = np.unique(np.concatenate([top_map, top_gma]))
    # sort by GMA mean (descending) for nicer plotting
    idx = idx[np.argsort(-p_mean[idx])]
    return idx

def tokens_from_indices(bpe: SimpleByteBPE, idxs: np.ndarray):
    return [bpe.decode([int(i)]) for i in idxs]

def plot_token_distribution_with_ci(bpe: SimpleByteBPE, ctx_text: str,
                                    p_map: np.ndarray, p_mean: np.ndarray, p_lo: np.ndarray, p_hi: np.ndarray,
                                    k: int = 20, title: str = "Next-token distribution: MAP vs GMA"):
    idx = topk_union_indices(p_map, p_mean, k=k)
    labels = tokens_from_indices(bpe, idx)
    x = np.arange(len(idx))
    width = 0.45

    fig, ax = plt.subplots(figsize=(max(8, len(idx)*0.5), 4))
    # MAP bars
    ax.bar(x - width/2, p_map[idx], width=width, label="MAP")
    # GMA mean with error bars
    y = p_mean[idx]
    yerr = np.vstack([y - p_lo[idx], p_hi[idx] - y])
    ax.errorbar(x + width/2, y, yerr=yerr, fmt='o', label="GMA (mean ± CI)")

    ax.set_xticks(x)
    # Keep plain text to avoid weird glyph warnings
    ax.set_xticklabels([lbl.replace("\n","\\n") for lbl in labels], rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(title + f"\nContext: {repr(ctx_text)}")
    ax.legend()
    ax.grid(True, axis='y', linestyle=':')
    plt.tight_layout()
    plt.show()

def plot_entropy_shift_over_stream(model, bpe, pairs, samples, mode, device="cuda", N: int = 100):
    """
    Plot entropy (MAP, GMA) and their difference over the first N positions in the stream.
    """
    Ns = min(N, len(pairs))
    Hm, Hg = [], []
    for i in range(Ns):
        ctx, _ = pairs[i]
        pm = probs_next_token_MAP(model, bpe, ctx, device=device)
        pg = probs_next_token_GMA(model, samples, mode, bpe, ctx, S=32, device=device)
        Hm.append(entropy_from_probs(pm))
        Hg.append(entropy_from_probs(pg))
    Hm = np.array(Hm); Hg = np.array(Hg); d = Hg - Hm

    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(np.arange(Ns), Hm, label="MAP")
    ax[0].plot(np.arange(Ns), Hg, label="GMA")
    ax[0].set_ylabel("Entropy (nats)")
    ax[0].set_title("Entropy across stream tokens")
    ax[0].legend()
    ax[0].grid(True, linestyle=':')

    ax[1].plot(np.arange(Ns), d, label="ΔEntropy (GMA - MAP)")
    ax[1].axhline(0.0, linestyle='--')
    ax[1].set_xlabel("Token index in stream")
    ax[1].set_ylabel("ΔEntropy")
    ax[1].grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

def find_most_interesting_context_by_entropy_shift(model, bpe, pairs, samples, mode, device="cuda", scan_N: int = 200):
    """
    Scan first scan_N pairs and return the context with the largest absolute entropy shift.
    """
    scan_N = min(scan_N, len(pairs))
    best_i, best_delta, best_ctx = None, -1.0, None
    for i in range(scan_N):
        ctx, _ = pairs[i]
        pm = probs_next_token_MAP(model, bpe, ctx, device=device)
        pg = probs_next_token_GMA(model, samples, mode, bpe, ctx, S=32, device=device)
        dH = abs(entropy_from_probs(pg) - entropy_from_probs(pm))
        if dH > best_delta:
            best_delta, best_i, best_ctx = dH, i, ctx
    return best_ctx, best_delta, best_i

# =========================================================
# 8) Main
# =========================================================
torch.manual_seed(111); np.random.seed(111)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --- Toy corpus ---
text = build_toy_text(TOY_LINES)

# --- Train BPE freshly on the toy text (avoid stale merges) ---
BPE_PATH = Path("bpe_vocab_toy.npz")
bpe = SimpleByteBPE()
print("[BPE] Training byte-level BPE on toy text...")
bpe.fit(text, target_vocab=1024, max_merges=5000)
bpe.save(str(BPE_PATH))
print(f"[BPE] Trained & saved: vocab={bpe.vocab_size}")

# --- Data ---
ctx = 256
ds = SubwordDataset(text, bpe, ctx=ctx, stride=4)
n_train = int(0.8 * len(ds))
tr, va = torch.utils.data.random_split(
    ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(1)
)
train_loader = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(va, batch_size=64, shuffle=False)

# --- Model ---
model = TinyGPT(vocab_size=bpe.vocab_size, ctx=ctx, n_layer=4, n_head=8, d_model=256, d_ff=1024)
model.head.weight = model.tok.weight  # weight tying

# --- MAP train ---
best_val_nll, map_time = train_map(
    model, train_loader, val_loader,
    steps=3000, lr=3e-4, wd=0.05, device=device
)
print(f"[MAP]  val NLL: {best_val_nll:.4f} | PPL: {math.exp(best_val_nll):.2f} | Train time: {map_time:.1f}s")

# --- GMA (head) ---
MODE = "head"
theta_map = flatten_subset(model, MODE).astype(np.float32)
prior_std = prior_scale_like(model, MODE, base_scale=0.05)
gma_samples, w_final = gma_subset_pgd(
    model, train_loader,
    prior_mean=theta_map, prior_std=prior_std, mode=MODE,
    N=200, M=8, K=200, sigma2=1e-3, eta0=0.2,
    device=device, seed=123
)
print("[GMA] samples:", gma_samples.shape,
      "weight entropy:", -np.sum(w_final * np.log(np.maximum(w_final, 1e-12))).round(3))

# --- Build pairs FROM TOKEN STREAM ---
pairs_stream = make_stream_token_pairs(text, bpe, token_ctx=16)
print(f"[EVAL] stream token pairs: {len(pairs_stream)}")

# Full token-level eval (ALL TOKENS)
res_all = evaluate_pairs(model, bpe, pairs_stream, samples=gma_samples, mode=MODE, device=device)
print(f"[EVAL/TOKENS] n={res_all['count']} | "
      f"Acc MAP={res_all['acc_map']:.3f} vs GMA={res_all['acc_gma']:.3f} | "
      f"NLL MAP={res_all['nll_map']:.3f} vs GMA={res_all['nll_gma']:.3f} | "
      f"PPL MAP={res_all['ppl_map']:.2f} vs GMA={res_all['ppl_gma']:.2f} | "
      f"Brier MAP={res_all['brier_map']:.3f} vs GMA={res_all['brier_gma']:.3f} | "
      f"H (entropy) MAP={res_all['ent_map']:.3f} vs GMA={res_all['ent_gma']:.3f}")

# Word-like subset (closer to “next word”)
pairs_wordlike = [(c, gid) for (c, gid) in pairs_stream if is_word_like(bpe.decode([gid]))]
res_wl = evaluate_pairs(model, bpe, pairs_wordlike, samples=gma_samples, mode=MODE, device=device)
print(f"[EVAL/WORD-LIKE] n={res_wl['count']} | "
      f"Acc MAP={res_wl['acc_map']:.3f} vs GMA={res_wl['acc_gma']:.3f} | "
      f"NLL MAP={res_wl['nll_map']:.3f} vs GMA={res_wl['nll_gma']:.3f} | "
      f"PPL MAP={res_wl['ppl_map']:.2f} vs GMA={res_wl['ppl_gma']:.2f} | "
      f"Brier MAP={res_wl['brier_map']:.3f} vs GMA={res_wl['brier_gma']:.3f} | "
      f"H (entropy) MAP={res_wl['ent_map']:.3f} vs GMA={res_wl['ent_gma']:.3f}")

# --- Show a few qualitative examples ---
print("\nExamples (token-level):")
for ex in res_all["examples"]:
    print("\nCTX:", repr(ex["ctx"]))
    print("GOLD token:", repr(ex["gold_str"]))
    print("MAP:  p(gold)={:.3f}; H={:.3f}; pred={}".format(ex["p_map"], ex["H_map"], repr(ex["pred_map_str"])))
    print("GMA:  p(gold)={:.3f}; H={:.3f}; pred={}".format(ex["p_gma"], ex["H_gma"], repr(ex["pred_gma_str"])))

# --- Pick an interesting context and plot predictive distributions (+ CI) ---
ctx_best, dH, idx_best = find_most_interesting_context_by_entropy_shift(
    model, bpe, pairs_stream, gma_samples, MODE, device=device, scan_N=200
)
if ctx_best is None:
    ctx_best = pairs_stream[0][0]
    print("\n[WARN] No interesting context found; defaulting to the first context.")
else:
    print(f"\n[INFO] Chosen context idx={idx_best} with |ΔEntropy|≈{dH:.3f}")

stats = posterior_predictive_stats_for_context(
    model, gma_samples, MODE, bpe, ctx_best, S=128, ci=0.90, device=device
)
if stats is not None:
    plot_token_distribution_with_ci(
        bpe, ctx_best,
        p_map=stats["p_map"], p_mean=stats["p_mean"], p_lo=stats["p_lo"], p_hi=stats["p_hi"],
        k=20,
        title="Posterior predictive (GMA) vs MAP (90% CI)"
    )

# --- Entropy shift overview across the stream ---
plot_entropy_shift_over_stream(model, bpe, pairs_stream, gma_samples, MODE, device=device, N=150)

# --- Optional qualitative continuations (unchanged style) ---
def visible(s: str) -> str:
    """Make spaces/newlines visible so continuations aren't 'blank'."""
    return s.replace(" ", "·").replace("\n", "\\n")

max_new_tokens = 100
prompt = "the quick brown "
print("\n--- PROMPT (visible) ---\n" + visible(prompt))

print("\n--- MAP continuation (visible) ---\n" + visible(
    generate_map_tail(model, prompt, bpe,
                      max_new_tokens=max_new_tokens, temperature=0.75, top_k=None, top_p=0.95, device=device)
))

mean_vec = gma_samples.mean(axis=0)
# 1) GMA ensemble (prediction averaging across GMA samples each step)
print("\n--- GMA ensemble continuation (visible) ---\n" + visible(
    generate_bayes_subset_ensemble_tail(model, gma_samples, MODE, prompt, bpe,
                                        max_new_tokens=max_new_tokens, temperature=0.75, top_k=None, top_p=0.95, ensemble_S=200, device=device)
))
# 2) GMA mean-weights: average GMA weights into a single point estimate, then generate
print("\n--- GMA mean-weights continuation (visible) ---\n" + visible(
    generate_from_subset_tail(model, mean_vec, MODE, prompt, bpe,
                              max_new_tokens=max_new_tokens, temperature=0.75, top_k=None, top_p=0.95, device=device)
))
# 3) GMA single-sample continuation (one posterior draw)
idx_single = 123  # fixed choice for the paper
single_vec = gma_samples[idx_single]
print(f"\n--- GMA single-sample continuation (visible) [draw #{idx_single}] ---\n" + visible(
    generate_from_subset_tail(model, single_vec, MODE, prompt, bpe,
                              max_new_tokens=max_new_tokens, temperature=0.75, top_k=None, top_p=0.95, device=device)
))

"""# E2: Long-sentence/paragraph continuation."""

# -*- coding: utf-8 -*-
# E2: Long-form continuation & uncertainty propagation on a larger corpus
#
# Adds to E1:
#  - ECE (Expected Calibration Error) & risk–coverage/AURC
#  - Adaptive decoding (switch MAP <-> GMA based on entropy)
#  - Posterior-mean reranking for MAP top-k candidates
#  - BETTER DECODING hygiene: top-k, repetition penalty, no-repeat n-grams
#  - Slightly larger tokenizer (BPE 4k) + a bit longer training
#
# Works with: PyTorch >= 2.1, numpy, matplotlib

import os, re, math, time, numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import matplotlib.pyplot as plt

# =========================================================
# 0) Tiny byte-level BPE (same as E1, with save/load)
# =========================================================
class SimpleByteBPE:
    def __init__(self):
        self.vocab_size = 256
        self.ranks: Dict[Tuple[int,int], int] = {}
        self.n_symbols = 256  # next new symbol id

    def fit(self, text: str, target_vocab: int = 4096, max_merges: int = 12000):
        assert target_vocab > 256
        data = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).tolist()
        seq = data; merges = []; self.n_symbols = 256

        def count_pairs(sequence: List[int]) -> Dict[Tuple[int,int], int]:
            if not sequence: return {}
            counts = {}
            prev = sequence[0]
            for x in sequence[1:]:
                p = (prev, x)
                counts[p] = counts.get(p, 0) + 1
                prev = x
            return counts

        def merge_sequence(sequence: List[int], pair: Tuple[int,int], new_sym: int) -> List[int]:
            out = []; i = 0; a, b = pair; L = len(sequence)
            while i < L:
                if i < L - 1 and sequence[i] == a and sequence[i+1] == b:
                    out.append(new_sym); i += 2
                else:
                    out.append(sequence[i]); i += 1
            return out

        while self.n_symbols < target_vocab and len(merges) < max_merges and len(seq) >= 2:
            pair_counts = count_pairs(seq)
            if not pair_counts: break
            best_pair, best_cnt = None, 0
            for p, c in pair_counts.items():
                if c > best_cnt:
                    best_pair, best_cnt = p, c
            if best_pair is None or best_cnt < 2: break
            new_id = self.n_symbols
            seq = merge_sequence(seq, best_pair, new_id)
            merges.append(best_pair)
            self.ranks[best_pair] = len(merges) - 1
            self.n_symbols += 1
        self.vocab_size = self.n_symbols

    def _get_pairs(self, tokens: List[int]) -> Iterable[Tuple[int,int]]:
        for i in range(len(tokens) - 1):
            yield (tokens[i], tokens[i+1])

    def encode(self, text: str) -> List[int]:
        tokens = list(np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8))
        if not tokens: return []
        ranks = self.ranks
        if not hasattr(self, "_pair2id"):
            self._pair2id = {p: 256 + r for p, r in ranks.items()}
        while True:
            pairs = list(self._get_pairs(tokens))
            if not pairs: break
            pair_ranks = [(ranks[p], i) for i, p in enumerate(pairs) if p in ranks]
            if not pair_ranks: break
            best_rank = min(pr for pr, _ in pair_ranks)
            best_pairs = {pairs[i] for pr, i in pair_ranks if pr == best_rank}
            out = []; i = 0; L = len(tokens); merged_any = False
            while i < L:
                if i < L - 1 and (tokens[i], tokens[i+1]) in best_pairs:
                    new_id = self._pair2id[(tokens[i], tokens[i+1])]
                    out.append(new_id); i += 2; merged_any = True
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
            if not merged_any: break
        return tokens

    def decode(self, ids: List[int]) -> str:
        if not ids: return ""
        if not hasattr(self, "_id2pair"):
            self._id2pair = {256 + r: p for p, r in self.ranks.items()}
        tokens = list(ids)
        max_id = max(tokens) if tokens else -1
        for cur_id in range(max_id, 255, -1):
            if cur_id not in self._id2pair: continue
            a, b = self._id2pair[cur_id]
            out = []
            for t in tokens:
                if t == cur_id: out.extend([a, b])
                else: out.append(t)
            tokens = out
        return bytes(tokens).decode("utf-8", errors="ignore")

    def save(self, path: str):
        pairs = np.array(list(self.ranks.keys()), dtype=np.int32)
        if pairs.size == 0:
            a = np.empty((0,), dtype=np.int32); b = np.empty((0,), dtype=np.int32); r = np.empty((0,), dtype=np.int32)
        else:
            a = pairs[:,0]; b = pairs[:,1]
            r = np.array([self.ranks[(int(x), int(y))] for x, y in pairs], dtype=np.int32)
        np.savez(path, vocab_size=np.int32(self.vocab_size), n_symbols=np.int32(self.n_symbols), a=a, b=b, r=r)

    def load(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.vocab_size = int(z["vocab_size"]); self.n_symbols = int(z["n_symbols"])
        a = z["a"]; b = z["b"]; r = z["r"]; self.ranks = {}
        for i in range(len(r)):
            self.ranks[(int(a[i]), int(b[i]))] = int(r[i])
        for attr in ("_pair2id","_id2pair"):
            if hasattr(self, attr): delattr(self, attr)

# =========================================================
# 1) Model (slightly larger than E1)
# =========================================================
class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int = 6, n_head: int = 8,
                 d_model: int = 384, d_ff: int = 1536, ctx: int = 512):
        super().__init__()
        self.ctx = ctx
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device, dtype=torch.long)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(h, mask=mask)
        h = self.ln(h)
        return self.head(h)  # (B,T,V)

# =========================================================
# 2) Dataset utilities (stream-grounded)
# =========================================================
class SubwordDataset(Dataset):
    def __init__(self, text: str, bpe: SimpleByteBPE, ctx: int = 512, stride: int = 1):
        self.ctx = ctx
        ids = bpe.encode(text)
        toks = torch.tensor(ids, dtype=torch.long)
        X, Y = [], []
        for i in range(0, max(0, len(toks) - ctx), max(1, stride)):
            X.append(toks[i:i+ctx]); Y.append(toks[i+1:i+ctx+1])
        if len(X) == 0:
            nrep = max(2*ctx, 8192) // max(1, len(toks))
            toks = toks.repeat(nrep)
            for i in range(0, len(toks) - ctx, max(1, stride)):
                X.append(toks[i:i+ctx]); Y.append(toks[i+1:i+ctx+1])
        self.x = torch.stack(X); self.y = torch.stack(Y)
        self.vocab = bpe.vocab_size

    def __len__(self): return self.x.size(0)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def build_pairs_from_stream(text: str, tok: SimpleByteBPE, token_ctx: int = 32):
    pairs = []
    for line in text.splitlines():
        if not line: continue
        ids = tok.encode(line)
        for t in range(1, len(ids)):
            ctx_ids = ids[max(0, t - token_ctx):t]
            gold_id = ids[t]
            ctx_txt = tok.decode(ctx_ids)
            pairs.append((ctx_txt, gold_id))
    return pairs

# =========================================================
# 3) Training/Eval helpers + head subset ops (same ideas as E1)
# =========================================================
@torch.no_grad()
def eval_nll(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    was_train = model.training; model.eval()
    tot_loss = 0.0; n_tok = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum")
        tot_loss += loss.item(); n_tok += yb.numel()
    if was_train: model.train()
    return tot_loss / max(1, n_tok)

def train_map(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              steps: int = 5000, lr: float = 3e-4, wd: float = 0.05, device: str = "cuda"):
    model.to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    best_val = float("inf"); it = iter(train_loader); start = time.time()
    for step in tqdm(range(steps), desc="MAP training"):
        try: xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(500, steps//20) == 0:
            val_nll = eval_nll(model, val_loader, device)
            best_val = min(best_val, val_nll)
            print(f"step {step+1}: train_loss={loss.item():.4f}, val_nll={val_nll:.4f}")
    return best_val, time.time() - start

def _flatten_tensors(tensors: List[torch.Tensor]) -> np.ndarray:
    if len(tensors) == 0: return np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        vecs = [t.detach().cpu().contiguous().view(-1) for t in tensors]
    return torch.cat(vecs).numpy().astype(np.float32)

def _assign_from_vec(tensors: List[torch.Tensor], vec: np.ndarray, device=None, dtype=None):
    p = 0
    for t in tensors:
        n = t.numel()
        src = torch.tensor(vec[p:p+n], device=device or t.device, dtype=dtype or t.dtype).view_as(t)
        with torch.no_grad(): t.copy_(src)
        p += n
    assert p == len(vec)

def collect_param_subset(model: nn.Module, mode: str = "head") -> List[torch.Tensor]:
    tensors = []
    if mode in ("head","head_ln"): tensors.append(model.head.weight)
    if mode == "head_ln": tensors += [model.ln.weight, model.ln.bias]
    return [t for t in tensors if t is not None]

def flatten_subset(model: nn.Module, mode: str = "head") -> np.ndarray:
    return _flatten_tensors(collect_param_subset(model, mode))

def set_subset_from_vec(model: nn.Module, vec: np.ndarray, mode: str = "head"):
    tensors = collect_param_subset(model, mode)
    if len(tensors) == 0: return
    device, dtype = tensors[0].device, tensors[0].dtype
    _assign_from_vec(tensors, vec, device=device, dtype=dtype)

def prior_scale_like(model: nn.Module, mode: str = "head", base_scale: float = 0.05) -> np.ndarray:
    return np.full_like(flatten_subset(model, mode), base_scale, dtype=np.float32)

# =========================================================
# 4) pGD–GMA (same algorithm as E1, parameters tunable)
# =========================================================
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]; cssv = np.cumsum(u) - 1; ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond): return np.full_like(v, 1.0 / v.size)
    rho = np.nonzero(cond)[0][-1]; theta = cssv[rho] / float(rho + 1)
    w = np.maximum(v - theta, 0.0); s = w.sum()
    if not np.isfinite(s) or s <= 0: return np.full_like(v, 1.0 / v.size)
    return w / s

def fixed_batch_iterator(loader: DataLoader):
    it = iter(loader); xb, yb = next(it)
    def get(): return xb, yb
    return get

@torch.no_grad()
def log_unnorm_p_subset_fixed(vec: np.ndarray, model: nn.Module, mode: str, get_batch,
                              prior_mean: np.ndarray, prior_std: np.ndarray,
                              beta: float = 1.0, device: str = "cuda") -> float:
    set_subset_from_vec(model, vec, mode)
    xb, yb = get_batch(); xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    ll = -F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum").item()
    z = (vec - prior_mean) / prior_std
    logp = -0.5 * float(np.dot(z, z)) - 0.5 * float(np.sum(np.log(2*np.pi*prior_std**2)))
    return beta * ll + logp

def gma_subset_pgd(model: nn.Module, train_loader: DataLoader,
                   prior_mean: np.ndarray, prior_std: np.ndarray, mode: str,
                   N: int = 200, M: int = 8, K: int = 200,
                   sigma2: float = 1e-3, eta0: float = 0.2,
                   device: str = "cuda", seed: int = 123):
    rng = np.random.default_rng(seed); d = prior_mean.size
    mu = prior_mean[None, :] + rng.normal(0, 0.01, size=(N, d)) * prior_std[None, :]
    flat = np.empty((N*M, d), dtype=np.float32)
    for i in range(N):
        s, t = i*M, (i+1)*M
        flat[s:t] = rng.normal(mu[i], np.sqrt(sigma2), size=(M, d)).astype(np.float32)
    cst = -0.5 * (d * np.log(2 * np.pi * sigma2))
    x2 = np.einsum('ij,ij->i', flat, flat)
    m2 = np.einsum('ij,ij->i', mu, mu)
    xdotm = flat @ mu.T
    logN = (cst - 0.5 * (x2[:, None] + m2[None, :]) / sigma2 + xdotm / sigma2).astype(np.float64)
    logN = np.clip(logN, -745.0, 80.0)
    P = np.exp(logN, dtype=np.float64)
    get_batch = fixed_batch_iterator(train_loader)
    logp = np.empty(N*M, dtype=np.float64)
    for r in tqdm(range(N*M), desc=f"precompute target logp (mode={mode})"):
        logp[r] = log_unnorm_p_subset_fixed(flat[r], model, mode, get_batch,
                                            prior_mean, prior_std, beta=1.0, device=device)
    mu_lp, sd_lp = float(logp.mean()), float(max(1e-8, logp.std()))
    logp_std = (logp - mu_lp) / sd_lp
    w = np.full(N, 1.0 / N, dtype=np.float64)
    for k in tqdm(range(1, K+1), desc="GMA pGD"):
        q = P @ w
        g = np.empty(N, dtype=np.float64)
        for i in range(N):
            s, t = i*M, (i+1)*M
            g[i] = 1.0 + (np.log(q[s:t] + 1e-300) - logp_std[s:t]).mean()
        v = w - (eta0 / k) * g
        w = project_to_simplex(v)
    comp = rng.choice(N, size=N*M, p=w, replace=True)
    rows = comp * M + rng.integers(0, M, size=N*M)
    samples = flat[rows]
    return samples.astype(np.float32), w.astype(np.float32)

# =========================================================
# 5) Probabilities, entropy, ECE, Risk–Coverage/AURC
# =========================================================
def entropy_from_probs(probs: torch.Tensor) -> float:
    p = probs.detach().cpu().float().clamp_min(1e-12)
    return float(-(p * p.log()).sum())

@torch.no_grad()
def probs_next_token_MAP(model, tok: SimpleByteBPE, ctx_text: str, device="cuda"):
    model.eval().to(device)
    ids = tok.encode(ctx_text)
    if not ids: return None
    x = torch.tensor(ids[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)[:, -1, :]
    return torch.softmax(logits.float(), dim=-1).squeeze(0).cpu()

@torch.no_grad()
def probs_next_token_GMA(model, samples: np.ndarray, mode: str, tok: SimpleByteBPE,
                         ctx_text: str, S=32, device="cuda"):
    model.eval().to(device)
    ids = tok.encode(ctx_text)
    if not ids: return None
    x = torch.tensor(ids[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    tensors = collect_param_subset(model, mode); backup = [t.detach().clone() for t in tensors]
    idx = np.random.choice(samples.shape[0], min(S, samples.shape[0]), replace=False)
    try:
        acc = None
        for s in idx:
            set_subset_from_vec(model, samples[s], mode)
            logits = model(x)[:, -1, :]
            p = torch.softmax(logits.float(), dim=-1)
            acc = p if acc is None else (acc + p)
        return (acc / len(idx)).squeeze(0).cpu()
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

def collect_confidence_correct_over_pairs(model, bpe, pairs, samples=None, mode="head",
                                          device="cuda", S_ensemble: int = 32):
    """Return confidences and correctness (0/1) per token for MAP and (optionally) GMA."""
    conf_map, corr_map = [], []
    conf_gma, corr_gma = [], []
    for ctx_txt, gold_id in pairs:
        pm = probs_next_token_MAP(model, bpe, ctx_txt, device=device)
        if pm is None: continue
        pred_m = int(pm.argmax()); conf_m = float(pm.max()); corr_m = int(pred_m == gold_id)
        conf_map.append(conf_m); corr_map.append(corr_m)
        if samples is not None:
            pg = probs_next_token_GMA(model, samples, mode, bpe, ctx_txt, S=S_ensemble, device=device)
            pred_g = int(pg.argmax()); conf_g = float(pg.max()); corr_g = int(pred_g == gold_id)
            conf_gma.append(conf_g); corr_gma.append(corr_g)
    return (np.array(conf_map), np.array(corr_map)), (np.array(conf_gma), np.array(corr_gma)) if samples is not None else None

def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 20) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1); ece = 0.0; n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = np.where((confidences >= lo) & (confidences < hi))[0]
        if idx.size == 0: continue
        acc = float(correctness[idx].mean()); conf = float(confidences[idx].mean())
        ece += (idx.size / n) * abs(acc - conf)
    return ece

def risk_coverage_curve(confidences: np.ndarray, errors01: np.ndarray):
    idx = np.argsort(-confidences)
    e_sorted = errors01[idx]
    coverages = np.arange(1, len(e_sorted)+1, dtype=np.float64) / len(e_sorted)
    risks = np.cumsum(e_sorted) / np.arange(1, len(e_sorted)+1)
    return coverages, risks

def aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    # NumPy >= 2.0 prefers trapezoid
    return float(np.trapezoid(risks, coverages))

# =========================================================
# 6) Decoding hygiene: top-k, repetition penalty, no-repeat n-grams
# =========================================================
def _apply_top_k_logits(logits: torch.Tensor, top_k: Optional[int]):
    if top_k is None: return logits
    V = logits.size(-1)
    if 1 <= top_k < V:
        v, _ = torch.topk(logits, top_k)
        thresh = v[..., [-1]]
        logits = torch.where(logits >= thresh, logits, torch.full_like(logits, float('-inf')))
    return logits

def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or not (0.0 < top_p < 1.0): return probs
    dtype = probs.dtype; probs = probs.float()
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = (cdf > top_p)
    mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    probs.zero_().scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
    probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    return probs.to(dtype)

def _apply_repetition_penalty(probs: torch.Tensor, recent_ids: List[int], penalty: float = 1.1):
    if penalty is None or penalty <= 1.0 or len(recent_ids) == 0: return probs
    with torch.no_grad():
        uniq = list(set(int(i) for i in recent_ids))
        probs[..., uniq] /= penalty
        probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    return probs

def _ban_repeating_ngrams(probs: torch.Tensor, hist: List[int], n: int = 3):
    if n is None or n < 2 or len(hist) < n - 1: return probs
    prefix = tuple(hist[-(n-1):])
    banned = []
    for i in range(len(hist) - n + 1):
        if tuple(hist[i:i+n-1]) == prefix:
            banned.append(int(hist[i+n-1]))
    if banned:
        probs[..., banned] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    return probs

def _postprocess_probs(probs: torch.Tensor,
                       hist: List[int],
                       top_p: Optional[float],
                       top_k: Optional[int],
                       rep_penalty: Optional[float],
                       no_repeat_ngram: Optional[int],
                       recent_window: int = 64):
    _apply_top_p(probs, top_p)
    probs = _apply_repetition_penalty(probs, hist[-recent_window:], penalty=rep_penalty)

    pre_ngram = probs.clone()
    probs = _ban_repeating_ngrams(probs, hist, n=no_repeat_ngram)

    # If banning nuked all mass (or produced NaNs), revert to pre-ban
    if (not torch.isfinite(probs).all()) or float(probs.sum()) <= 1e-12:
        probs = pre_ngram

    # final normalize just in case
    probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    return probs

# =========================================================
# 7) Generation modes: MAP, GMA ensemble, GMA mean-weight, GMA single-sample, Adaptive
# =========================================================
@torch.no_grad()
def generate_map_tail(model, prompt, bpe,
                      max_new_tokens=200,
                      temperature=0.7,
                      top_k=50,
                      top_p=0.90,
                      repetition_penalty=1.1,
                      no_repeat_ngram=3,
                      recent_window=64,
                      device="cuda"):
    model.to(device).eval()
    ids0 = bpe.encode(prompt)
    if not ids0: return ""
    x = torch.tensor(ids0[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    hist = list(ids0[-model.ctx:])
    outs = []
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]
        logits = _apply_top_k_logits(logits, top_k)
        probs  = torch.softmax((logits / max(1e-8, temperature)).float(), dim=-1)
        probs  = _postprocess_probs(probs, hist, top_p, top_k, repetition_penalty, no_repeat_ngram, recent_window)
        next_tok = torch.multinomial(probs.detach().cpu(), 1).to(x.device)
        t = int(next_tok.item())
        outs.append(t); hist.append(t)
        x = torch.cat([x, next_tok], dim=1)[:, -model.ctx:]
    return bpe.decode(outs)

@torch.no_grad()
def generate_bayes_subset_ensemble_tail(model, samples, mode, prompt, bpe,
                                        max_new_tokens=200,
                                        temperature=0.7,
                                        top_k=50,
                                        top_p=0.90,
                                        repetition_penalty=1.1,
                                        no_repeat_ngram=3,
                                        recent_window=64,
                                        ensemble_S=32,
                                        device="cuda"):
    model.to(device).eval()
    ids0 = bpe.encode(prompt)
    if not ids0: return ""
    x = torch.tensor(ids0[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    hist = list(ids0[-model.ctx:])
    outs = []
    S = min(ensemble_S, samples.shape[0])
    idx = np.random.choice(samples.shape[0], S, replace=False)
    tensors = collect_param_subset(model, mode); backup = [t.detach().clone() for t in tensors]
    try:
        for _ in range(max_new_tokens):
            probs_accum = None
            for s in idx:
                set_subset_from_vec(model, samples[s], mode)
                logits = model(x)[:, -1, :]
                logits = _apply_top_k_logits(logits, top_k)
                p = torch.softmax((logits / max(1e-8, temperature)).float(), dim=-1)
                _apply_top_p(p, top_p)
                probs_accum = p if probs_accum is None else (probs_accum + p)
            probs = (probs_accum / float(S)).contiguous()
            probs = _postprocess_probs(probs, hist, top_p, top_k, repetition_penalty, no_repeat_ngram, recent_window)
            if torch.isnan(probs).any() or probs.sum() <= 1e-8: probs[:] = 1.0 / probs.size(-1)
            next_tok = torch.multinomial(probs.detach().cpu(), 1).to(x.device)
            t = int(next_tok.item())
            outs.append(t); hist.append(t)
            x = torch.cat([x, next_tok], dim=1)[:, -model.ctx:]
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup), device=tensors[0].device, dtype=tensors[0].dtype)
    return bpe.decode(outs)

@torch.no_grad()
def generate_from_subset_tail(model, weights_vec, mode, prompt, bpe,
                              max_new_tokens=200,
                              temperature=0.7,
                              top_k=50,
                              top_p=0.90,
                              repetition_penalty=1.1,
                              no_repeat_ngram=3,
                              recent_window=64,
                              device="cuda"):
    model.to(device).eval()
    ids0 = bpe.encode(prompt)
    if not ids0: return ""
    x = torch.tensor(ids0[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)
    hist = list(ids0[-model.ctx:])
    tensors = collect_param_subset(model, mode); backup = [t.detach().clone() for t in tensors]
    try:
        set_subset_from_vec(model, weights_vec, mode)
        outs = []
        for _ in range(max_new_tokens):
            logits = model(x)[:, -1, :]
            logits = _apply_top_k_logits(logits, top_k)
            probs  = torch.softmax((logits / max(1e-8, temperature)).float(), dim=-1)
            probs  = _postprocess_probs(probs, hist, top_p, top_k, repetition_penalty, no_repeat_ngram, recent_window)
            next_tok = torch.multinomial(probs.detach().cpu(), 1).to(x.device)
            t = int(next_tok.item())
            outs.append(t); hist.append(t)
            x = torch.cat([x, next_tok], dim=1)[:, -model.ctx:]
        return bpe.decode(outs)
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup), device=tensors[0].device, dtype=tensors[0].dtype)

@torch.no_grad()
def generate_adaptive_tail(model, samples, mode, prompt, bpe,
                           max_new_tokens=200, H_thresh=0.6, k_steps=5,
                           temperature=0.7,
                           top_k=50,
                           top_p=0.90,
                           repetition_penalty=1.1,
                           no_repeat_ngram=3,
                           recent_window=64,
                           device="cuda", S_ensemble=32):
    """
    MAP by default; when MAP entropy >= H_thresh, switch to GMA-ensemble for the next k_steps.
    """
    model.to(device).eval()
    ids0 = bpe.encode(prompt)
    if not ids0:
        return ""
    x = torch.tensor(ids0[-model.ctx:], dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
    hist = list(ids0[-model.ctx:])
    outs = []
    step_mode = "MAP"
    countdown = 0

    tensors = collect_param_subset(model, mode)
    backup = [t.detach().clone() for t in tensors]

    try:
        for _ in range(max_new_tokens):
            # --- compute probs (may be 1D or 2D depending on branch) ---
            if step_mode == "MAP":
                logits = model(x)[:, -1, :]                               # (1,V)
                logits = _apply_top_k_logits(logits, top_k)
                probs = torch.softmax((logits / max(1e-8, temperature)).float(), dim=-1)  # (1,V)
                H = entropy_from_probs(probs)
                if H >= H_thresh:
                    step_mode, countdown = "GMA", k_steps
            else:  # step_mode == "GMA"
                ctx_text = bpe.decode(x[0].tolist())
                probs = probs_next_token_GMA(                               # returns (V,)
                    model, samples, mode, bpe, ctx_text, S=S_ensemble, device=device
                )
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)                              # -> (1,V)
                countdown -= 1
                if countdown <= 0:
                    step_mode = "MAP"

            # --- hygiene & safety ---
            probs = _postprocess_probs(
                probs, hist, top_p, top_k, repetition_penalty, no_repeat_ngram, recent_window
            )
            # final safety fallback (if constraints zeroed everything)
            if (not torch.isfinite(probs).all()) or float(probs.sum()) <= 1e-12:
                probs = torch.full_like(probs, 1.0 / probs.size(-1))

            # --- sample and append ---
            next_tok = torch.multinomial(probs.detach().cpu(), 1).to(x.device)  # (1,1)
            t = int(next_tok.item())
            outs.append(t)
            hist.append(t)
            x = torch.cat([x, next_tok], dim=1)[:, -model.ctx:]
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

    return bpe.decode(outs)

# =========================================================
# 8) Reranking with posterior mean (GMA) for MAP top-k
# =========================================================
def rerank_with_gma(model, samples, mode, bpe, ctx_text, k=10, S=64, device="cuda"):
    pm = probs_next_token_MAP(model, bpe, ctx_text, device=device)
    topk_ids = torch.topk(pm, k).indices.tolist()
    pg = probs_next_token_GMA(model, samples, mode, bpe, ctx_text, S=S, device=device)
    scores_map = [float(pm[i]) for i in topk_ids]
    scores_gma = [float(pg[i]) for i in topk_ids]
    toks = [bpe.decode([i]) for i in topk_ids]
    order_map = np.argsort(-np.array(scores_map))
    order_gma = np.argsort(-np.array(scores_gma))
    return {
        "candidates": toks,
        "map_probs": [scores_map[i] for i in order_map],
        "gma_probs": [scores_gma[i] for i in order_gma],
        "rank_map": order_map.tolist(),
        "rank_gma": order_gma.tolist(),
    }

# =========================================================
# 9) Data loading for E2 (online with caching)
# =========================================================
corpus_urls = [
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Adventures of Sherlock Holmes
    "https://www.gutenberg.org/files/834/834-0.txt",    # Memoirs of Sherlock Holmes
    "https://www.gutenberg.org/files/2852/2852-0.txt",  # Hound of the Baskervilles
    "https://www.gutenberg.org/files/244/244-0.txt",    # A Study in Scarlet
    "https://www.gutenberg.org/files/2097/2097-0.txt",  # The Sign of the Four
]
cache_path = "holmes_bundle.txt"

def _strip_html(s: str) -> str:
    import html, re
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p>", "\n\n", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_gutenberg_boilerplate(s: str) -> str:
    import re
    s = re.sub(r"(?is)^.*?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\n", "", s)
    s = re.sub(r"(?is)\n*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", "", s)
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def load_text_from_urls(urls: List[str],
                        cache_path: str = "corpus_cache.txt",
                        min_chars: int = 10_000,
                        timeout: int = 20) -> str:
    """
    Download and concatenate pages; cache to disk. If cache exists and is large
    enough, use it. Strips simple HTML and Gutenberg boilerplate.
    """
    dest = Path(cache_path)
    if dest.exists():
        txt = dest.read_text(encoding="utf-8", errors="ignore")
        if len(txt) >= min_chars:
            print(f"Loaded cached corpus: {dest.name}, {len(txt):,} chars")
            return txt
        else:
            print(f"Cached corpus too small ({len(txt)} chars); re-downloading.")

    parts = []
    try:
        import requests
        for url in urls:
            print(f"[FETCH] {url}")
            r = requests.get(url, timeout=timeout); r.raise_for_status()
            body = r.text
            looks_html = ("<html" in body.lower()) or ("<p" in body.lower()) or ("<br" in body.lower())
            if looks_html:
                body = _strip_html(body)
            body = strip_gutenberg_boilerplate(body)
            parts.append(body)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")

    text = "\n\n".join([p for p in parts if p]).strip()
    if len(text) < min_chars:
        raise RuntimeError(f"Downloaded corpus too small ({len(text)} chars). "
                           f"Provide more/longer URLs or lower min_chars.")
    dest.write_text(text, encoding="utf-8")
    print(f"Saved corpus to {dest.resolve()} ({len(text):,} chars)")
    return text

def load_corpus(corpus_paths: List[str]) -> str:
    texts = []
    for p in corpus_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(strip_gutenberg_boilerplate(f.read()))
    return "\n\n".join(texts)

# =========================================================
# 10) Main: end-to-end E2 run
# =========================================================
torch.manual_seed(222); np.random.seed(222)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- Prefer online bundle (with cache); fallback to local paths if you want ----
use_online = True
if use_online:
    text_full = load_text_from_urls(corpus_urls, cache_path=cache_path, min_chars=200_000)
else:
    # If you have local files instead:
    corpus_paths = []  # fill if needed
    text_full = load_corpus(corpus_paths)

# Light punctuation normalization helps byte-level models a bit
text_full = (text_full
             .replace("“", '"').replace("”", '"')
             .replace("’", "'").replace("—", "--"))

print(f"[DATA] Corpus chars: {len(text_full):,}")

# ---- Tokenizer ----
BPE_PATH = Path("bpe_vocab_e2.npz")
bpe = SimpleByteBPE()
print("[BPE] Training byte-level BPE on corpus...")
bpe.fit(text_full, target_vocab=4096, max_merges=12000)
bpe.save(str(BPE_PATH))
print(f"[BPE] Trained & saved: vocab={bpe.vocab_size}")

# ---- Dataset ----
ctx = 512
stride = 8
ds = SubwordDataset(text_full, bpe, ctx=ctx, stride=stride)
n_total = len(ds)
n_train = int(0.98 * n_total)  # plenty of data; small held-out
n_val = n_total - n_train
tr, va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
val_loader   = DataLoader(va, batch_size=64, shuffle=False, num_workers=0)
print(f"[DATA] Train iters: {len(train_loader)}, Val iters: {len(val_loader)}")

# ---- Model ----
model = TinyGPT(vocab_size=bpe.vocab_size, ctx=ctx, n_layer=6, n_head=8, d_model=384, d_ff=1536)
model.head.weight = model.tok.weight  # weight tying

# ---- MAP training ----
steps = 30000
best_val_nll, map_time = train_map(model, train_loader, val_loader, steps=steps, lr=3e-4, wd=0.05, device=device)
print(f"[MAP]  val NLL: {best_val_nll:.4f} | PPL: {math.exp(best_val_nll):.2f} | Train time: {map_time/60:.1f} min")

# ---- GMA (head-only) ----
MODE = "head"
theta_map = flatten_subset(model, MODE).astype(np.float32)
prior_std = prior_scale_like(model, MODE, base_scale=0.05)
gma_samples, w_final = gma_subset_pgd(
    model, train_loader,
    prior_mean=theta_map, prior_std=prior_std, mode=MODE,
    N=200, M=30, K=1000, sigma2=1e-3, eta0=0.2,
    device=device, seed=123
)
print("[GMA] samples:", gma_samples.shape,
      "weight entropy:", -np.sum(w_final * np.log(np.maximum(w_final, 1e-12))).round(3))
# ---- bar plot of top 10 final weights ----
top_idx = np.argsort(w_final)[::-1][:10]  # sort descending
plt.figure(figsize=(6, 4))
plt.bar(range(10), w_final[top_idx])
plt.xticks(range(10), [f"{i}" for i in top_idx], rotation=45)
plt.ylabel("Final weight")
plt.xlabel("Component index")
plt.title("Top 10 GMA Component Weights")
plt.tight_layout()
plt.show()

# ---- Held-out evaluation pairs for calibration (sampled) ----
heldout_text = text_full[-min(len(text_full)//10, 300_000):]  # final slice as a pseudo-VAL set
pairs_eval = build_pairs_from_stream(heldout_text, bpe, token_ctx=128)
np.random.shuffle(pairs_eval)
pairs_eval = pairs_eval[:5000]  # subsample for speed
print(f"[EVAL] #pairs for ECE/AURC: {len(pairs_eval)}")

# ---- ECE / Risk–Coverage / AURC ----
(conf_m, corr_m), gma_pack = collect_confidence_correct_over_pairs(
    model, bpe, pairs_eval, samples=gma_samples, mode=MODE, device=device, S_ensemble=128
)
ece_map = compute_ece(conf_m, corr_m, n_bins=20)
cov_m, risk_m = risk_coverage_curve(conf_m, 1 - corr_m)
aurc_map = aurc(cov_m, risk_m)

if gma_pack is not None:
    conf_g, corr_g = gma_pack
    ece_gma = compute_ece(conf_g, corr_g, n_bins=20)
    cov_g, risk_g = risk_coverage_curve(conf_g, 1 - corr_g)
    aurc_gma = aurc(cov_g, risk_g)
    print(f"[CAL] ECE: MAP={ece_map:.4f} | GMA={ece_gma:.4f}")
    print(f"[CAL] AURC: MAP={aurc_map:.4f} | GMA={aurc_gma:.4f}")
else:
    print(f"[CAL] ECE: MAP={ece_map:.4f}; AURC={aurc_map:.4f}")

# ---- Long-form generation modes (with improved decoding) ----
def visible(s: str) -> str: return s.replace(" ", "·").replace("\n", "\\n")
prompt = "Holmes looked at me and smiled. "
max_new_tokens = 150

gen_kwargs = dict(temperature=0.7, top_k=50, top_p=0.90,
                  repetition_penalty=1.1, no_repeat_ngram=3,
                  recent_window=128, device=device)

out_map = generate_map_tail(model, prompt, bpe, max_new_tokens=max_new_tokens, **gen_kwargs)
out_ens = generate_bayes_subset_ensemble_tail(model, gma_samples, MODE, prompt, bpe,
                                              max_new_tokens=max_new_tokens, ensemble_S=128, **gen_kwargs)
mean_vec = gma_samples.mean(axis=0)
out_mean = generate_from_subset_tail(model, mean_vec, MODE, prompt, bpe,
                                     max_new_tokens=max_new_tokens, **gen_kwargs)
# Single-sample draw (e.g., row 123 if exists)
s_idx = min(123, gma_samples.shape[0]-1)
out_single = generate_from_subset_tail(model, gma_samples[s_idx], MODE, prompt, bpe,
                                       max_new_tokens=max_new_tokens, **gen_kwargs)

out_adapt = generate_adaptive_tail(model, gma_samples, MODE, prompt, bpe,
                                   max_new_tokens=max_new_tokens, H_thresh=0.6, k_steps=5,
                                   S_ensemble=128, **gen_kwargs)

print("\n--- MAP continuation (visible) ---\n", visible(out_map))
print("\n--- GMA ensemble (S=128) continuation (visible) ---\n", visible(out_ens))
print("\n--- GMA mean-weight continuation (visible) ---\n", visible(out_mean))
print(f"\n--- GMA single-sample continuation (visible) [draw #{s_idx}] ---\n", visible(out_single))
print("\n--- Adaptive continuation (visible) ---\n", visible(out_adapt))

# ---- Posterior-mean reranking demo on a context ----
demo_ctx = "We arrived at Baker Street where we found"
rer = rerank_with_gma(model, gma_samples, MODE, bpe, demo_ctx, k=10, S=256, device=device)
print("\n[RERANK] Context:", repr(demo_ctx))
print("[RERANK] Candidates (MAP order):")
for i, idx in enumerate(rer["rank_map"]):
    tok = rer["candidates"][idx]; p = rer["map_probs"][i]
    print(f"  MAP#{i+1}: {repr(tok)}  p={p:.4f}")
print("[RERANK] Candidates (GMA order):")
for i, idx in enumerate(rer["rank_gma"]):
    tok = rer["candidates"][idx]; p = rer["gma_probs"][i]
    print(f"  GMA#{i+1}: {repr(tok)}  p={p:.4f}")

# ---- Optional: quick plots (ECE histogram & Risk–Coverage) ----
try:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(conf_m, bins=20, alpha=0.6, label=f"MAP (ECE={ece_map:.3f})", density=True)
    if gma_pack is not None:
        ax.hist(conf_g, bins=20, alpha=0.6, label=f"GMA (ECE={ece_gma:.3f})", density=True)
    ax.set_title("Confidence histogram")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Density"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(cov_m, risk_m, label=f"MAP (AURC={aurc_map:.3f})")
    if gma_pack is not None:
        ax.plot(cov_g, risk_g, label=f"GMA (AURC={aurc_gma:.3f})")
    ax.set_title("Risk–Coverage"); ax.set_xlabel("Coverage"); ax.set_ylabel("Risk")
    ax.grid(alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()
except Exception as e:
    print("[PLOT] Skipped plotting due to:", e)

# =========================================================
# 11) Token-level metrics on a prompt continuation (E2)
#      - Teacher-force on a reference continuation (MAP by default)
#      - Metrics: Accuracy, NLL, Perplexity, Brier for each predictive mode
# =========================================================
def _brier_multiclass(probs: torch.Tensor, true_idx: int) -> float:
    # Brier = sum_k (p_k - y_k)^2, with y true one-hot
    p = probs.detach().cpu().float().clamp_min(0).clamp_max(1)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        p = torch.full_like(p, 1.0 / p.numel())
    else:
        p = p / s
    y = torch.zeros_like(p); y[true_idx] = 1.0
    return float(torch.sum((p - y) ** 2))

@torch.no_grad()
def _probs_next_token_MEANWEIGHT(model, mean_vec: np.ndarray, mode: str,
                                 tok: SimpleByteBPE, ctx_text: str, device="cuda"):
    tensors = collect_param_subset(model, mode); backup = [t.detach().clone() for t in tensors]
    try:
        set_subset_from_vec(model, mean_vec, mode)
        return probs_next_token_MAP(model, tok, ctx_text, device=device)
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

@torch.no_grad()
def _probs_next_token_SINGLE(model, sample_vec: np.ndarray, mode: str,
                             tok: SimpleByteBPE, ctx_text: str, device="cuda"):
    tensors = collect_param_subset(model, mode); backup = [t.detach().clone() for t in tensors]
    try:
        set_subset_from_vec(model, sample_vec, mode)
        return probs_next_token_MAP(model, tok, ctx_text, device=device)
    finally:
        _assign_from_vec(tensors, _flatten_tensors(backup),
                         device=tensors[0].device, dtype=tensors[0].dtype)

def evaluate_prompt_continuation_metrics(model, bpe, prompt_text: str,
                                         reference_continuation: str,
                                         samples: np.ndarray,
                                         mode: str = "head",
                                         device: str = "cuda",
                                         S_ensemble: int = 128,
                                         single_idx: int = 123):
    """
    Teacher-force on a fixed 'reference_continuation'. At each step:
      - build the current context (prompt + reference[:t])
      - get per-mode next-token probs
      - score the true next token (from reference)
    Returns a dict of metrics per mode.
    """
    prompt_ids = bpe.encode(prompt_text)
    ref_ids    = bpe.encode(reference_continuation)
    if len(ref_ids) == 0:
        raise ValueError("reference_continuation encodes to 0 tokens.")

    def _iter_steps():
        ctx_ids = list(prompt_ids)
        for t in range(len(ref_ids)):
            gold = ref_ids[t]
            ctx_text = bpe.decode(ctx_ids[-model.ctx:])
            yield ctx_text, gold
            ctx_ids.append(gold)

    stats = {
        "MAP":    {"n":0, "acc":0, "nll":0.0, "brier":0.0},
        "ENSEMBLE": {"n":0, "acc":0, "nll":0.0, "brier":0.0},
        "MEAN":   {"n":0, "acc":0, "nll":0.0, "brier":0.0},
        "SINGLE": {"n":0, "acc":0, "nll":0.0, "brier":0.0},
    }

    mean_vec = samples.mean(axis=0)
    s_idx = min(single_idx, samples.shape[0]-1)

    for ctx_text, gold in _iter_steps():
        # MAP
        pm = probs_next_token_MAP(model, bpe, ctx_text, device=device)
        if pm is None: continue
        p_true = float(pm[gold].clamp_min(1e-12))
        stats["MAP"]["n"]    += 1
        stats["MAP"]["acc"]  += int(int(pm.argmax()) == gold)
        stats["MAP"]["nll"]  += -math.log(p_true)
        stats["MAP"]["brier"]+= _brier_multiclass(pm, gold)

        # ENSEMBLE
        pg = probs_next_token_GMA(model, samples, mode, bpe, ctx_text, S=S_ensemble, device=device)
        p_true = float(pg[gold].clamp_min(1e-12))
        stats["ENSEMBLE"]["n"]    += 1
        stats["ENSEMBLE"]["acc"]  += int(int(pg.argmax()) == gold)
        stats["ENSEMBLE"]["nll"]  += -math.log(p_true)
        stats["ENSEMBLE"]["brier"]+= _brier_multiclass(pg, gold)

        # MEAN-WEIGHT
        pmean = _probs_next_token_MEANWEIGHT(model, mean_vec, mode, bpe, ctx_text, device=device)
        p_true = float(pmean[gold].clamp_min(1e-12))
        stats["MEAN"]["n"]    += 1
        stats["MEAN"]["acc"]  += int(int(pmean.argmax()) == gold)
        stats["MEAN"]["nll"]  += -math.log(p_true)
        stats["MEAN"]["brier"]+= _brier_multiclass(pmean, gold)

        # SINGLE-SAMPLE
        psing = _probs_next_token_SINGLE(model, samples[s_idx], mode, bpe, ctx_text, device=device)
        p_true = float(psing[gold].clamp_min(1e-12))
        stats["SINGLE"]["n"]    += 1
        stats["SINGLE"]["acc"]  += int(int(psing.argmax()) == gold)
        stats["SINGLE"]["nll"]  += -math.log(p_true)
        stats["SINGLE"]["brier"]+= _brier_multiclass(psing, gold)

    def _finalize(rec):
        n = max(1, rec["n"])
        acc  = rec["acc"] / n
        nll  = rec["nll"] / n
        ppl  = math.exp(nll)
        brier= rec["brier"] / n
        return acc, nll, ppl, brier

    results = {
        "MAP":        _finalize(stats["MAP"]),
        "GMA ens":    _finalize(stats["ENSEMBLE"]),
        "GMA mean":   _finalize(stats["MEAN"]),
        "GMA single": _finalize(stats["SINGLE"]),
    }
    return results

# --- Run metrics on the MAP continuation as reference ---
reference = out_map  # switch to out_ens if you prefer to score on the ensemble continuation
metrics = evaluate_prompt_continuation_metrics(
    model, bpe, prompt, reference, gma_samples, mode=MODE,
    device=device, S_ensemble=128, single_idx=s_idx
)

# Pretty print
print("\n[TOKEN METRICS on prompt continuation]")
print("{:<14}  {:>9}  {:>9}  {:>11}  {:>9}".format("Method","Accuracy","NLL","Perplexity","Brier"))
for name, (acc, nll, ppl, brier) in metrics.items():
    print("{:<14}  {:>9.4f}  {:>9.4f}  {:>11.4f}  {:>9.4f}".format(name, acc, nll, ppl, brier))

# Emit LaTeX table (paste into your paper)
def _fmt(x): return f"{x:.4f}"
latex_lines = []
latex_lines.append("\\begin{table}[ht]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Token-level evaluation metrics for TinyGPT under different predictive modes. Metrics are defined in Appendix.~\\ref{app:metrics}.}")
latex_lines.append("\\label{tab:tinygpt_metrics}")
latex_lines.append("\\begin{tabular}{lcccc}")
latex_lines.append("\\toprule")
latex_lines.append("\\textbf{Method} & \\textbf{Accuracy} $\\uparrow$ & \\textbf{NLL} $\\downarrow$ & \\textbf{Perplexity} $\\downarrow$ & \\textbf{Brier} $\\downarrow$ \\\\")
latex_lines.append("\\midrule")
latex_lines.append(f"MAP (deterministic)     & {_fmt(metrics['MAP'][0])} & {_fmt(metrics['MAP'][1])} & {_fmt(metrics['MAP'][2])} & {_fmt(metrics['MAP'][3])} \\\\")
latex_lines.append(f"GMA ensemble            & {_fmt(metrics['GMA ens'][0])} & {_fmt(metrics['GMA ens'][1])} & {_fmt(metrics['GMA ens'][2])} & {_fmt(metrics['GMA ens'][3])} \\\\")
latex_lines.append(f"GMA mean-weight         & {_fmt(metrics['GMA mean'][0])} & {_fmt(metrics['GMA mean'][1])} & {_fmt(metrics['GMA mean'][2])} & {_fmt(metrics['GMA mean'][3])} \\\\")
latex_lines.append(f"GMA single-sample       & {_fmt(metrics['GMA single'][0])} & {_fmt(metrics['GMA single'][1])} & {_fmt(metrics['GMA single'][2])} & {_fmt(metrics['GMA single'][3])} \\\\")
latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")
print("\n".join(latex_lines))

# Notes: the continuation metrics are not useful because:
# “Teacher-forced on the MAP continuation” means we fix a reference token sequence (here, the text produced by the MAP sampling decode with temp/top-k/top-p) and then, for each step t, we condition on the reference prefix and score the next token under different prediction modes (MAP, WGMA ensemble, mean-weight, single-sample). It’s a self-evaluation on a sampled reference, not ground-truth text.
# Because the reference was sampled (not greedy argmax), the MAP mode’s argmax at step t often differs from the sampled token. Combined with exposure-bias effects (conditioning on a sampled, possibly low-probability prefix) and re-scoring with raw softmax (no sampling filters), the MAP column’s Accuracy < 1 and NLL/Brier are non-trivial.
# If, instead, the reference were built by greedy argmax at every step from the same logits, then re-scoring would make the MAP accuracy 1.0 (up to ties). Here, that’s intentionally not the setup—we use the sampled MAP continuation to compare how modes behave on the same fixed sequence.
# Note that when metrics are teacher-forced on a \emph{MAP-sampled} continuation, per-prompt NLL/Brier for the ensemble can be slightly worse than MAP (a known protocol bias toward the sequence generator), while held-out calibration still improves.

"""# end."""
