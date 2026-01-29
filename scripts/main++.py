# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# =========================================================
# Hyperparameters (EDIT HERE ONLY)
# =========================================================

JSON_PATHS = [
    ""
]

# Put the proxy model here
MODEL_ID  = ""
MODEL_DTYPE = torch.bfloat16

# ---- black-box setting hyperparameter ----
X_TAIL = 13
RENYI_Q = 1.6
WZ = 0.1

MAX_LENGTH = 512
BOS = True

CHUNK_SIZE = 32
EPS = 1e-12

# sampling discrepancy for z-score
N_SAMPLES = 100
SAMPLE_SEED = 42
TAIL_DROP = 0


# =========================================================
# Utils
# =========================================================
def q_tag(x: float) -> str:
    return str(x).replace(".", "p")


def encode_raw_text_bos_noeos(tokenizer, text: str, max_length: int, bos: bool) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    if len(ids) > max_length:
        ids = ids[:max_length]
    return torch.tensor([ids], dtype=torch.long)


# =========================================================
# ES helper (mean of sorted[drop:k])
# =========================================================
def es_bottom_k_sorted(prefix: np.ndarray, k: int, drop: int = 0) -> float:
    """
    mean of sorted[drop : k] using prefix sums (sorted ascending).
    prefix is cumsum(sorted).
    """
    d = int(min(max(drop, 0), k - 1))  # at least keep 1 token
    denom = max(1, k - d)
    if d == 0:
        return float(prefix[k - 1] / denom)
    return float((prefix[k - 1] - prefix[d - 1]) / denom)


def es_bottom_k_sorted_batch(prefix: np.ndarray, k: int, drop: int = 0) -> np.ndarray:
    d = int(min(max(drop, 0), k - 1))
    denom = max(1, k - d)
    if d == 0:
        return prefix[:, k - 1] / denom
    return (prefix[:, k - 1] - prefix[:, d - 1]) / denom


# =========================================================
# Core feature extraction (only tail=13 + q=1.5)
# =========================================================
@torch.no_grad()
def extract_base_for_best_feature(
    text: str,
    tokenizer,
    model,
) -> Dict[str, float]:
    """
    Compute:
      - S_es13_raw, S_es13 (z-score)
      - H_renyi1p5_es13_norm
    """
    def nan_output() -> Dict[str, float]:
        return {
            "T": 0.0,
            "S_es13_raw": np.nan,
            "S_es13": np.nan,
            "H_renyi1p5_es13_norm": np.nan,
        }

    if not text or not isinstance(text, str):
        return nan_output()

    input_ids = encode_raw_text_bos_noeos(tokenizer, text, MAX_LENGTH, BOS)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)

    if int(input_ids.shape[1]) < 2:
        return nan_output()

    out_model = model(input_ids=input_ids, use_cache=False)
    logits = out_model.logits[0, :-1, :]     # (T, V)
    targets = input_ids[0, 1:]               # (T,)
    T = int(logits.shape[0])
    V = int(logits.shape[1])
    lnV = float(np.log(max(2, V)))  # for normalization

    # observed scores and sampled scores (LOG-PROB)
    S_obs = np.empty((T,), dtype=np.float32)
    S_samp = np.empty((N_SAMPLES, T), dtype=np.float32)

    # per-position Renyi entropy (only q=1.5)
    H_renyi = np.empty((T,), dtype=np.float32)

    for start in range(0, T, CHUNK_SIZE):
        end = min(T, start + CHUNK_SIZE)
        chunk_logits = logits[start:end].to(torch.float32)      # (c, V)
        c = int(chunk_logits.shape[0])

        logp = torch.log_softmax(chunk_logits, dim=-1)          # (c, V)

        # ---- Renyi entropy per position ----
        log_sum = torch.logsumexp((RENYI_Q * logp), dim=-1)     # (c,)
        Hq = log_sum / (1.0 - RENYI_Q)                          # (c,)
        H_renyi[start:end] = Hq.detach().cpu().numpy().astype(np.float32)

        # ---- observed token LOG-PROB ----
        tgt = targets[start:end]                                # (c,)
        ar = torch.arange(c, device=device)
        logp_tok = logp[ar, tgt]                                # (c,)
        S_obs[start:end] = logp_tok.detach().cpu().numpy().astype(np.float32)

        # ---- conditional-independent sampling LOG-PROB for z-score ----
        samp_ids = torch.distributions.Categorical(logits=chunk_logits).sample((N_SAMPLES,))  # (N, c)
        logp_samp = logp.gather(1, samp_ids.T).T                                              # (N, c)
        S_samp[:, start:end] = logp_samp.detach().cpu().numpy().astype(np.float32)

        del chunk_logits, logp, log_sum, Hq, tgt, ar, logp_tok, samp_ids, logp_samp

    # ---- ES@13% + tail Renyi entropy ----
    result: Dict[str, float] = {"T": float(T)}

    obs = S_obs.astype(np.float64)     # (T,)
    samp = S_samp.astype(np.float64)   # (N, T)

    obs_sorted = np.sort(obs)          # ascending
    obs_prefix = np.cumsum(obs_sorted)

    samp_sorted = np.sort(samp, axis=1)
    samp_prefix = np.cumsum(samp_sorted, axis=1)

    k = int(np.ceil((X_TAIL / 100.0) * T))
    k = max(1, min(k, T))

    # ES raw (drop lowest inside the X% tail)
    raw_es = es_bottom_k_sorted(obs_prefix, k, TAIL_DROP)

    # ES sample vec (for z-score, same drop rule)
    samp_es_vec = es_bottom_k_sorted_batch(samp_prefix, k, TAIL_DROP)

    mu = float(np.mean(samp_es_vec))
    sd = float(np.std(samp_es_vec))
    z = (raw_es - mu) / (sd + EPS)

    result["S_es13_raw"] = float(raw_es)
    result["S_es13"] = float(z)

    # ---- tail positions defined by smallest obs scores, then drop lowest inside that tail ----
    tail_idx_k = np.argpartition(obs, k - 1)[:k]
    d = int(min(TAIL_DROP, k - 1))
    if d > 0:
        rm_local = np.argpartition(obs[tail_idx_k], d - 1)[:d]
        mask = np.ones(k, dtype=bool)
        mask[rm_local] = False
        tail_idx = tail_idx_k[mask]
    else:
        tail_idx = tail_idx_k

    Hq_tail = float(np.mean(H_renyi[tail_idx].astype(np.float64)))
    result["H_renyi1p5_es13_norm"] = float(Hq_tail / (lnV + EPS))

    return result


def best_feature_from_base(base: Dict[str, float]) -> float:
    """
    F_es13_rq1p5_wz0p1 = wz*z + (1-wz)*(1 - Hq_norm)
    """
    z = float(base.get("S_es13", np.nan))
    Hq = float(base.get("H_renyi1p5_es13_norm", np.nan))
    return float(WZ * z + (1.0 - WZ) * (1.0 - Hq))


def compute_auc(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, str]:
    """
    Returns:
      auc_raw(ai=1), auc_best(max(auc,1-auc)), direction
    """
    mask = ~np.isnan(x)
    x2 = x[mask]
    y2 = y[mask].astype(int)

    if len(np.unique(y2)) < 2 or np.nanstd(x2) < 1e-12:
        return float("nan"), float("nan"), "na/constant"

    auc = float(roc_auc_score(y2, x2))
    if auc >= 0.5:
        return auc, auc, "ai higher"
    else:
        return auc, 1.0 - auc, "human higher (invert)"


def process_single_json(json_path: str, tokenizer, model) -> None:

    if not os.path.exists(json_path):
        print(f"\n==================== FILE NOT FOUND ====================")
        print(f"JSON file does not exist: {json_path}")
        print(f"=========================================================\n")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n==================== LOAD JSON ERROR ====================")
        print(f"Failed to load JSON file: {json_path}")
        print(f"Error message: {repr(e)}")
        print(f"=========================================================\n")
        return

    xs: List[float] = []
    ys: List[int] = []

    for item in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):
        for key, label in [("original_text", 0), ("ai_generated_text", 1)]:
            text = item.get(key, "")
            base = extract_base_for_best_feature(text, tokenizer, model)
            feat = best_feature_from_base(base)
            xs.append(feat)
            ys.append(label)

    # AUROC
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.int64)
    auc_raw, auc_best, direction = compute_auc(x, y)

    mask = ~np.isnan(x)
    n_total = len(x)
    n_valid = int(mask.sum())
    mean_h = float(np.nanmean(x[y == 0])) if np.any(y == 0) else float("nan")
    mean_a = float(np.nanmean(x[y == 1])) if np.any(y == 1) else float("nan")

    print(f"\n==================== RESULT FOR {os.path.basename(json_path)} ====================")
    print(f"JSON:   {json_path}")
    print(f"Model:  {MODEL_ID}")
    print(f"Feat:   F_es{X_TAIL}_rq{q_tag(RENYI_Q)}_wz{q_tag(WZ)}   (drop={TAIL_DROP})")
    print(f"Total samples: {n_total}  |  Valid(non-nan): {n_valid}")
    print(f"Mean feature: human={mean_h:.6f}, ai={mean_a:.6f}")
    print(f"AUROC raw (ai=1):  {auc_raw:.6f}")
    print(f"AUROC best:        {auc_best:.6f}   ({direction})")
    print(f"===============================================================\n")


# =========================================================
# Main
# =========================================================
def main():
    torch.manual_seed(SAMPLE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SAMPLE_SEED)
    np.random.seed(SAMPLE_SEED)

    print(f"Initializing tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=MODEL_DTYPE,
        device_map="auto",
    )
    model.eval()
    print("Tokenizer and model initialization completed.\n")

    for json_path in JSON_PATHS:
        process_single_json(json_path, tokenizer, model)

    print("All JSON files processing completed!")


if __name__ == "__main__":
    main()