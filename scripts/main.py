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
# EDIT HERE ONLY
# =========================================================


# Put the datasets here
JSON_PATHS = [
    ""
]

# Put the proxy model here
MODEL_ID  = ""
MODEL_DTYPE = torch.bfloat16

# ---- Black-box setting hyperparameters ----
X_TAIL = 7         # bottom 7%
RENYI_Q = 2.0      # Renyi alpha
WZ = 0.8           # fusion weight




MAX_LENGTH = 512
BOS = True
CHUNK_SIZE = 32
TAIL_DROP = 4
EPS = 1e-12
GLOBAL_SEED = 42

# ---- Feature choice (single setting, no grid search) ----



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

def compute_auc(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, str]:
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


# =========================================================
# Core: extract ONE feature (NO SAMPLING)
# =========================================================
@torch.no_grad()
def extract_feature_one_text(text: str, tokenizer, model) -> float:
    """
    Return fused feature F for one text:
      - compute per-token logP and Renyi entropy
      - tail select bottom X_TAIL% positions by logP
      - drop lowest TAIL_DROP inside tail
      - mean logP tail -> P_norm
      - mean Renyi tail -> H_norm
      - F = WZ*P_norm + (1-WZ)*(1-H_norm)
    """
    if not text or not isinstance(text, str):
        return float("nan")

    input_ids = encode_raw_text_bos_noeos(tokenizer, text, MAX_LENGTH, BOS)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)

    if int(input_ids.shape[1]) < 2:
        return float("nan")

    out_model = model(input_ids=input_ids, use_cache=False)
    logits = out_model.logits[0, :-1, :]     # (T, V)
    targets = input_ids[0, 1:]               # (T,)
    T = int(logits.shape[0])
    V = int(logits.shape[1])
    lnV = float(np.log(max(2, V)))

    # per-position observed logP and Renyi entropy
    logp_obs = np.empty((T,), dtype=np.float32)
    H_renyi = np.empty((T,), dtype=np.float32)

    for start in range(0, T, CHUNK_SIZE):
        end = min(T, start + CHUNK_SIZE)
        chunk_logits = logits[start:end].to(torch.float32)  # (c, V)
        c = int(chunk_logits.shape[0])

        logp = torch.log_softmax(chunk_logits, dim=-1)      # (c, V)

        # observed token logP
        tgt = targets[start:end]
        ar = torch.arange(c, device=device)
        logp_tok = logp[ar, tgt]
        logp_obs[start:end] = logp_tok.detach().cpu().numpy().astype(np.float32)

        # Renyi entropy (single q)
        log_sum = torch.logsumexp((RENYI_Q * logp), dim=-1)  # (c,)
        Hq = log_sum / (1.0 - RENYI_Q)
        H_renyi[start:end] = Hq.detach().cpu().numpy().astype(np.float32)

        del chunk_logits, logp, tgt, ar, logp_tok, log_sum, Hq

    obs = logp_obs.astype(np.float64)  # (T,)

    # tail size
    k = int(np.ceil((X_TAIL / 100.0) * T))
    k = max(1, min(k, T))

    # indices of k smallest logP
    tail_idx_k = np.argpartition(obs, k - 1)[:k]

    # drop lowest tokens within tail (keep >=1)
    d = int(min(TAIL_DROP, k - 1))
    if d > 0:
        rm_local = np.argpartition(obs[tail_idx_k], d - 1)[:d]
        mask = np.ones(k, dtype=bool)
        mask[rm_local] = False
        tail_idx = tail_idx_k[mask]
    else:
        tail_idx = tail_idx_k

    # mean logP on remaining tail
    mean_logp_tail = float(np.mean(obs[tail_idx]))

    # normalize logP mean to ~[0,1]
    P_norm = (mean_logp_tail + lnV) / (lnV + EPS)
    P_norm = float(np.clip(P_norm, 0.0, 1.0))

    # mean Renyi entropy on remaining tail, normalized
    mean_Hq_tail = float(np.mean(H_renyi[tail_idx].astype(np.float64)))
    H_norm = mean_Hq_tail / (lnV + EPS)
    H_norm = float(np.clip(H_norm, 0.0, 1.0))

    # fusion
    F = WZ * P_norm + (1.0 - WZ) * (1.0 - H_norm)
    return float(F)


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

    for item in tqdm(data, desc=f"Processing {os.path.basename(json_path)} (F_es{X_TAIL}_rq{q_tag(RENYI_Q)}_wz{q_tag(WZ)})"):
        for key, label in [("original_text", 0), ("ai_generated_text", 1)]:
            feat = extract_feature_one_text(item.get(key, ""), tokenizer, model)
            xs.append(feat)
            ys.append(label)

    # AUROC
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.int64)
    auc_raw, auc_best, direction = compute_auc(x, y)

    # Statistics
    mask = ~np.isnan(x)
    n_total = len(x)
    n_valid = int(mask.sum())
    mean_h = float(np.nanmean(x[y == 0])) if np.any(y == 0) else float("nan")
    mean_a = float(np.nanmean(x[y == 1])) if np.any(y == 1) else float("nan")

    # Output
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
    # seed
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

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