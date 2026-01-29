# -*- coding: utf-8 -*-

import os
import json
import csv
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# =========================================================
# Paths / Model
# =========================================================
MODEL_ID = ""
DATA_DIR = "../datasets/main"
OUTPUT_ROOT = "../outputs/main++_black_reddit_hyper_experiment"
HP_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "hp_analysis_reddit_es")
CURVE_DIR = os.path.join(HP_OUTPUT_ROOT, "hyperparam_curves")

MODEL_DTYPE = torch.bfloat16

MAX_LENGTH = 512
BOS = True
CHUNK_SIZE = 32
EPS = 1e-12
TAIL_DROP = 0

# seed
SAMPLE_SEED = 42

# =========================================================
# Experiment configs (your requests)
# =========================================================
# default
X_DEFAULT = 15
Q_DEFAULT = 1.6
WZ_DEFAULT = 0.1
N_DEFAULT = 100

# analysis 1
A1_Q = 1.6
A1_WZ = 0.1
A1_N = 100
A1_X_LIST = list(range(1, 51))  # 1..50

# analysis 2 (skip 1.0)
A2_X = 15
A2_WZ = 0.1
A2_N = 100
A2_Q_LIST = [round(x, 1) for x in np.arange(0.5, 2.0 + 1e-9, 0.1) if abs(x - 1.0) > 1e-9]

# analysis 3
A3_X = 15
A3_Q = 1.6
A3_N = 100
A3_WZ_LIST = [round(x, 1) for x in np.arange(0.1, 0.9 + 1e-9, 0.1)]

# analysis 4
A4_X = 15
A4_Q = 1.6
A4_WZ = 0.1
A4_N_LIST = [10, 100, 1000, 10000]


# =========================================================
# Target datasets (exact list you gave)
# =========================================================
TARGET_FILES = [
    "reddit_bloom_7b.raw_data.json",
    "reddit_falcon_7b.raw_data.json",
    "reddit_gemma_7b.raw_data.json",
    "reddit_gpt2_xl.raw_data.json",
    "reddit_gptneo_2.7b.raw_data.json",
    "reddit_llama1_13b.raw_data.json",
    "reddit_llama2_13b.raw_data.json",
    "reddit_llama3_8b.raw_data.json",
    "reddit_opt_13b.raw_data.json",
    "reddit_opt_2.7b.raw_data.json",
    "reddit_phi2.raw_data.json",
    "reddit_gpt4turbo.raw_data.json",
]


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_id(s) -> str:
    s = str(s).replace(os.sep, "_").replace(" ", "_")
    return s[:200]

def dataset_name_from_path(p: str) -> str:
    base = os.path.basename(p)
    return os.path.splitext(base)[0]

def encode_raw_text_bos_noeos(tokenizer, text: str, max_length: int, bos: bool) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    if len(ids) > max_length:
        ids = ids[:max_length]
    return torch.tensor([ids], dtype=torch.long)

def q_tag(q: float) -> str:
    return str(q).replace(".", "p")

def compute_auc_from_scores(y: np.ndarray, s: np.ndarray) -> float:
    s = s.astype(float)
    y = y.astype(int)
    mask = ~np.isnan(s)
    s2 = s[mask]
    y2 = y[mask]
    if len(np.unique(y2)) < 2:
        return float("nan")
    if np.nanstd(s2) < 1e-12:
        return float("nan")
    auc = float(roc_auc_score(y2, s2))
    return auc if auc >= 0.5 else (1.0 - auc)


# =========================================================
# Smooth plotting (avg curve only)
# =========================================================
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y_pad, kernel, mode="valid")

def smooth_curve(x: np.ndarray, y: np.ndarray, num_points: int = 500, ma_window: int = 9):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x2 = x[mask].astype(float)
    y2 = y[mask].astype(float)
    if x2.size < 2:
        return x2, y2

    order = np.argsort(x2)
    x2, y2 = x2[order], y2[order]

    # dedup x by averaging y
    ux, inv = np.unique(x2, return_inverse=True)
    uy = np.zeros_like(ux, dtype=float)
    cnt = np.zeros_like(ux, dtype=float)
    for i, j in enumerate(inv):
        uy[j] += y2[i]
        cnt[j] += 1.0
    uy = uy / np.maximum(cnt, 1.0)

    if ux.size < 2:
        return ux, uy

    xd = np.linspace(float(ux.min()), float(ux.max()), num_points)
    yd = np.interp(xd, ux, uy)
    yd = moving_average(yd, ma_window)
    return xd, yd

def plot_avg_smooth(avg_csv: str, out_png: str, title: str, xlabel: str, ylabel: str) -> None:
    df = pd.read_csv(avg_csv)
    # x col is the non-y col
    if "auc_best_mean" not in df.columns:
        raise ValueError(f"avg csv missing auc_best_mean: {avg_csv}")
    x_cols = [c for c in df.columns if c != "auc_best_mean"]
    if len(x_cols) != 1:
        raise ValueError(f"avg csv should have exactly 1 x col: {avg_csv}, got {x_cols}")
    x_col = x_cols[0]

    x = pd.to_numeric(df[x_col], errors="coerce").values.astype(float)
    y = pd.to_numeric(df["auc_best_mean"], errors="coerce").values.astype(float)
    xs, ys = smooth_curve(x, y, num_points=500, ma_window=9)

    plt.figure()
    plt.plot(xs, ys, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================================================
# Cache helpers
# =========================================================
def meta_payload(cache_version: str, n_samples: int, x_list: List[int], q_list: List[float]) -> Dict[str, np.ndarray]:
    return {
        "version": np.array(cache_version),
        "meta_n_samples": np.array(int(n_samples), dtype=np.int64),
        "meta_max_length": np.array(int(MAX_LENGTH), dtype=np.int64),
        "meta_bos": np.array(int(1 if BOS else 0), dtype=np.int64),
        "meta_x_list": np.array(x_list, dtype=np.int64),
        "meta_q_list": np.array(q_list, dtype=np.float32),
        "meta_tail_drop": np.array(int(TAIL_DROP), dtype=np.int64),
    }

def _np_str(x) -> str:
    try:
        if isinstance(x, np.ndarray):
            if x.shape == ():
                return str(x.item())
            if x.size == 1:
                return str(x.reshape(-1)[0].item())
        return str(x)
    except Exception:
        return str(x)

def load_cache_npz(cache_path: str, cache_version: str, n_samples: int, x_list: List[int], q_list: List[float]) -> Optional[Dict[str, object]]:
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)

        if "version" not in data.files or _np_str(data["version"]) != cache_version:
            return None
        if "meta_n_samples" not in data.files or int(data["meta_n_samples"]) != int(n_samples):
            return None
        if "meta_max_length" not in data.files or int(data["meta_max_length"]) != int(MAX_LENGTH):
            return None
        if "meta_bos" not in data.files or int(data["meta_bos"]) != int(1 if BOS else 0):
            return None
        if "meta_tail_drop" not in data.files or int(data["meta_tail_drop"]) != int(TAIL_DROP):
            return None

        if "meta_x_list" not in data.files or not np.array_equal(
            data["meta_x_list"].astype(np.int64), np.array(x_list, np.int64)
        ):
            return None
        if "meta_q_list" not in data.files or not np.allclose(
            data["meta_q_list"].astype(np.float32), np.array(q_list, np.float32)
        ):
            return None

        return {k: data[k] for k in data.files}
    except Exception:
        return None

def save_cache_npz(cache_path: str, payload: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(cache_path))
    np.savez_compressed(cache_path, **payload)


# =========================================================
# ES helpers (sorted-prefix for multi-k; partition for single-k)
# =========================================================
def es_bottom_k_sorted_prefix(prefix_1d: np.ndarray, k: int, drop: int) -> float:
    d = int(min(max(drop, 0), k - 1))
    denom = max(1, k - d)
    if d == 0:
        return float(prefix_1d[k - 1] / denom)
    return float((prefix_1d[k - 1] - prefix_1d[d - 1]) / denom)

def es_bottom_k_sorted_prefix_batch(prefix_2d: np.ndarray, k: int, drop: int) -> np.ndarray:
    d = int(min(max(drop, 0), k - 1))
    denom = max(1, k - d)
    if d == 0:
        return prefix_2d[:, k - 1] / denom
    return (prefix_2d[:, k - 1] - prefix_2d[:, d - 1]) / denom

def es_bottom_k_partition_1d(x: np.ndarray, k: int, drop: int) -> float:
    k = max(1, min(k, x.size))
    tail = np.partition(x, k - 1)[:k]
    d = int(min(max(drop, 0), k - 1))
    if d > 0:
        tail = np.partition(tail, d)[d:]
    return float(np.mean(tail))

def es_bottom_k_partition_2d(x: np.ndarray, k: int, drop: int) -> np.ndarray:
    n, t = x.shape
    k = max(1, min(k, t))
    tail = np.partition(x, k - 1, axis=1)[:, :k]
    d = int(min(max(drop, 0), k - 1))
    if d > 0:
        tail = np.partition(tail, d, axis=1)[:, d:]
    return np.mean(tail, axis=1)


# =========================================================
# Core feature extraction (parametrized)
# =========================================================
@torch.no_grad()
def extract_base_one_text(
    text: str,
    tokenizer,
    model,
    x_list: List[int],
    q_list: List[float],
    n_samples: int,
    # output control:
    q_for_all_x: float,          # e.g., 1.3 (analysis1 needs across x=1..50)
    x_for_all_q: int,            # e.g., 13  (analysis2 needs q-sweep at x=13)
) -> Dict[str, float]:
    """
    Returns base dict containing:
      - T
      - for each x in x_list: S_es{x}  (z-score)
      - for each x in x_list: H_renyi{q_for_all_x}_es{x}_norm
      - for x == x_for_all_q: H_renyi{q}_es{x}_norm for all q in q_list
    """
    def nan_output() -> Dict[str, float]:
        out: Dict[str, float] = {"T": 0.0}
        for x_tail in x_list:
            out[f"S_es{x_tail}"] = np.nan
            out[f"H_renyi{q_tag(q_for_all_x)}_es{x_tail}_norm"] = np.nan
        # q sweep only at x_for_all_q
        for q in q_list:
            out[f"H_renyi{q_tag(q)}_es{x_for_all_q}_norm"] = np.nan
        return out

    if not text or not isinstance(text, str):
        return nan_output()

    input_ids = encode_raw_text_bos_noeos(tokenizer, text, MAX_LENGTH, BOS)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    if int(input_ids.shape[1]) < 2:
        return nan_output()

    out_model = model(input_ids=input_ids, use_cache=False)
    logits = out_model.logits[0, :-1, :]   # (T, V)
    targets = input_ids[0, 1:]             # (T,)
    T = int(logits.shape[0])
    V = int(logits.shape[1])
    lnV = float(np.log(max(2, V)))

    # observed and sampled LOG-PROB scores
    S_obs = np.empty((T,), dtype=np.float32)
    S_samp = np.empty((n_samples, T), dtype=np.float32)

    # Renyi entropies per position for all q in q_list
    H_renyi = np.empty((len(q_list), T), dtype=np.float32)

    for start in range(0, T, CHUNK_SIZE):
        end = min(T, start + CHUNK_SIZE)
        chunk_logits = logits[start:end].to(torch.float32)   # (c, V)
        c = int(chunk_logits.shape[0])

        logp = torch.log_softmax(chunk_logits, dim=-1)       # (c, V)

        # observed token log-prob
        tgt = targets[start:end]
        ar = torch.arange(c, device=chunk_logits.device)
        logp_tok = logp[ar, tgt]
        S_obs[start:end] = logp_tok.detach().cpu().numpy().astype(np.float32)

        # Renyi entropies
        for qi, q in enumerate(q_list):
            log_sum = torch.logsumexp((q * logp), dim=-1)
            Hq = log_sum / (1.0 - q)
            H_renyi[qi, start:end] = Hq.detach().cpu().numpy().astype(np.float32)

        # sampling for z-score
        samp_ids = torch.distributions.Categorical(logits=chunk_logits).sample((n_samples,))  # (N, c)
        logp_samp = logp.gather(1, samp_ids.T).T                                              # (N, c)
        S_samp[:, start:end] = logp_samp.detach().cpu().numpy().astype(np.float32)

        del chunk_logits, logp, tgt, ar, logp_tok, samp_ids, logp_samp

    result: Dict[str, float] = {"T": float(T)}

    obs = S_obs.astype(np.float64)      # (T,)
    samp = S_samp.astype(np.float64)    # (N, T)

    # precompute sorted-prefix if multi-x, else partition for speed
    use_sorted = (len(x_list) > 1)
    if use_sorted:
        obs_sorted = np.sort(obs)
        obs_prefix = np.cumsum(obs_sorted)
        samp_sorted = np.sort(samp, axis=1)
        samp_prefix = np.cumsum(samp_sorted, axis=1)

    # build q index map
    q_to_idx = {float(q): i for i, q in enumerate(q_list)}
    if float(q_for_all_x) not in q_to_idx:
        raise ValueError(f"q_for_all_x={q_for_all_x} must be in q_list")

    q_all_idx = q_to_idx[float(q_for_all_x)]

    for x_tail in x_list:
        k = int(np.ceil((x_tail / 100.0) * T))
        k = max(1, min(k, T))

        # ES raw + z-score
        if use_sorted:
            raw_es = es_bottom_k_sorted_prefix(obs_prefix, k, TAIL_DROP)
            samp_es_vec = es_bottom_k_sorted_prefix_batch(samp_prefix, k, TAIL_DROP)
        else:
            raw_es = es_bottom_k_partition_1d(obs, k, TAIL_DROP)
            samp_es_vec = es_bottom_k_partition_2d(samp, k, TAIL_DROP)

        mu = float(np.mean(samp_es_vec))
        sd = float(np.std(samp_es_vec))
        z = (raw_es - mu) / (sd + EPS)
        result[f"S_es{x_tail}"] = float(z)

        # tail indices by obs (k smallest), then drop lowest inside tail
        tail_idx_k = np.argpartition(obs, k - 1)[:k]
        d = int(min(TAIL_DROP, k - 1))
        if d > 0:
            rm_local = np.argpartition(obs[tail_idx_k], d - 1)[:d]
            mask = np.ones(k, dtype=bool)
            mask[rm_local] = False
            tail_idx = tail_idx_k[mask]
        else:
            tail_idx = tail_idx_k

        # ---- Renyi entropy: q_for_all_x for all x ----
        Hq_all_tail = float(np.mean(H_renyi[q_all_idx, tail_idx].astype(np.float64)))
        result[f"H_renyi{q_tag(q_for_all_x)}_es{x_tail}_norm"] = float(Hq_all_tail / (lnV + EPS))

        # ---- Renyi entropy: full q sweep only at x_for_all_q ----
        if int(x_tail) == int(x_for_all_q):
            for q in q_list:
                qi = q_to_idx[float(q)]
                Hq_tail = float(np.mean(H_renyi[qi, tail_idx].astype(np.float64)))
                result[f"H_renyi{q_tag(q)}_es{x_for_all_q}_norm"] = float(Hq_tail / (lnV + EPS))

    return result


# =========================================================
# Dataset runner: extract BASE csv (cached by npz per id/kind)
# =========================================================
def extract_base_for_dataset(
    json_path: str,
    tokenizer,
    model,
    x_list: List[int],
    q_list: List[float],
    n_samples: int,
    q_for_all_x: float,
    x_for_all_q: int,
    cache_tag: str,
) -> str:
    """
    Writes features_base.csv and returns its path.
    """
    dname = dataset_name_from_path(json_path)
    out_dir = os.path.join(HP_OUTPUT_ROOT, dname)
    ensure_dir(out_dir)

    probs_dir = os.path.join(out_dir, f"probs_{cache_tag}")
    ensure_dir(probs_dir)

    base_csv = os.path.join(out_dir, f"features_base_{cache_tag}.csv")

    # columns we will write
    cols = ["T"]
    for x in x_list:
        cols.append(f"S_es{x}")
        cols.append(f"H_renyi{q_tag(q_for_all_x)}_es{x}_norm")
    for q in q_list:
        cols.append(f"H_renyi{q_tag(q)}_es{x_for_all_q}_norm")

    fieldnames = ["id", "kind", "label"] + cols

    cache_version = f"hp_es_base_{cache_tag}_v1"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(base_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(data, desc=f"[{dname}] BASE {cache_tag}"):
            sid = safe_id(item.get("id"))

            for kind, label, key in [("human", 0, "original_text"), ("ai", 1, "ai_generated_text")]:
                text = item.get(key, "")
                cache_path = os.path.join(probs_dir, f"id{sid}_{kind}.npz")

                cached = load_cache_npz(cache_path, cache_version, n_samples, x_list, q_list)
                if cached is not None:
                    base = {k: float(cached[k].item()) for k in cached.keys() if k in cols}
                    base["T"] = float(cached.get("T", np.array(np.nan)).item()) if "T" in cached else np.nan
                else:
                    feats = extract_base_one_text(
                        text=text,
                        tokenizer=tokenizer,
                        model=model,
                        x_list=x_list,
                        q_list=q_list,
                        n_samples=n_samples,
                        q_for_all_x=q_for_all_x,
                        x_for_all_q=x_for_all_q,
                    )
                    payload = meta_payload(cache_version, n_samples, x_list, q_list)
                    for k, v in feats.items():
                        payload[k] = np.array(v)
                    save_cache_npz(cache_path, payload)
                    base = feats

                row = {"id": item.get("id"), "kind": kind, "label": label}
                for c in cols:
                    row[c] = base.get(c, np.nan)
                writer.writerow(row)

    return base_csv


# =========================================================
# Build fused score from base csv
# =========================================================
def fused_score_from_df(df: pd.DataFrame, x: int, q: float, wz: float) -> np.ndarray:
    z_col = f"S_es{x}"
    h_col = f"H_renyi{q_tag(q)}_es{x}_norm"
    z = pd.to_numeric(df[z_col], errors="coerce").values.astype(float)
    h = pd.to_numeric(df[h_col], errors="coerce").values.astype(float)
    return wz * z + (1.0 - wz) * (1.0 - h)


# =========================================================
# Run analyses and save curves
# =========================================================
def run_curve_analysis(
    analysis_name: str,
    datasets: List[str],
    base_csv_map: Dict[str, str],
    x_values: List[float],
    build_cfg_fn,  # pv -> (x,q,wz)
    x_col_name: str,
    out_prefix: str,
    all_method_auc: Dict[str, Dict[str, float]],
) -> None:
    """
    Saves:
      - {out_prefix}_curve_per_dataset.csv
      - {out_prefix}_curve_avg.csv
    """
    rows = []
    for dname in datasets:
        df = pd.read_csv(base_csv_map[dname])
        y = df["label"].values.astype(int)

        for pv in x_values:
            x, q, wz = build_cfg_fn(pv)
            s = fused_score_from_df(df, x=x, q=q, wz=wz)
            auc_best = compute_auc_from_scores(y, s)
            rows.append({"dataset": dname, x_col_name: pv, "auc_best": auc_best})

            method = f"F_es{x}_rq{q_tag(q)}_wz{q_tag(wz)}"
            all_method_auc.setdefault(method, {})[dname] = auc_best

    curve_df = pd.DataFrame(rows)
    avg_df = curve_df.groupby(x_col_name)["auc_best"].mean().reset_index().rename(columns={"auc_best": "auc_best_mean"})

    ensure_dir(CURVE_DIR)
    per_path = os.path.join(CURVE_DIR, f"{out_prefix}_curve_per_dataset.csv")
    avg_path = os.path.join(CURVE_DIR, f"{out_prefix}_curve_avg.csv")
    curve_df.to_csv(per_path, index=False)
    avg_df.to_csv(avg_path, index=False)


def run_sample_analysis4(
    datasets: List[str],
    base_csv_map_ns: Dict[int, Dict[str, str]],  # n_samples -> {dataset -> base_csv}
    all_method_auc: Dict[str, Dict[str, float]],
) -> str:
    """
    Returns avg csv path for plotting.
    """
    rows = []
    for n in A4_N_LIST:
        for dname in datasets:
            df = pd.read_csv(base_csv_map_ns[n][dname])
            y = df["label"].values.astype(int)
            s = fused_score_from_df(df, x=A4_X, q=A4_Q, wz=A4_WZ)
            auc_best = compute_auc_from_scores(y, s)
            rows.append({"dataset": dplot_name(dname), "sample": float(n), "auc_best": auc_best})

            method = f"F_es{A4_X}_rq{q_tag(A4_Q)}_wz{q_tag(A4_WZ)}_ns{n}"
            all_method_auc.setdefault(method, {})[dname] = auc_best

    curve_df = pd.DataFrame(rows)
    avg_df = curve_df.groupby("sample")["auc_best"].mean().reset_index().rename(columns={"auc_best": "auc_best_mean"})

    ensure_dir(CURVE_DIR)
    per_path = os.path.join(CURVE_DIR, "analysis4_sample_curve_per_dataset.csv")
    avg_path = os.path.join(CURVE_DIR, "analysis4_sample_curve_avg.csv")
    curve_df.to_csv(per_path, index=False)
    avg_df.to_csv(avg_path, index=False)
    return avg_path


def dplot_name(dname: str) -> str:
    return dname


# =========================================================
# Summary table (best method across all tested configs)
# =========================================================
def generate_global_results_and_summary(
    datasets: List[str],
    all_method_auc: Dict[str, Dict[str, float]],
    out_root: str,
) -> None:
    ensure_dir(out_root)

    # per-dataset best
    res_rows = []
    for d in datasets:
        best_m, best_auc = None, -1.0
        for m, mp in all_method_auc.items():
            if d in mp and mp[d] == mp[d]:
                if mp[d] > best_auc:
                    best_auc = mp[d]
                    best_m = m
        res_rows.append({"dataset": d, "best_feature": best_m if best_m else "na", "best_auc": best_auc if best_m else np.nan})

    result_csv = os.path.join(out_root, "result.csv")
    pd.DataFrame(res_rows).to_csv(result_csv, index=False)

    # method averages + cover_all
    method_rows = []
    for m, mp in all_method_auc.items():
        vals = [mp.get(d, np.nan) for d in datasets]
        mean_auc = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan
        cover_all = all((d in mp) and (mp[d] == mp[d]) for d in datasets)
        method_rows.append({"method": m, "avg_auc": mean_auc, "cover_all": cover_all})

    all_methods_df = pd.DataFrame(method_rows).sort_values("avg_auc", ascending=False)
    all_methods_csv = os.path.join(out_root, "summary_table_all_methods.csv")
    all_methods_df.to_csv(all_methods_csv, index=False)

    cover_df = all_methods_df[all_methods_df["cover_all"] == True]
    if len(cover_df) == 0:
        print("\n[WARNING] No method covers all datasets.")
        print("Saved:", result_csv, all_methods_csv)
        return

    best_method = str(cover_df.iloc[0]["method"])
    summary_data = {d: all_method_auc[best_method][d] * 100.0 for d in datasets}
    summary_data["Avg."] = float(np.mean(list(summary_data.values())))

    summary_df = pd.DataFrame([summary_data])
    summary_df.index = [best_method]
    summary_csv = os.path.join(out_root, "summary_table.csv")
    summary_df.to_csv(summary_csv)

    print("\n" + "=" * 100)
    print(f"SUMMARY TABLE - Best Method: {best_method}")
    print("=" * 100)
    print("\nSaved:")
    print(" -", result_csv)
    print(" -", summary_csv)
    print(" -", all_methods_csv)
    print("=" * 100)


# =========================================================
# Main
# =========================================================
def main():
    torch.manual_seed(SAMPLE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SAMPLE_SEED)
    np.random.seed(SAMPLE_SEED)

    ensure_dir(HP_OUTPUT_ROOT)
    ensure_dir(CURVE_DIR)

    # resolve dataset paths
    json_files = []
    for fn in TARGET_FILES:
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            raise RuntimeError(f"Missing dataset file: {p}")
        json_files.append(p)

    datasets = [dataset_name_from_path(p) for p in json_files]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=MODEL_DTYPE,
        device_map="auto",
    )
    model.eval()

    all_method_auc: Dict[str, Dict[str, float]] = {}

    # ---------- BASE extraction for analyses 1-3 (N=100) ----------
    # We need:
    # - z for x=1..50
    # - entropy q=1.3 for all x=1..50
    # - entropy q-sweep only at x=13
    X_LIST_N100 = A1_X_LIST[:]  # 1..50
    Q_LIST_N100 = A2_Q_LIST[:]  # 0.5..2 step0.1 skip 1.0 (includes 1.3,1.5)

    base_csv_map_n100: Dict[str, str] = {}
    for jp, dname in zip(json_files, datasets):
        base_csv = extract_base_for_dataset(
            json_path=jp,
            tokenizer=tokenizer,
            model=model,
            x_list=X_LIST_N100,
            q_list=Q_LIST_N100,
            n_samples=100,
            q_for_all_x=A1_Q,      # 1.3 across all x
            x_for_all_q=13,        # q sweep only at x=13
            cache_tag="ns100_grid",
        )
        base_csv_map_n100[dname] = base_csv

    # ---------- Analysis 1: AUC vs tail_percent ----------
    run_curve_analysis(
        analysis_name="analysis1",
        datasets=datasets,
        base_csv_map=base_csv_map_n100,
        x_values=[float(x) for x in A1_X_LIST],
        build_cfg_fn=lambda pv: (int(pv), float(A1_Q), float(A1_WZ)),
        x_col_name="tail_percent",
        out_prefix="analysis1_tail_percent",
        all_method_auc=all_method_auc,
    )

    # ---------- Analysis 2: AUC vs alpha ----------
    run_curve_analysis(
        analysis_name="analysis2",
        datasets=datasets,
        base_csv_map=base_csv_map_n100,
        x_values=[float(q) for q in A2_Q_LIST],
        build_cfg_fn=lambda pv: (int(A2_X), float(pv), float(A2_WZ)),
        x_col_name="alpha",
        out_prefix="analysis2_alpha",
        all_method_auc=all_method_auc,
    )

    # ---------- Analysis 3: AUC vs w_z ----------
    run_curve_analysis(
        analysis_name="analysis3",
        datasets=datasets,
        base_csv_map=base_csv_map_n100,
        x_values=[float(w) for w in A3_WZ_LIST],
        build_cfg_fn=lambda pv: (int(A3_X), float(A3_Q), float(pv)),
        x_col_name="w_z",
        out_prefix="analysis3_w_z",
        all_method_auc=all_method_auc,
    )

    # ---------- BASE extraction for analysis 4 (vary N_SAMPLES) ----------
    # For N=100, we can reuse the ns100_grid base csv
    base_csv_map_ns: Dict[int, Dict[str, str]] = {100: base_csv_map_n100}

    for n in A4_N_LIST:
        if n == 100:
            continue
        base_csv_map_ns[n] = {}
        for jp, dname in zip(json_files, datasets):
            base_csv = extract_base_for_dataset(
                json_path=jp,
                tokenizer=tokenizer,
                model=model,
                x_list=[A4_X],           # only x=13
                q_list=[A4_Q],           # only q=1.5
                n_samples=n,
                q_for_all_x=A4_Q,        # same
                x_for_all_q=A4_X,        # same (so it writes H_renyi1.5_es13_norm)
                cache_tag=f"ns{n}_x13_q15",
            )
            base_csv_map_ns[n][dname] = base_csv

    # ---------- Analysis 4 curve ----------
    avg_path_a4 = run_sample_analysis4(
        datasets=datasets,
        base_csv_map_ns=base_csv_map_ns,
        all_method_auc=all_method_auc,
    )

    # ---------- Plot avg smooth curves (no best point) ----------
    # Titles / labels as you requested
    # 1
    plot_avg_smooth(
        avg_csv=os.path.join(CURVE_DIR, "analysis1_tail_percent_curve_avg.csv"),
        out_png=os.path.join(CURVE_DIR, "tail_percent_avg_smooth.png"),
        title="tail_percent (alpha=1.6, w_z=0.1, sample=100)",
        xlabel="tail_percent",
        ylabel="AUC on Reddit (Avg)",
    )
    # 2
    plot_avg_smooth(
        avg_csv=os.path.join(CURVE_DIR, "analysis2_alpha_curve_avg.csv"),
        out_png=os.path.join(CURVE_DIR, "alpha_avg_smooth.png"),
        title="alpha (tail_percent=13, w_z=0.1)",
        xlabel="alpha",
        ylabel="AUC on Reddit (Avg)",
    )
    # 3
    plot_avg_smooth(
        avg_csv=os.path.join(CURVE_DIR, "analysis3_w_z_curve_avg.csv"),
        out_png=os.path.join(CURVE_DIR, "w_z_avg_smooth.png"),
        title="w_z (tail_percent=13, alpha=1.6)",
        xlabel="w_z",
        ylabel="AUC on Reddit (Avg)",
    )
    # 4 (exact title you specified)
    plot_avg_smooth(
        avg_csv=avg_path_a4,
        out_png=os.path.join(CURVE_DIR, "sample_avg_smooth.png"),
        title="sample (tail_percent=13, alpha=1.6,  w_z=0.1)",
        xlabel="sample",
        ylabel="AUC on Reddit (Avg)",
    )

    # ---------- Final summary (like your original) ----------
    generate_global_results_and_summary(
        datasets=datasets,
        all_method_auc=all_method_auc,
        out_root=HP_OUTPUT_ROOT,
    )

    print("\n[OK] Done.")
    print("Curves CSV & figs saved under:", CURVE_DIR)
    print("Final summary under:", HP_OUTPUT_ROOT)


if __name__ == "__main__":
    main()
