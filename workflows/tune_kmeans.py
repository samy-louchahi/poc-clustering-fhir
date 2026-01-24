"""
Tune KMeans:
- Choose k using elbow (inertia), silhouette, and stability (ARI across seeds)
- Works on precomputed X_reduced.npy (from preprocess.py)

Outputs to:
  results/k_mean/tuning/
    kmeans_k_metrics.csv
    elbow_inertia.png
    silhouette_vs_k.png
    stability_ari_vs_k.png
    chosen_k.json

Run:
  python -u tune_kmeans.py
"""

import os
import json
import argparse
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score


DEFAULT_X_PATH = "results/pre_process/artifacts/X_reduced.npy"
DEFAULT_OUT_DIR = "results/k_mean/tuning"


def knee_point(k_values: List[int], inertia_values: List[float]) -> int:
    """
    Simple 'knee' detection:
    compute distance of each (k, inertia) point to the line between first and last.
    choose k with maximum distance.
    """
    ks = np.array(k_values, dtype=float)
    ys = np.array(inertia_values, dtype=float)

    # Normalize to [0,1] for numeric stability
    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)

    p1 = np.array([ks_n[0], ys_n[0]])
    p2 = np.array([ks_n[-1], ys_n[-1]])

    # distance from point to line
    line = p2 - p1
    line_norm = np.linalg.norm(line) + 1e-12

    distances = []
    for x, y in zip(ks_n, ys_n):
        p = np.array([x, y])
        # area of parallelogram / base length
        dist = np.abs(np.cross(line, p - p1)) / line_norm
        distances.append(dist)

    best_idx = int(np.argmax(distances))
    return int(k_values[best_idx])


def compute_stability_ari(X: np.ndarray, k: int, seeds: List[int]) -> Dict[str, float]:
    """
    Compute stability as mean/std ARI between clusterings from different seeds.
    """
    labels_list = []
    for seed in seeds:
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
        labels_list.append(km.fit_predict(X))

    # pairwise ARI
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    if not aris:
        return {"ari_mean": float("nan"), "ari_std": float("nan")}
    return {"ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris))}


def plot_line(x, y, title, xlabel, ylabel, path):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run_kmeans_tuning_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path", default=DEFAULT_X_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=None)  # adaptive if None
    parser.add_argument("--repeats", type=int, default=5, help="number of seeds for stability")
    parser.add_argument("--seed_base", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(args.x_path)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]

    # Adaptive k_max
    if args.k_max is None:
        args.k_max = min(12, int(np.sqrt(n)))
        args.k_max = max(args.k_max, args.k_min + 1)

    k_values = list(range(args.k_min, args.k_max + 1))
    seeds = [args.seed_base + i * 17 for i in range(args.repeats)]

    print(f"Loaded X: shape={X.shape}")
    print(f"Testing k in {k_values}")
    print(f"Stability seeds: {seeds}")

    rows = []
    for k in k_values:
        t0 = time.perf_counter()

        km = KMeans(n_clusters=k, n_init="auto", random_state=args.seed_base)
        labels = km.fit_predict(X)

        inertia = float(km.inertia_)

        # cluster quality metrics
        sil = float(silhouette_score(X, labels)) if k >= 2 else float("nan")
        dbi = float(davies_bouldin_score(X, labels)) if k >= 2 else float("nan")
        ch = float(calinski_harabasz_score(X, labels)) if k >= 2 else float("nan")

        stab = compute_stability_ari(X, k, seeds)

        dt = time.perf_counter() - t0
        rows.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch,
            "stability_ari_mean": stab["ari_mean"],
            "stability_ari_std": stab["ari_std"],
            "runtime_sec": float(dt),
        })

        print(f"k={k:2d} inertia={inertia:.2f} sil={sil:.3f} ARI={stab['ari_mean']:.3f} time={dt:.2f}s")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "kmeans_k_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Choose k: knee point then tie-breakers
    k_knee = knee_point(df["k"].tolist(), df["inertia"].tolist())

    # Consider candidates around knee
    candidates = sorted(set([k_knee - 1, k_knee, k_knee + 1]))
    candidates = [k for k in candidates if k in k_values]

    # Score: prioritize stability then silhouette then lower k (simplicity)
    def score_row(r):
        # higher is better
        return (
            r["stability_ari_mean"] if not np.isnan(r["stability_ari_mean"]) else -1.0,
            r["silhouette"] if not np.isnan(r["silhouette"]) else -1.0,
            -r["k"],  # prefer smaller k if tied
        )

    df_cand = df[df["k"].isin(candidates)].copy()
    best = df_cand.sort_values(by=["stability_ari_mean", "silhouette", "k"], ascending=[False, False, True]).iloc[0]

    chosen = {
        "knee_k": int(k_knee),
        "candidates": candidates,
        "k_opt": int(best["k"]),
        "decision_rule": "knee(inertia) then max(stability_ari_mean, silhouette) then min(k)",
        "metrics_at_k_opt": {col: (float(best[col]) if col not in ["k"] else int(best[col])) for col in best.index},
    }

    with open(os.path.join(args.out_dir, "chosen_k.json"), "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    print(f"Saved: {args.out_dir}/chosen_k.json")
    print(f"Chosen k_opt={chosen['k_opt']} (knee={chosen['knee_k']}, candidates={chosen['candidates']})")

    # Plots
    plot_line(df["k"], df["inertia"], "KMeans Elbow (Inertia)", "k", "inertia (WCSS)", os.path.join(args.out_dir, "elbow_inertia.png"))
    plot_line(df["k"], df["silhouette"], "Silhouette vs k", "k", "silhouette", os.path.join(args.out_dir, "silhouette_vs_k.png"))
    plot_line(df["k"], df["stability_ari_mean"], "Stability (ARI mean) vs k", "k", "ARI mean", os.path.join(args.out_dir, "stability_ari_vs_k.png"))

    print(f"Saved plots to: {args.out_dir}")


def run_kmeans_tuning(
    artifacts: dict,
    out_dir: str = DEFAULT_OUT_DIR,
    k_min: int = 2,
    k_max: Optional[int] = None,
    repeats: int = 5,
    seed_base: int = 42,
    force: bool = False,
) -> dict:
    """
    Uses artifacts['x_path'] (X_reduced.npy).
    Writes results/k_mean/tuning/.
    Returns the loaded chosen_k dict.
    """
    os.makedirs(out_dir, exist_ok=True)

    chosen_path = os.path.join(out_dir, "chosen_k.json")
    if (not force) and os.path.exists(chosen_path):
        with open(chosen_path, "r", encoding="utf-8") as f:
            return json.load(f)

    x_path = artifacts.get("x_path", DEFAULT_X_PATH)
    X = np.load(x_path).astype(np.float32)
    n = X.shape[0]

    if k_max is None:
        k_max = min(12, int(np.sqrt(n)))
        k_max = max(k_max, k_min + 1)

    k_values = list(range(k_min, k_max + 1))
    seeds = [seed_base + i * 17 for i in range(repeats)]

    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed_base)
        labels = km.fit_predict(X)

        inertia = float(km.inertia_)
        sil = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        stab = compute_stability_ari(X, k, seeds)

        rows.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch,
            "stability_ari_mean": stab["ari_mean"],
            "stability_ari_std": stab["ari_std"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "kmeans_k_metrics.csv"), index=False)

    k_knee = knee_point(df["k"].tolist(), df["inertia"].tolist())
    candidates = [k for k in sorted(set([k_knee - 1, k_knee, k_knee + 1])) if k in k_values]
    df_cand = df[df["k"].isin(candidates)].copy()
    best = df_cand.sort_values(by=["stability_ari_mean", "silhouette", "k"], ascending=[False, False, True]).iloc[0]

    chosen = {
        "knee_k": int(k_knee),
        "candidates": candidates,
        "k_opt": int(best["k"]),
        "decision_rule": "knee(inertia) then max(stability_ari_mean, silhouette) then min(k)",
        "metrics_at_k_opt": {col: (float(best[col]) if col != "k" else int(best[col])) for col in best.index},
    }

    with open(chosen_path, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)

    # plots
    plot_line(df["k"], df["inertia"], "KMeans Elbow (Inertia)", "k", "inertia (WCSS)", os.path.join(out_dir, "elbow_inertia.png"))
    plot_line(df["k"], df["silhouette"], "Silhouette vs k", "k", "silhouette", os.path.join(out_dir, "silhouette_vs_k.png"))
    plot_line(df["k"], df["stability_ari_mean"], "Stability (ARI mean) vs k", "k", "ARI mean", os.path.join(out_dir, "stability_ari_vs_k.png"))

    return chosen