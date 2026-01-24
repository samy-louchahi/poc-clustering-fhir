"""
Serious clustering benchmark (KMeans / DBSCAN / HDBSCAN)

- Preprocess ONCE: build matrix -> TFIDF -> SVD (X_reduced)
- For each method:
  - Run full clustering + metrics
  - Run BOOTSTRAP stability: repeat N_BOOTSTRAPS times on subsamples
  - Run repeated fits: KMeans across different seeds

Outputs:
- results/serious_benchmark/summary.csv  (aggregated)
- results/serious_benchmark/runs.csv     (per-run details)
"""

import os
import time
import math
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans, DBSCAN

import hdbscan  # you already have it (your earlier run worked)

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem


DATA_DIR = "data"
OUTPUT_DIR = "results/serious_benchmark"

# Preprocessing (shared for all methods)
APPLY_TFIDF = True
DIM_REDUCTION = "svd"
N_COMPONENTS = 30

# Benchmark rigor
N_BOOTSTRAPS = 10
BOOTSTRAP_FRAC = 0.8
KMEANS_REPEATS = 5

# Method params (tune later)
KMEANS_K = 3
DBSCAN_EPS = 0.6
DBSCAN_MIN_SAMPLES = 4

HDB_MIN_CLUSTER_SIZE = 10
HDB_MIN_SAMPLES = None


def _safe_metric(metric_fn, X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Compute metric ignoring noise and requiring >=2 clusters."""
    mask = labels != -1
    if mask.sum() < 2:
        return None
    y = labels[mask]
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(metric_fn(X[mask], y))
    except Exception:
        return None


def _centroid_stats(X: np.ndarray, labels: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute:
    - intra_mean: mean distance to cluster centroid (excluding noise)
    - inter_mean: mean pairwise centroid distance (excluding noise)
    - ratio: inter_mean / intra_mean
    """
    mask = labels != -1
    if mask.sum() < 2:
        return None, None, None

    Xf = X[mask]
    yf = labels[mask]
    clusters = np.unique(yf)
    if len(clusters) < 2:
        return None, None, None

    # centroids
    centroids = []
    intra_dists = []
    for c in clusters:
        pts = Xf[yf == c]
        if len(pts) == 0:
            continue
        mu = pts.mean(axis=0)
        centroids.append(mu)
        intra_dists.append(np.linalg.norm(pts - mu, axis=1))

    if not centroids:
        return None, None, None

    intra_mean = float(np.concatenate(intra_dists).mean())

    C = np.vstack(centroids)
    # pairwise centroid distances
    # simple O(k^2) since k is small
    dsum, n = 0.0, 0
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            dsum += float(np.linalg.norm(C[i] - C[j]))
            n += 1
    inter_mean = float(dsum / n) if n > 0 else None
    ratio = float(inter_mean / intra_mean) if (inter_mean is not None and intra_mean > 0) else None
    return intra_mean, inter_mean, ratio


def _size_stats(labels: np.ndarray) -> Tuple[Optional[float], Optional[float], int, int]:
    """Entropy of cluster sizes (excluding noise), imbalance max/min, n_clusters, noise_points."""
    noise_points = int(np.sum(labels == -1))
    labs = labels[labels != -1]
    if labs.size == 0:
        return None, None, 0, noise_points
    uniq, counts = np.unique(labs, return_counts=True)
    n_clusters = int(len(uniq))
    p = counts / counts.sum()

    entropy = float(-(p * np.log(p + 1e-12)).sum())
    if counts.min() == 0:
        imbalance = None
    else:
        imbalance = float(counts.max() / counts.min())
    return entropy, imbalance, n_clusters, noise_points


def _fit_and_measure(method: str, X: np.ndarray, seed: Optional[int] = None) -> Dict[str, Any]:
    """Fit one method on full X and return labels + metrics + time + mem peak."""
    t0 = time.perf_counter()

    tracemalloc.start()
    try:
        if method == "kmeans":
            model = KMeans(n_clusters=KMEANS_K, n_init="auto", random_state=seed)
            labels = model.fit_predict(X)
        elif method == "dbscan":
            model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            labels = model.fit_predict(X)
        elif method == "hdbscan":
            model = hdbscan.HDBSCAN(
                min_cluster_size=HDB_MIN_CLUSTER_SIZE,
                min_samples=HDB_MIN_SAMPLES,
            )
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unknown method: {method}")
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    runtime = time.perf_counter() - t0
    peak_mb = float(peak) / (1024 * 1024)

    labels = labels.astype(int)

    sil = _safe_metric(silhouette_score, X, labels)
    dbi = _safe_metric(davies_bouldin_score, X, labels)
    ch = _safe_metric(calinski_harabasz_score, X, labels)

    intra, inter, ratio = _centroid_stats(X, labels)
    entropy, imbalance, n_clusters, noise_points = _size_stats(labels)
    noise_ratio = float(noise_points / len(labels))

    return {
        "labels": labels,
        "runtime_sec": float(runtime),
        "peak_mem_mb": peak_mb,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": ch,
        "intra_mean_dist": intra,
        "inter_centroid_mean_dist": inter,
        "inter_over_intra": ratio,
        "size_entropy": entropy,
        "size_imbalance": imbalance,
        "n_clusters": n_clusters,
        "noise_points": noise_points,
        "noise_ratio": noise_ratio,
    }


def _bootstrap_stability(method: str, X: np.ndarray, full_labels: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    """Bootstrap stability: cluster on subsample and compare to full clustering restricted to subsample."""
    aris = []
    for _ in range(N_BOOTSTRAPS):
        idx = rng.choice(len(X), size=int(BOOTSTRAP_FRAC * len(X)), replace=False)
        X_sub = X[idx]
        out = _fit_and_measure(method, X_sub, seed=rng.integers(0, 10_000) if method == "kmeans" else None)
        sub_labels = out["labels"]

        # Compare only on sampled indices: full_labels restricted vs sub_labels
        ari = adjusted_rand_score(full_labels[idx], sub_labels)
        aris.append(float(ari))

    return {
        "bootstrap_ari_mean": float(np.mean(aris)),
        "bootstrap_ari_std": float(np.std(aris)),
        "bootstrap_ari_min": float(np.min(aris)),
        "bootstrap_ari_max": float(np.max(aris)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load patients (cached)
    patients = FHIRParser.load_directory(DATA_DIR, use_cache=True)
    if not patients:
        print("No patients found.")
        return
    print(f"Patients: {len(patients)}")

    # 2) Preprocess ONCE using your pipeline (we only use it to get X_reduced)
    print("\nPreprocessing once (matrix -> TFIDF -> SVD)...")
    pre = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=APPLY_TFIDF,
        dimensionality_reduction=DIM_REDUCTION,
        n_components=N_COMPONENTS,
        clustering_method="kmeans",   # dummy, we won’t keep its result
        n_clusters=KMEANS_K,
    )
    pre.fit(patients)

    X = pre.reduced_data
    if X is None:
        raise RuntimeError("Preprocessing did not produce reduced_data.")
    X = np.asarray(X, dtype=np.float32)
    print(f"X_reduced shape: {X.shape}")

    rng = np.random.default_rng(42)

    methods = ["kmeans", "dbscan", "hdbscan"]
    run_rows = []
    summary_rows = []

    for method in methods:
        print(f"\n=== {method.upper()} ===")

        # Full fit (with repeats for kmeans)
        full_runs = []
        if method == "kmeans":
            for r in range(KMEANS_REPEATS):
                seed = int(rng.integers(0, 1_000_000))
                out = _fit_and_measure(method, X, seed=seed)
                out["method"] = method
                out["run_type"] = "full"
                out["seed"] = seed
                run_rows.append({k: v for k, v in out.items() if k != "labels"})
                full_runs.append(out)
                print(f"  full run {r+1}/{KMEANS_REPEATS}: sil={out['silhouette']:.3f} | n_clusters={out['n_clusters']} | time={out['runtime_sec']:.2f}s")
            # choose the “best” full run for stability reference (highest silhouette)
            best_full = max(full_runs, key=lambda d: (d["silhouette"] if d["silhouette"] is not None else -1))
            full_labels_ref = best_full["labels"]
        else:
            out = _fit_and_measure(method, X)
            out["method"] = method
            out["run_type"] = "full"
            out["seed"] = None
            run_rows.append({k: v for k, v in out.items() if k != "labels"})
            full_labels_ref = out["labels"]
            print(f"  full: sil={out['silhouette']} | n_clusters={out['n_clusters']} | noise={out['noise_ratio']:.2%} | time={out['runtime_sec']:.2f}s")

        # Bootstrap stability
        stab = _bootstrap_stability(method, X, full_labels_ref, rng)

        # Aggregate (mean over full runs if kmeans)
        if method == "kmeans":
            agg_source = full_runs
            def _mean(key):
                vals = [d[key] for d in agg_source if d[key] is not None]
                return float(np.mean(vals)) if vals else None
            def _std(key):
                vals = [d[key] for d in agg_source if d[key] is not None]
                return float(np.std(vals)) if vals else None

            summary = {
                "method": method,
                "full_silhouette_mean": _mean("silhouette"),
                "full_silhouette_std": _std("silhouette"),
                "full_davies_bouldin_mean": _mean("davies_bouldin"),
                "full_calinski_harabasz_mean": _mean("calinski_harabasz"),
                "full_inter_over_intra_mean": _mean("inter_over_intra"),
                "full_runtime_sec_mean": _mean("runtime_sec"),
                "full_peak_mem_mb_mean": _mean("peak_mem_mb"),
                "full_n_clusters": KMEANS_K,
                "full_noise_ratio_mean": _mean("noise_ratio"),
                **stab,
            }
        else:
            # single full run
            summary = {
                "method": method,
                "full_silhouette_mean": run_rows[-1]["silhouette"],
                "full_silhouette_std": None,
                "full_davies_bouldin_mean": run_rows[-1]["davies_bouldin"],
                "full_calinski_harabasz_mean": run_rows[-1]["calinski_harabasz"],
                "full_inter_over_intra_mean": run_rows[-1]["inter_over_intra"],
                "full_runtime_sec_mean": run_rows[-1]["runtime_sec"],
                "full_peak_mem_mb_mean": run_rows[-1]["peak_mem_mb"],
                "full_n_clusters": run_rows[-1]["n_clusters"],
                "full_noise_ratio_mean": run_rows[-1]["noise_ratio"],
                **stab,
            }

        summary_rows.append(summary)

        print(
            f"  bootstrap stability ARI: mean={stab['bootstrap_ari_mean']:.3f} "
            f"(min={stab['bootstrap_ari_min']:.3f}, max={stab['bootstrap_ari_max']:.3f})"
        )

    df_runs = pd.DataFrame(run_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_runs.to_csv(os.path.join(OUTPUT_DIR, "runs.csv"), index=False)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    print(f"\nSaved: {OUTPUT_DIR}/runs.csv")
    print(f"Saved: {OUTPUT_DIR}/summary.csv\n")
    print(df_summary)


if __name__ == "__main__":
    main()