"""
Tune HDBSCAN:
- Grid search over (min_cluster_size, min_samples) on X_reduced.npy
- Uses constraints (noise ratio, dominance, n_clusters) + quality metrics
- Optional bootstrap stability (ARI) for "serious" mode

Outputs to:
  results/hdbscan/tuning/
    hdbscan_grid_metrics.csv
    chosen_params.json
    plots/
      noise_ratio_vs_mcs.png
      n_clusters_vs_mcs.png
      silhouette_vs_mcs.png
      dominance_vs_mcs.png

Run:
  python -u tune_hdbscan.py
"""

import os
import json
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

import hdbscan


DEFAULT_X_PATH = "results/pre_process/artifacts/X_reduced.npy"
DEFAULT_OUT_DIR = "results/hdbscan/tuning"


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


def cluster_stats(labels: np.ndarray) -> Dict[str, Any]:
    """Compute cluster count, noise ratio, dominance, min/max sizes (excluding noise)."""
    n = len(labels)
    noise = int(np.sum(labels == -1))
    noise_ratio = float(noise / n)

    labs = labels[labels != -1]
    if labs.size == 0:
        return {
            "n_clusters": 0,
            "noise_points": noise,
            "noise_ratio": noise_ratio,
            "dominance": None,
            "min_cluster_size_found": None,
            "max_cluster_size_found": None,
        }

    uniq, counts = np.unique(labs, return_counts=True)
    n_clusters = int(len(uniq))
    max_size = int(counts.max())
    min_size = int(counts.min())
    dominance = float(max_size / (n - noise)) if (n - noise) > 0 else None

    return {
        "n_clusters": n_clusters,
        "noise_points": noise,
        "noise_ratio": noise_ratio,
        "dominance": dominance,  # fraction of non-noise captured by largest cluster
        "min_cluster_size_found": min_size,
        "max_cluster_size_found": max_size,
    }


def bootstrap_stability_ari(
    X: np.ndarray,
    full_labels: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
    n_boot: int,
    frac: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Bootstrap stability ARI between full labeling restricted to sample and sample-fit labeling."""
    aris = []
    n = len(X)
    m = int(frac * n)
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        Xs = X[idx]

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        sub_labels = model.fit_predict(Xs).astype(int)

        ari = adjusted_rand_score(full_labels[idx], sub_labels)
        aris.append(float(ari))

    return {
        "bootstrap_ari_mean": float(np.mean(aris)),
        "bootstrap_ari_std": float(np.std(aris)),
        "bootstrap_ari_min": float(np.min(aris)),
        "bootstrap_ari_max": float(np.max(aris)),
    }


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


def run_hdbscan_tuning_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path", default=DEFAULT_X_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    # adaptive grid
    parser.add_argument("--mcs_fracs", default="0.01,0.02,0.03,0.05", help="fractions of N for min_cluster_size candidates")
    parser.add_argument("--mcs_min", type=int, default=5)
    parser.add_argument("--mcs_max", type=int, default=60)

    # constraints (adaptable)
    parser.add_argument("--noise_min", type=float, default=0.15)
    parser.add_argument("--noise_max", type=float, default=0.60)
    parser.add_argument("--dominance_max", type=float, default=0.85)
    parser.add_argument("--min_clusters", type=int, default=2)

    # stability
    parser.add_argument("--bootstrap", action="store_true", default=True)
    parser.add_argument("--n_boot", type=int, default=8)
    parser.add_argument("--boot_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    X = np.load(args.x_path)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]

    # Build min_cluster_size candidates from fractions of N
    fracs = [float(x.strip()) for x in args.mcs_fracs.split(",") if x.strip()]
    mcs_candidates = sorted(set(
        max(args.mcs_min, min(args.mcs_max, int(round(n * f))))
        for f in fracs
    ))
    # ensure uniqueness + sorted
    mcs_candidates = sorted(set(mcs_candidates))

    # min_samples candidates per mcs
    def min_samples_candidates(mcs: int) -> List[Optional[int]]:
        half = max(1, int(round(mcs / 2)))
        return [None, half, mcs]

    print(f"Loaded X: shape={X.shape}")
    print(f"min_cluster_size candidates: {mcs_candidates}")
    print(f"Constraints: noise in [{args.noise_min}, {args.noise_max}], dominance<{args.dominance_max}, n_clusters>={args.min_clusters}")
    print(f"Bootstrap stability: {args.bootstrap} (n_boot={args.n_boot}, frac={args.boot_frac})")

    rng = np.random.default_rng(args.seed)

    rows = []
    # For plotting summaries by mcs (we will keep best-per-mcs later)
    for mcs in mcs_candidates:
        for ms in min_samples_candidates(mcs):
            t0 = time.perf_counter()

            model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            labels = model.fit_predict(X).astype(int)

            dt = time.perf_counter() - t0

            stats = cluster_stats(labels)

            sil = _safe_metric(silhouette_score, X, labels)
            dbi = _safe_metric(davies_bouldin_score, X, labels)
            ch = _safe_metric(calinski_harabasz_score, X, labels)

            stab = {"bootstrap_ari_mean": None, "bootstrap_ari_std": None, "bootstrap_ari_min": None, "bootstrap_ari_max": None}
            if args.bootstrap:
                stab = bootstrap_stability_ari(
                    X=X,
                    full_labels=labels,
                    min_cluster_size=mcs,
                    min_samples=ms,
                    n_boot=args.n_boot,
                    frac=args.boot_frac,
                    rng=rng,
                )

            rows.append({
                "min_cluster_size": int(mcs),
                "min_samples": (None if ms is None else int(ms)),
                "runtime_sec": float(dt),

                "silhouette": sil,
                "davies_bouldin": dbi,
                "calinski_harabasz": ch,

                **stats,
                **stab,
            })

            print(
                f"mcs={mcs:2d} ms={str(ms):>4} "
                f"n_clusters={stats['n_clusters']:2d} noise={stats['noise_ratio']:.1%} "
                f"dom={stats['dominance'] if stats['dominance'] is not None else 'NA'} "
                f"sil={sil if sil is not None else 'NA'} time={dt:.2f}s"
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "hdbscan_grid_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ---- Selection under constraints ----
    def is_ok(r) -> bool:
        if r["n_clusters"] < args.min_clusters:
            return False
        if not (args.noise_min <= r["noise_ratio"] <= args.noise_max):
            return False
        if r["dominance"] is None:
            return False
        if r["dominance"] >= args.dominance_max:
            return False
        # must have at least silhouette or DBI to judge
        if r["silhouette"] is None and r["davies_bouldin"] is None:
            return False
        return True

    ok = df[df.apply(is_ok, axis=1)].copy()

    if ok.empty:
        # fallback: relax constraints slightly, choose best silhouette among non-degenerate configs
        print("\n[WARN] No config satisfied constraints. Falling back to best silhouette among configs with >=2 clusters.")
        ok = df[df["n_clusters"] >= 2].copy()

    # Composite score: prioritize stability then silhouette then lower noise (closer to mid), then smaller mcs (simplicity)
    noise_mid = (args.noise_min + args.noise_max) / 2.0

    def score(r) -> float:
        # higher is better
        stab = r["bootstrap_ari_mean"] if (r["bootstrap_ari_mean"] is not None and not np.isnan(r["bootstrap_ari_mean"])) else 0.0
        silv = r["silhouette"] if (r["silhouette"] is not None and not np.isnan(r["silhouette"])) else -1.0
        # penalize noise far from mid
        noise_pen = abs(r["noise_ratio"] - noise_mid)
        # DBI lower is better -> convert to a bonus
        db_bonus = 0.0
        if r["davies_bouldin"] is not None and not np.isnan(r["davies_bouldin"]):
            db_bonus = 1.0 / (1.0 + r["davies_bouldin"])
        # smaller mcs and ms slightly preferred (simplicity)
        complexity = 0.001 * r["min_cluster_size"] + (0.0005 * (r["min_samples"] if r["min_samples"] is not None else r["min_cluster_size"]))
        return (2.0 * stab) + (1.0 * silv) + (0.5 * db_bonus) - (0.5 * noise_pen) - complexity

    ok["score"] = ok.apply(score, axis=1)
    best = ok.sort_values("score", ascending=False).iloc[0]

    chosen = {
        "min_cluster_size_opt": int(best["min_cluster_size"]),
        "min_samples_opt": (None if pd.isna(best["min_samples"]) else int(best["min_samples"])),
        "decision_rule": "constraints(noise, dominance, n_clusters) then max(score=2*stability + silhouette + db_bonus - noise_pen - complexity)",
        "constraints": {
            "noise_min": args.noise_min,
            "noise_max": args.noise_max,
            "dominance_max": args.dominance_max,
            "min_clusters": args.min_clusters,
        },
        "metrics_at_opt": {k: (None if (pd.isna(best[k]) or best[k] is None) else float(best[k]) if isinstance(best[k], (float, np.floating)) else best[k]) for k in best.index if k != "score"},
    }

    with open(os.path.join(args.out_dir, "chosen_params.json"), "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    print(f"Saved: {args.out_dir}/chosen_params.json")
    print(f"Chosen: mcs={chosen['min_cluster_size_opt']}, min_samples={chosen['min_samples_opt']}")

    # ---- Plots by mcs (use best ms per mcs for readability) ----
    # pick best row per mcs (max score, using ok if possible else full df)
    base = ok if not ok.empty else df.copy()
    best_per_mcs = base.sort_values("score" if "score" in base.columns else "silhouette", ascending=False).groupby("min_cluster_size", as_index=False).head(1)
    best_per_mcs = best_per_mcs.sort_values("min_cluster_size")

    plot_line(
        best_per_mcs["min_cluster_size"],
        best_per_mcs["noise_ratio"],
        "HDBSCAN noise ratio vs min_cluster_size (best min_samples)",
        "min_cluster_size",
        "noise_ratio",
        os.path.join(plots_dir, "noise_ratio_vs_mcs.png"),
    )
    plot_line(
        best_per_mcs["min_cluster_size"],
        best_per_mcs["n_clusters"],
        "HDBSCAN n_clusters vs min_cluster_size (best min_samples)",
        "min_cluster_size",
        "n_clusters",
        os.path.join(plots_dir, "n_clusters_vs_mcs.png"),
    )
    plot_line(
        best_per_mcs["min_cluster_size"],
        best_per_mcs["silhouette"],
        "HDBSCAN silhouette vs min_cluster_size (best min_samples)",
        "min_cluster_size",
        "silhouette",
        os.path.join(plots_dir, "silhouette_vs_mcs.png"),
    )
    plot_line(
        best_per_mcs["min_cluster_size"],
        best_per_mcs["dominance"],
        "HDBSCAN dominance vs min_cluster_size (best min_samples)",
        "min_cluster_size",
        "dominance (largest cluster / non-noise)",
        os.path.join(plots_dir, "dominance_vs_mcs.png"),
    )

    print(f"Saved plots to: {plots_dir}")
    print("\nHDBSCAN tuning complete.")


def run_hdbscan_tuning(
    artifacts: dict,
    out_dir: str = DEFAULT_OUT_DIR,
    mcs_fracs: str = "0.01,0.02,0.03,0.05",
    mcs_min: int = 5,
    mcs_max: int = 60,
    noise_min: float = 0.15,
    noise_max: float = 0.60,
    dominance_max: float = 0.85,
    min_clusters: int = 2,
    bootstrap: bool = True,
    n_boot: int = 8,
    boot_frac: float = 0.8,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """
    Uses artifacts['x_path'] (X_reduced.npy).
    Writes results/hdbscan/tuning/.
    Returns chosen params dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    chosen_path = os.path.join(out_dir, "chosen_params.json")
    if (not force) and os.path.exists(chosen_path):
        with open(chosen_path, "r", encoding="utf-8") as f:
            return json.load(f)

    x_path = artifacts.get("x_path", DEFAULT_X_PATH)
    X = np.load(x_path).astype(np.float32)
    n = X.shape[0]

    fracs = [float(x.strip()) for x in mcs_fracs.split(",") if x.strip()]
    mcs_candidates = sorted(set(max(mcs_min, min(mcs_max, int(round(n * f)))) for f in fracs))

    def min_samples_candidates(mcs: int):
        half = max(1, int(round(mcs / 2)))
        return [None, half, mcs]

    rng = np.random.default_rng(seed)

    rows = []
    for mcs in mcs_candidates:
        for ms in min_samples_candidates(mcs):
            model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            labels = model.fit_predict(X).astype(int)

            stats = cluster_stats(labels)
            sil = _safe_metric(silhouette_score, X, labels)
            dbi = _safe_metric(davies_bouldin_score, X, labels)
            ch = _safe_metric(calinski_harabasz_score, X, labels)

            stab = {"bootstrap_ari_mean": None, "bootstrap_ari_std": None, "bootstrap_ari_min": None, "bootstrap_ari_max": None}
            if bootstrap:
                stab = bootstrap_stability_ari(X, labels, mcs, ms, n_boot, boot_frac, rng)

            rows.append({
                "min_cluster_size": int(mcs),
                "min_samples": (None if ms is None else int(ms)),
                "silhouette": sil,
                "davies_bouldin": dbi,
                "calinski_harabasz": ch,
                **stats,
                **stab,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "hdbscan_grid_metrics.csv"), index=False)

    def is_ok(r):
        if r["n_clusters"] < min_clusters:
            return False
        if not (noise_min <= r["noise_ratio"] <= noise_max):
            return False
        if r["dominance"] is None or r["dominance"] >= dominance_max:
            return False
        if r["silhouette"] is None and r["davies_bouldin"] is None:
            return False
        return True

    ok = df[df.apply(is_ok, axis=1)].copy()
    if ok.empty:
        ok = df[df["n_clusters"] >= 2].copy()

    noise_mid = (noise_min + noise_max) / 2.0

    def score(r):
        stab = r["bootstrap_ari_mean"] if (r["bootstrap_ari_mean"] is not None and not np.isnan(r["bootstrap_ari_mean"])) else 0.0
        silv = r["silhouette"] if (r["silhouette"] is not None and not np.isnan(r["silhouette"])) else -1.0
        noise_pen = abs(r["noise_ratio"] - noise_mid)
        db_bonus = 0.0
        if r["davies_bouldin"] is not None and not np.isnan(r["davies_bouldin"]):
            db_bonus = 1.0 / (1.0 + r["davies_bouldin"])
        complexity = 0.001 * r["min_cluster_size"] + (0.0005 * (r["min_samples"] if r["min_samples"] is not None else r["min_cluster_size"]))
        return (2.0 * stab) + (1.0 * silv) + (0.5 * db_bonus) - (0.5 * noise_pen) - complexity

    ok["score"] = ok.apply(score, axis=1)
    best = ok.sort_values("score", ascending=False).iloc[0]

    chosen = {
        "min_cluster_size_opt": int(best["min_cluster_size"]),
        "min_samples_opt": (None if pd.isna(best["min_samples"]) else int(best["min_samples"])),
        "decision_rule": "constraints then composite score",
        "constraints": {
            "noise_min": noise_min, "noise_max": noise_max,
            "dominance_max": dominance_max, "min_clusters": min_clusters,
        },
        "metrics_at_opt": {k: (None if (pd.isna(best[k]) or best[k] is None) else float(best[k]) if isinstance(best[k], (float, np.floating)) else best[k]) for k in best.index if k != "score"},
    }

    with open(chosen_path, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)

    # plots: best per mcs
    base = ok
    best_per_mcs = base.sort_values("score", ascending=False).groupby("min_cluster_size", as_index=False).head(1).sort_values("min_cluster_size")
    plot_line(best_per_mcs["min_cluster_size"], best_per_mcs["noise_ratio"], "Noise ratio vs min_cluster_size", "min_cluster_size", "noise_ratio", os.path.join(plots_dir, "noise_ratio_vs_mcs.png"))
    plot_line(best_per_mcs["min_cluster_size"], best_per_mcs["n_clusters"], "n_clusters vs min_cluster_size", "min_cluster_size", "n_clusters", os.path.join(plots_dir, "n_clusters_vs_mcs.png"))
    plot_line(best_per_mcs["min_cluster_size"], best_per_mcs["silhouette"], "silhouette vs min_cluster_size", "min_cluster_size", "silhouette", os.path.join(plots_dir, "silhouette_vs_mcs.png"))
    plot_line(best_per_mcs["min_cluster_size"], best_per_mcs["dominance"], "dominance vs min_cluster_size", "min_cluster_size", "dominance", os.path.join(plots_dir, "dominance",))

    return chosen