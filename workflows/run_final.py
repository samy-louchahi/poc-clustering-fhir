"""
Final clustering run (KMeans + HDBSCAN) using tuned parameters.

Called by run_poc.py (do not run this file directly).
"""

import os
import json
import pandas as pd

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem
from fhir_clustering.visualization import save_all_plots
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def _flatten_top_codes(top_codes: dict) -> pd.DataFrame:
    rows = []
    for cid, codes in top_codes.items():
        for rank, (code, score) in enumerate(codes, start=1):
            rows.append({"cluster_id": int(cid), "rank": int(rank), "code": code, "score": float(score)})
    return pd.DataFrame(rows)

def _run_hdbscan_in_kmeans(
    X: np.ndarray,
    kmeans_labels: np.ndarray,
    patient_ids: np.ndarray,
    interpreter,
    min_cluster_size: int,
    min_samples,
    min_points_factor: int = 2,
    min_points_floor: int = 25,
    offset_base: int = 1000,
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run HDBSCAN inside each KMeans cluster (hierarchical).
    Returns:
      - assignments_df: patient_id, kmeans_cluster, hdbscan_in_kmeans_local, hdbscan_in_kmeans_global
      - top_codes_df: kmeans_cluster, subcluster_local_id, subcluster_global_id, rank, code, score
    """
    n = len(patient_ids)
    X = np.asarray(X, dtype=np.float32)
    kmeans_labels = kmeans_labels.astype(int)

    # Init all as noise / not clustered
    hdb_local = np.full(n, -1, dtype=int)
    hdb_global = np.full(n, -1, dtype=int)

    top_rows = []

    for k_id in sorted(np.unique(kmeans_labels)):
        idx = np.where(kmeans_labels == k_id)[0]
        n_k = len(idx)

        # Skip if not enough points
        min_needed = max(min_points_floor, min_points_factor * min_cluster_size)
        if n_k < min_needed:
            continue

        X_sub = X[idx]

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        sub_labels = model.fit_predict(X_sub).astype(int)  # -1 = noise

        # store local labels
        hdb_local[idx] = sub_labels

        # globalize labels to avoid collisions between kmeans clusters
        offset = int(k_id) * offset_base
        for loc_i, glob_i in enumerate(idx):
            if sub_labels[loc_i] != -1:
                hdb_global[glob_i] = offset + int(sub_labels[loc_i])

        # Use interpreter to compute top codes for subclusters of this kmeans cluster
        if interpreter is not None:
            full_labels = np.full(n, -1, dtype=int)
            full_labels[idx] = hdb_global[idx]  # already offset global ids

            top_dict = interpreter.get_top_codes_per_cluster(
                full_labels, top_n=top_n, method="distinctiveness"
            )

            # flatten with extra info
            for global_sub_id, codes in top_dict.items():
                local_sub_id = int(global_sub_id) - offset
                for rank, (code, score) in enumerate(codes, start=1):
                    top_rows.append({
                        "kmeans_cluster": int(k_id),
                        "subcluster_local_id": int(local_sub_id),
                        "subcluster_global_id": int(global_sub_id),
                        "rank": int(rank),
                        "code": code,
                        "score": float(score),
                    })

    assignments_df = pd.DataFrame({
        "patient_id": patient_ids,
        "kmeans_cluster": kmeans_labels,
        "hdbscan_in_kmeans_local": hdb_local,
        "hdbscan_in_kmeans_global": hdb_global,
    })

    top_codes_df = pd.DataFrame(top_rows)

    return assignments_df, top_codes_df



def _save_bar_subclusters(summary_hk: pd.DataFrame, hk_dir: str):
    """
    For each kmeans_cluster, save a bar chart of subcluster sizes (including -1).
    """
    plots_dir = os.path.join(hk_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for k_id in sorted(summary_hk["kmeans_cluster"].unique()):
        dfk = summary_hk[summary_hk["kmeans_cluster"] == k_id].copy()

        plt.figure(figsize=(10, 4))
        plt.bar(dfk["hdbscan_in_kmeans_local"].astype(str), dfk["n_patients"])
        plt.title(f"HDBSCAN-in-KMeans cluster sizes (KMeans={k_id})")
        plt.xlabel("hdbscan_in_kmeans_local")
        plt.ylabel("n_patients")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"subcluster_sizes_kmeans_{k_id}.png"), dpi=300)
        plt.close()


def _save_scatter_hierarchical(X_reduced: np.ndarray, df_h_in_k: pd.DataFrame, hk_dir: str):
    """
    PCA 2D scatter:
      - color by hdbscan_in_kmeans_global
      - noise (-1) shown separately
    """
    plots_dir = os.path.join(hk_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    X = np.asarray(X_reduced, dtype=np.float32)
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    labels = df_h_in_k["hdbscan_in_kmeans_global"].to_numpy().astype(int)
    noise_mask = labels == -1

    plt.figure(figsize=(10, 7))
    if noise_mask.any():
        plt.scatter(X2[noise_mask, 0], X2[noise_mask, 1], s=8, alpha=0.35, label="noise")

    for lab in sorted(np.unique(labels[~noise_mask])) if (~noise_mask).any() else []:
        m = labels == lab
        plt.scatter(X2[m, 0], X2[m, 1], s=10, alpha=0.7, label=str(lab))

    plt.title("HDBSCAN-in-KMeans (global ids) on PCA(2D) of SVD space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=7, loc="best")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hierarchical_scatter_2d.png"), dpi=300)
    plt.close()


def _save_heatmap_hierarchical_top_codes(df_top_h_in_k: pd.DataFrame, hk_dir: str, top_n: int = 10):
    """
    Save a heatmap of top codes per hierarchical subcluster.
    One heatmap per KMeans cluster, to keep it readable.
    """
    plots_dir = os.path.join(hk_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = df_top_h_in_k[df_top_h_in_k["rank"] <= top_n].copy()
    if df.empty:
        return

    for k_id in sorted(df["kmeans_cluster"].unique()):
        d = df[df["kmeans_cluster"] == k_id].copy()
        if d.empty:
            continue

        subclusters = sorted(d["subcluster_local_id"].unique())
        codes = list(dict.fromkeys(d.sort_values(["subcluster_local_id", "rank"])["code"].tolist()))

        mat = np.zeros((len(subclusters), len(codes)), dtype=float)
        for i, sc in enumerate(subclusters):
            dd = d[d["subcluster_local_id"] == sc]
            for _, row in dd.iterrows():
                j = codes.index(row["code"])
                mat[i, j] = row["score"]

        plt.figure(figsize=(min(18, 1 + len(codes) * 0.5), 1 + len(subclusters) * 0.6))
        plt.imshow(mat, aspect="auto")
        plt.title(f"Top codes (distinctiveness) — HDBSCAN-in-KMeans (KMeans={k_id})")
        plt.yticks(range(len(subclusters)), [str(x) for x in subclusters])
        plt.xticks(range(len(codes)), codes, rotation=90)
        plt.colorbar(label="distinctiveness")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"top_codes_heatmap_kmeans_{k_id}.png"), dpi=300)
        plt.close()

def run_final(
    artifacts: dict,
    chosen_k: dict,
    chosen_hdb: dict,
    data_dir: str = "data",
    out_kmeans: str = "results/k_mean/final",
    out_hdbscan: str = "results/hdbscan/final",
    force: bool = False,
) -> dict:
    """
    Runs final KMeans and HDBSCAN pipelines using tuned parameters.
    Saves CSV + plots.

    Returns:
      dict with paths to main outputs.
    """
    os.makedirs(out_kmeans, exist_ok=True)
    os.makedirs(out_hdbscan, exist_ok=True)

    merged_path = "results/assignments_merged_final.csv"
    hk_dir = os.path.join(out_kmeans, "hdbscan_in_kmeans")
    hk_assign_path = os.path.join(hk_dir, "assignments_hdbscan_in_kmeans.csv")
    hk_top_path = os.path.join(hk_dir, "top_codes_hdbscan_in_kmeans.csv")
    hk_plots_dir = os.path.join(hk_dir, "plots")
    hk_scatter_path = os.path.join(hk_plots_dir, "hierarchical_scatter_2d.png")
    hk_any_bar_path = os.path.join(hk_plots_dir, "subcluster_sizes_kmeans_0.png") 

    kmeans_ok = os.path.exists(os.path.join(out_kmeans, "assignments.csv"))
    hdbscan_ok = os.path.exists(os.path.join(out_hdbscan, "assignments.csv"))
    merged_ok = os.path.exists(merged_path)
    hier_ok = os.path.exists(hk_assign_path) and os.path.exists(hk_top_path)
    hier_plots_ok = os.path.exists(hk_scatter_path) 

    # If hierarchical CSV exists but plots are missing, generate plots only (fast)
    if (not force) and hier_ok and (not hier_plots_ok):
        print("[FINAL] Hierarchical CSV found but plots missing -> generating plots only...")

        os.makedirs(hk_plots_dir, exist_ok=True)

        # Reload required data from existing files
        df_h_in_k = pd.read_csv(hk_assign_path)
        df_top_h_in_k = pd.read_csv(hk_top_path)
        summary_hk = pd.read_csv(os.path.join(hk_dir, "summary_hdbscan_in_kmeans.csv"))

        # Load X_reduced from preprocess artifacts (no need to rerun clustering)
        X_reduced = np.load(artifacts["x_path"]).astype(np.float32)

        _save_bar_subclusters(summary_hk, hk_dir)
        _save_scatter_hierarchical(X_reduced, df_h_in_k, hk_dir)
        _save_heatmap_hierarchical_top_codes(df_top_h_in_k, hk_dir, top_n=10)

        print(f"[FINAL] Hierarchical plots generated in: {hk_plots_dir}")

    if (not force) and kmeans_ok and hdbscan_ok and merged_ok and hier_ok and hier_plots_ok:
        return {
            "kmeans_dir": out_kmeans,
            "hdbscan_dir": out_hdbscan,
            "merged_assignments": merged_path,
            "hierarchical_dir": hk_dir,
        }

    # --- Read preprocessing config to stay consistent with tuning ---
    cfg = artifacts.get("config") or {}
    apply_tfidf = bool(cfg.get("apply_tfidf", True))
    n_components = int(cfg.get("n_components", 30))
    dim_reduction = cfg.get("dimensionality_reduction", "svd")

    k_opt = int(chosen_k["k_opt"])
    mcs = int(chosen_hdb["min_cluster_size_opt"])
    ms = chosen_hdb.get("min_samples_opt", None)
    if ms is not None:
        ms = int(ms)

    print("\n=== [FINAL] Configuration ===")
    print(f"Preprocess: TFIDF={apply_tfidf}, {dim_reduction} n_components={n_components}")
    print(f"KMeans: k_opt={k_opt}")
    print(f"HDBSCAN: min_cluster_size={mcs}, min_samples={ms}")
    print()

    # --- Load patients (parser cache ok) ---
    patients = FHIRParser.load_directory(data_dir, use_cache=True)
    if not patients:
        raise RuntimeError("No patients found for final run.")
    print(f"Patients loaded: {len(patients)}")

    systems = [CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM]

    # -------------------------
    # 1) Final KMeans
    # -------------------------
    print("\n[1/2] Final KMeans...")
    pipe_k = FHIRClusteringPipeline(
        include_systems=systems,
        apply_tfidf=apply_tfidf,
        dimensionality_reduction=dim_reduction,
        n_components=n_components,
        clustering_method="kmeans",
        n_clusters=k_opt,
    )
    pipe_k.fit(patients)

    df_assign_k = pipe_k.get_patient_assignments().rename(columns={"cluster_id": "kmeans_cluster"})
    df_assign_k.to_csv(os.path.join(out_kmeans, "assignments.csv"), index=False)

    df_sum_k = pipe_k.get_cluster_summary()
    df_sum_k.to_csv(os.path.join(out_kmeans, "summary.csv"), index=False)

    df_top_k = _flatten_top_codes(pipe_k.get_top_codes_per_cluster(top_n=15, method="distinctiveness"))
    df_top_k.to_csv(os.path.join(out_kmeans, "top_codes_distinctiveness.csv"), index=False)

    save_all_plots(pipe_k, output_dir=os.path.join(out_kmeans, "plots"))

    # -------------------------
    # 2) Final HDBSCAN
    # -------------------------
    print("\n[2/2] Final HDBSCAN...")
    pipe_h = FHIRClusteringPipeline(
        include_systems=systems,
        apply_tfidf=apply_tfidf,
        dimensionality_reduction=dim_reduction,
        n_components=n_components,
        clustering_method="hdbscan",
        n_clusters=None,
    )
    pipe_h.fit(patients, min_cluster_size=mcs, min_samples=ms)

    df_assign_h = pipe_h.get_patient_assignments().rename(columns={"cluster_id": "hdbscan_cluster"})
    df_assign_h.to_csv(os.path.join(out_hdbscan, "assignments.csv"), index=False)

    df_sum_h = pipe_h.get_cluster_summary()
    df_sum_h.to_csv(os.path.join(out_hdbscan, "summary.csv"), index=False)

    df_top_h = _flatten_top_codes(pipe_h.get_top_codes_per_cluster(top_n=15, method="distinctiveness"))
    df_top_h.to_csv(os.path.join(out_hdbscan, "top_codes_distinctiveness.csv"), index=False)

    save_all_plots(pipe_h, output_dir=os.path.join(out_hdbscan, "plots"))

    # -------------------------
    # 2bis) HDBSCAN inside each KMeans cluster (hierarchical)
    # -------------------------
    print("\n[2bis/3] HDBSCAN inside each KMeans cluster...")

    X_reduced = pipe_k.reduced_data
    labels_k = pipe_k.cluster_labels
    if X_reduced is None or labels_k is None:
        raise RuntimeError("KMeans must have reduced_data and cluster_labels for hierarchical HDBSCAN.")

    interpreter = pipe_k.interpreter
    if interpreter is None:
        raise RuntimeError("KMeans pipeline interpreter is missing.")

    patient_ids = df_assign_k["patient_id"].to_numpy()

    # Use tuned HDBSCAN params (same as global), but applied locally
    df_h_in_k, df_top_h_in_k = _run_hdbscan_in_kmeans(
        X=X_reduced,
        kmeans_labels=labels_k,
        patient_ids=patient_ids,
        interpreter=interpreter,
        min_cluster_size=mcs,
        min_samples=ms,
        top_n=15,
    )

    # Save hierarchical outputs under KMeans final folder
    hk_dir = os.path.join(out_kmeans, "hdbscan_in_kmeans")
    os.makedirs(hk_dir, exist_ok=True)

    df_h_in_k.to_csv(os.path.join(hk_dir, "assignments_hdbscan_in_kmeans.csv"), index=False)
    df_top_h_in_k.to_csv(os.path.join(hk_dir, "top_codes_hdbscan_in_kmeans.csv"), index=False)

    # Optional: simple summary (counts)
    summary_hk = (
        df_h_in_k.groupby(["kmeans_cluster", "hdbscan_in_kmeans_local"])
        .size()
        .reset_index(name="n_patients")
        .sort_values(["kmeans_cluster", "n_patients"], ascending=[True, False])
    )
    summary_hk.to_csv(os.path.join(hk_dir, "summary_hdbscan_in_kmeans.csv"), index=False)
    _save_bar_subclusters(summary_hk, hk_dir)
    _save_scatter_hierarchical(X_reduced, df_h_in_k, hk_dir)
    _save_heatmap_hierarchical_top_codes(df_top_h_in_k, hk_dir, top_n=10)

    print(f"Saved hierarchical HDBSCAN outputs to: {hk_dir}")

    # -------------------------
    # 3) Merge assignments
    # -------------------------
    merged = (
    df_assign_k
    .merge(df_assign_h, on="patient_id", how="left")
    .merge(df_h_in_k[["patient_id", "hdbscan_in_kmeans_local", "hdbscan_in_kmeans_global"]], on="patient_id", how="left")
    )
    os.makedirs("results", exist_ok=True)
    merged.to_csv(merged_path, index=False)
    print(f"\nSaved merged assignments: {merged_path}")

    return {
        "kmeans_dir": out_kmeans,
        "hdbscan_dir": out_hdbscan,
        "merged_assignments": merged_path,
        "k_opt": k_opt,
        "hdbscan_params": {"min_cluster_size": mcs, "min_samples": ms},
    }