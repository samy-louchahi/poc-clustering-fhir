"""
POC: Compare top codes for
1) KMeans clusters (macro)
2) HDBSCAN clusters (micro)
3) HDBSCAN inside each KMeans cluster (hierarchical)

Outputs in: results/hierarchical_top_codes/
- top_codes_kmeans.csv
- top_codes_hdbscan_global.csv
- top_codes_hdbscan_in_kmeans.csv
- assignments_merged.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import hdbscan

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem


DATA_DIR = "data"
OUTPUT_DIR = "results/hierarchical_top_codes"

# Shared preprocessing
N_COMPONENTS = 30
APPLY_TFIDF = True
DIM_REDUCTION = "svd"

# KMeans
N_CLUSTERS_KMEANS = 3

# HDBSCAN params
HDB_MIN_CLUSTER_SIZE = 10
HDB_MIN_SAMPLES = None

TOP_N = 10


def top_codes_to_df(top_codes: dict, algo: str) -> pd.DataFrame:
    rows = []
    for cid, codes in top_codes.items():
        for rank, (code, score) in enumerate(codes, start=1):
            rows.append({
                "algo": algo,
                "cluster_id": int(cid),
                "rank": rank,
                "code": code,
                "score": float(score),
            })
    return pd.DataFrame(rows)

def save_cluster_sizes(labels: np.ndarray, title: str, path: str):
    """Bar chart of cluster sizes (including noise if present)."""
    uniq, counts = np.unique(labels, return_counts=True)
    # sort, put noise (-1) first if present
    order = np.argsort(uniq)
    uniq = uniq[order]
    counts = counts[order]

    plt.figure()
    plt.bar([str(u) for u in uniq], counts)
    plt.title(title)
    plt.xlabel("cluster_id")
    plt.ylabel("n_patients")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_2d_scatter(X: np.ndarray, labels: np.ndarray, title: str, path: str, noise_label: int = -1):
    """
    2D projection with PCA (fast) and scatter colored by labels.
    Noise gets its own color by default (still OK for a POC).
    """
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure()
    # plot noise first (if any) to not hide clusters
    noise_mask = labels == noise_label
    if noise_mask.any():
        plt.scatter(X2[noise_mask, 0], X2[noise_mask, 1], s=8, alpha=0.4, label="noise")

    # plot each cluster
    for c in sorted([x for x in np.unique(labels) if x != noise_label]):
        m = labels == c
        plt.scatter(X2[m, 0], X2[m, 1], s=10, alpha=0.7, label=str(c))

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_top_codes_heatmap(df_top: pd.DataFrame, title: str, path: str, top_n: int = 10):
    """
    Heatmap-like plot: clusters on y, codes on x, value=score.
    Uses matplotlib imshow to avoid seaborn.
    """
    # keep only top_n ranks
    df = df_top[df_top["rank"] <= top_n].copy()
    clusters = sorted(df["cluster_id"].unique())
    codes = list(dict.fromkeys(df.sort_values(["cluster_id", "rank"])["code"].tolist()))  # preserve order

    mat = np.zeros((len(clusters), len(codes)), dtype=float)
    for i, c in enumerate(clusters):
        sub = df[df["cluster_id"] == c]
        for _, row in sub.iterrows():
            j = codes.index(row["code"])
            mat[i, j] = row["score"]

    plt.figure(figsize=(min(16, 1 + len(codes) * 0.6), 1 + len(clusters) * 0.6))
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.yticks(range(len(clusters)), [str(c) for c in clusters])
    plt.xticks(range(len(codes)), codes, rotation=90)
    plt.colorbar(label="distinctiveness score")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_flow_table(df_merged: pd.DataFrame, path: str):
    """
    Create a flow table (KMeans -> HDBSCAN global) for Sankey/analysis.
    """
    flow = (
        df_merged.groupby(["kmeans_cluster", "hdbscan_global_cluster"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    flow.to_csv(path, index=False)

KMEANS_LABELS = {
    0: "Suivi biologique intensif",
    1: "Surveillance clinique modérée",
    2: "Profil mixte complexe",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------
    # Load patients
    # -----------------------
    patients = FHIRParser.load_directory(DATA_DIR, use_cache=True)
    if not patients:
        print("No patients found.")
        return
    print(f"Patients loaded: {len(patients)}")

    # -----------------------
    # 1) KMeans (macro)
    # -----------------------
    print("\n[1] KMeans...")
    pipe_k = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=APPLY_TFIDF,
        dimensionality_reduction=DIM_REDUCTION,
        n_components=N_COMPONENTS,
        clustering_method="kmeans",
        n_clusters=N_CLUSTERS_KMEANS,
    )
    pipe_k.fit(patients)

    X = pipe_k.reduced_data
    labels_k = pipe_k.cluster_labels
    if X is None or labels_k is None:
        raise RuntimeError("KMeans pipeline did not produce reduced_data/cluster_labels.")

    df_assign_k = pipe_k.get_patient_assignments().rename(columns={"cluster_id": "kmeans_cluster"})
    df_assign_k.to_csv(os.path.join(OUTPUT_DIR, "assignments_kmeans.csv"), index=False)
    df_assign_k["kmeans_label"] = df_assign_k["kmeans_cluster"].map(KMEANS_LABELS).fillna("Unknown")
    df_assign_k.to_csv(os.path.join(OUTPUT_DIR, "assignments_kmeans.csv"), index=False)

    top_k = pipe_k.get_top_codes_per_cluster(top_n=TOP_N, method="distinctiveness")
    df_top_k = top_codes_to_df(top_k, "kmeans")
    df_top_k.to_csv(os.path.join(OUTPUT_DIR, "top_codes_kmeans.csv"), index=False)

    # --- Plots: KMeans ---
    save_cluster_sizes(labels_k, "KMeans cluster sizes", os.path.join(OUTPUT_DIR, "kmeans_cluster_sizes.png"))
    save_2d_scatter(X, labels_k, "KMeans clusters (PCA 2D on SVD space)", os.path.join(OUTPUT_DIR, "kmeans_scatter_2d.png"))
    save_top_codes_heatmap(df_top_k, "KMeans top codes (distinctiveness)", os.path.join(OUTPUT_DIR, "kmeans_top_codes_heatmap.png"), top_n=TOP_N)

    # -------------------------
    # 2) HDBSCAN global (micro)
    # -------------------------
    print("\n[2] HDBSCAN global...")
    pipe_h = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=APPLY_TFIDF,
        dimensionality_reduction=DIM_REDUCTION,
        n_components=N_COMPONENTS,
        clustering_method="hdbscan",
        n_clusters=None,
    )
    pipe_h.fit(patients, min_cluster_size=HDB_MIN_CLUSTER_SIZE, min_samples=HDB_MIN_SAMPLES)

    labels_h = pipe_h.cluster_labels
    df_assign_h = pipe_h.get_patient_assignments().rename(columns={"cluster_id": "hdbscan_global_cluster"})
    df_assign_h.to_csv(os.path.join(OUTPUT_DIR, "assignments_hdbscan_global.csv"), index=False)

    top_h = pipe_h.get_top_codes_per_cluster(top_n=TOP_N, method="distinctiveness")
    df_top_h = top_codes_to_df(top_h, "hdbscan_global")
    df_top_h.to_csv(os.path.join(OUTPUT_DIR, "top_codes_hdbscan_global.csv"), index=False)

    # --- Plots: HDBSCAN global ---
    save_cluster_sizes(labels_h, "HDBSCAN global cluster sizes (incl. noise=-1)", os.path.join(OUTPUT_DIR, "hdbscan_global_cluster_sizes.png"))
    save_2d_scatter(X, labels_h, "HDBSCAN global clusters (PCA 2D on SVD space)", os.path.join(OUTPUT_DIR, "hdbscan_global_scatter_2d.png"))
    save_top_codes_heatmap(df_top_h, "HDBSCAN global top codes (distinctiveness)", os.path.join(OUTPUT_DIR, "hdbscan_global_top_codes_heatmap.png"), top_n=TOP_N)

    # -----------------------
    # 3) HDBSCAN inside each KMeans cluster
    # -----------------------
    print("\n[3] HDBSCAN inside each KMeans cluster...")
    interpreter = pipe_k.interpreter
    if interpreter is None:
        raise RuntimeError("Interpreter missing in KMeans pipeline.")

    patient_ids = df_assign_k["patient_id"].to_numpy()
    X = np.asarray(X, dtype=np.float32)

    hierarchical_assign = []
    hierarchical_top_tables = []

    for k_id in sorted(np.unique(labels_k)):
        idx = np.where(labels_k == k_id)[0]
        X_sub = X[idx]

        print(f"  - KMeans cluster {k_id}: n={len(idx)}")

        # if too small, don't subcluster
        if len(idx) < max(2 * HDB_MIN_CLUSTER_SIZE, 25):
            for g in idx:
                hierarchical_assign.append({
                    "patient_id": patient_ids[g],
                    "kmeans_cluster": int(k_id),
                    "hdbscan_in_kmeans": -1,
                })
            continue

        h = hdbscan.HDBSCAN(
            min_cluster_size=HDB_MIN_CLUSTER_SIZE,
            min_samples=HDB_MIN_SAMPLES,
        )
        sub_labels = h.fit_predict(X_sub).astype(int)

        # assignments
        for local_i, global_i in enumerate(idx):
            hierarchical_assign.append({
                "patient_id": patient_ids[global_i],
                "kmeans_cluster": int(k_id),
                "hdbscan_in_kmeans": int(sub_labels[local_i]),
            })

        # Build full-length label vector for interpreter
        # Offset to avoid collisions across parent KMeans clusters
        offset = int(k_id) * 1000
        full_labels = np.full(shape=(len(patient_ids),), fill_value=-1, dtype=int)

        for local_i, global_i in enumerate(idx):
            if sub_labels[local_i] != -1:
                full_labels[global_i] = offset + int(sub_labels[local_i])

        top_sub = interpreter.get_top_codes_per_cluster(full_labels, top_n=TOP_N, method="distinctiveness")
        df_sub = top_codes_to_df(top_sub, algo=f"hdbscan_in_kmeans_{k_id}")
        df_sub["kmeans_cluster"] = int(k_id)
        df_sub["subcluster_global_id"] = df_sub["cluster_id"]
        df_sub["subcluster_local_id"] = df_sub["cluster_id"].apply(lambda x: int(x) - offset)
        hierarchical_top_tables.append(df_sub)

    df_hier_assign = pd.DataFrame(hierarchical_assign)
    df_hier_assign.to_csv(os.path.join(OUTPUT_DIR, "assignments_hdbscan_in_kmeans.csv"), index=False)

    if hierarchical_top_tables:
        df_hier_top = pd.concat(hierarchical_top_tables, ignore_index=True)
    else:
        df_hier_top = pd.DataFrame()

    df_hier_top.to_csv(os.path.join(OUTPUT_DIR, "top_codes_hdbscan_in_kmeans.csv"), index=False)

    # -----------------------
    # 4) Merge assignments (for analysis/figures)
    # -----------------------
    df_merged = (
        df_assign_k
        .merge(df_assign_h, on="patient_id", how="left")
        .merge(df_hier_assign, on=["patient_id", "kmeans_cluster"], how="left")
    )
    df_merged.to_csv(os.path.join(OUTPUT_DIR, "assignments_merged.csv"), index=False)
    df_merged["kmeans_label"] = df_merged["kmeans_cluster"].map(KMEANS_LABELS).fillna("Unknown")
    df_merged.to_csv(os.path.join(OUTPUT_DIR, "assignments_merged.csv"), index=False)

    print("\nDone.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()