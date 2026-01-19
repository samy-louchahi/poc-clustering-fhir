"""
Benchmark different clustering methods (KMeans, DBSCAN, HDBSCAN)
using shared preprocessing (TF-IDF + SVD) to compare key metrics.

Run with: python compare_clustering_methods.py
Outputs CSV and console summary in results/clustering_benchmark.
"""
print("has started ")
import os
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem


DATA_DIR = "data"
OUTPUT_DIR = "results/clustering_benchmark"

# Shared preprocessing configuration
APPLY_TFIDF = True
DIM_REDUCTION = "svd"
N_COMPONENTS = 30
N_CLUSTERS_KMEANS = 3

# Clustering experiments (init params go into pipeline ctor; fit params go into pipeline.fit)
EXPERIMENTS = [
    {
        "name": "kmeans",
        "init_params": {"n_clusters": N_CLUSTERS_KMEANS},
        "fit_params": {},
        "notes": "Baseline KMeans with k set",
    },
    {
        "name": "dbscan",
        "init_params": {"n_clusters": None},
        "fit_params": {"eps": 0.6, "min_samples": 4},
        "notes": "Density-based; auto cluster count",
    },
    {
        "name": "hdbscan",
        "init_params": {"n_clusters": None},
        "fit_params": {"min_cluster_size": 5, "min_samples": None},
        "notes": "Hierarchical DBSCAN (skips if package missing)",
    },
]


def _safe_metric(metric_fn, X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Compute metric ignoring noise and ensuring >=2 clusters."""
    mask = labels != -1
    if mask.sum() < 2:
        return None
    filtered_labels = labels[mask]
    if len(np.unique(filtered_labels)) < 2:
        return None
    try:
        return float(metric_fn(X[mask], filtered_labels))
    except Exception:
        return None
    
def _has_hdbscan() -> bool:
    try:
        import hdbscan  
        return True
    except Exception:
        return False

def run_benchmark() -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    HDBSCAN_AVAILABLE = _has_hdbscan()

    patients = FHIRParser.load_directory(DATA_DIR)
    if not patients:
        print(f"No patients found in '{DATA_DIR}'.")
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []

    print("\n=== Clustering benchmark ===")
    print(f"Patients: {len(patients)}")

    for exp in EXPERIMENTS:
        method = exp["name"]
        init_params = exp["init_params"]
        fit_params = exp["fit_params"]

        if method == "hdbscan" and not HDBSCAN_AVAILABLE:
            print("- hdbscan skipped (package not installed). Install with: pip install hdbscan")
            results.append({
                "method": method,
                "status": "skipped_missing_dep",
                "silhouette": None,
                "davies_bouldin": None,
                "calinski_harabasz": None,
                "n_clusters": None,
                "noise_points": None,
                "runtime_sec": None,
                "notes": exp.get("notes"),
            })
            continue

        print(f"- Running {method} ...")
        start = time.perf_counter()

        pipeline = FHIRClusteringPipeline(
            include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
            apply_tfidf=APPLY_TFIDF,
            dimensionality_reduction=DIM_REDUCTION,
            n_components=N_COMPONENTS,
            clustering_method=method,
            n_clusters=init_params.get("n_clusters"),
        )

        pipeline.fit(patients, **fit_params)

        labels = pipeline.get_cluster_labels()
        X = pipeline.reduced_data if pipeline.reduced_data is not None else pipeline.transformed_matrix

        silhouette = _safe_metric(silhouette_score, X, labels)
        davies_bouldin = _safe_metric(davies_bouldin_score, X, labels)
        calinski_harabasz = _safe_metric(calinski_harabasz_score, X, labels)

        runtime = time.perf_counter() - start
        n_clusters = pipeline.clusterer.get_n_clusters()
        noise_points = int(np.sum(labels == -1))

        results.append({
            "method": method,
            "status": "ok",
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "n_clusters": n_clusters,
            "noise_points": noise_points,
            "runtime_sec": runtime,
            "notes": exp.get("notes"),
        })

        print(
            f"  n_clusters={n_clusters} | noise={noise_points} | "
            f"silhouette={silhouette} | runtime={runtime:.2f}s"
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df)

    return df


if __name__ == "__main__":
    run_benchmark()
