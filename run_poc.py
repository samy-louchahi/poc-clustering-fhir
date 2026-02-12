"""
Single entry point for the full POC pipeline.

Run:
  python -u run_poc.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Workflow imports
from workflows.pre_process import run_preprocess
from workflows.tune_kmeans import run_kmeans_tuning
from workflows.tune_hdbscan import run_hdbscan_tuning
from workflows.run_final import run_final
from workflows.terminology_layer import build_terminology_layer

# Visualization imports
from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem
from fhir_clustering.visualization import plot_biplot
from fhir_clustering.dimensionality_reduction import DimensionalityReducer

def _run_visual_tests(data_dir: str, out_dir: str):
    """
    Runs experimental visualizations: Biplot (SVD) and UMAP.
    Requires 'raw_codes' context to access feature names for Biplot.
    """
    print(f"\n[VIZ] Running experimental visualizations in {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Patients (Fast if cached)
    patients = FHIRParser.load_directory(data_dir, use_cache=True)
    if not patients:
        print("[VIZ] No patients found, skipping plots.")
        return

    # 2. Setup Pipeline (SVD is required for Biplot)
    # We use a standard config: TF-IDF + SVD (30 components)
    print("[VIZ] Fitting SVD pipeline for Biplot...")
    pipeline = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=True,
        dimensionality_reduction="svd",
        n_components=30,
        clustering_method="kmeans", # Dummy, we just want the reduction
        n_clusters=2
    )
    pipeline.fit(patients)

    # --- A. BIPLOT (SVD) ---
    print("[VIZ] Generating Biplot...")
    try:
        fig, ax = plt.subplots(figsize=(14, 12))
        plot_biplot(pipeline, ax=ax, top_n_arrows=15)
        
        # Save
        biplot_path = os.path.join(out_dir, "biplot_interpretation_svd.png")
        plt.tight_layout()
        plt.savefig(biplot_path, dpi=300)
        plt.close()
        print(f"  -> Saved: {biplot_path}")
    except Exception as e:
        print(f"[VIZ] Error generating Biplot: {e}")

    # --- B. UMAP PROJECTION ---
    print("[VIZ] Running UMAP projection...")
    try:
        # We reuse the transformed matrix (TF-IDF) from the pipeline
        # UMAP is non-linear, usually 2 components for visualization
        reducer_umap = DimensionalityReducer(method='umap', n_components=2)
        
        # This might take a moment
        X_umap = reducer_umap.fit_transform(pipeline.transformed_matrix)
        
        # Simple plot (without labels, just density structure)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], s=5, alpha=0.6, c='royalblue')
        plt.title("UMAP Projection (Structure Locale)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True, alpha=0.2)
        
        umap_path = os.path.join(out_dir, "umap_projection.png")
        plt.savefig(umap_path, dpi=300)
        plt.close()
        print(f"  -> Saved: {umap_path}")

    except ImportError:
        print("[VIZ] UMAP not installed (pip install umap-learn). Skipping.")
    except Exception as e:
        print(f"[VIZ] Error generating UMAP: {e}")


def main():
    # 0) Build terminology layer (needed for domain_rollup)
    build_terminology_layer(
        data_dir="data",
        omop_dir="terminology_omop",
        out_dir="results/terminology",
        force=False,
        max_ancestor_distance=3,
    )

    # ============================================================
    # 1) RAW CODES
    # ============================================================
    print("\n================ RAW CODES ================\n")
    artifacts_raw = run_preprocess(
        force=False,
        feature_mode="raw_codes",
        out_dir="results/pre_process",
    )
    chosen_k_raw = run_kmeans_tuning(artifacts_raw, force=False)
    chosen_hdb_raw = run_hdbscan_tuning(artifacts_raw, force=False)

    print("\n=== Decisions (RAW) ===")
    print("KMeans:", chosen_k_raw)
    print("HDBSCAN:", chosen_hdb_raw)

    run_final(
        artifacts_raw,
        chosen_k_raw,
        chosen_hdb_raw,
        out_kmeans="results/raw_codes/k_mean/final",
        out_hdbscan="results/raw_codes/hdbscan/final",
        force=False,
    )

    # --- NEW: Run Experimental Visualizations (Biplot / UMAP) ---
    _run_visual_tests(
        data_dir="data", 
        out_dir="results/raw_codes/plots_experimental"
    )
    # ------------------------------------------------------------

    # ============================================================
    # 2) DOMAIN ROLLUP
    # ============================================================
    print("\n================ DOMAIN ROLLUP ================\n")
    artifacts_dom = run_preprocess(
        force=False,
        feature_mode="domain_rollup",
        out_dir="results/pre_process",
        terminology_pkl="results/terminology/terminology.pkl",
    )
    chosen_k_dom = run_kmeans_tuning(artifacts_dom, force=False)
    chosen_hdb_dom = run_hdbscan_tuning(artifacts_dom, force=False)

    print("\n=== Decisions (DOMAIN) ===")
    print("KMeans:", chosen_k_dom)
    print("HDBSCAN:", chosen_hdb_dom)

    run_final(
        artifacts_dom,
        chosen_k_dom,
        chosen_hdb_dom,
        out_kmeans="results/domain_rollup/k_mean/final",
        out_hdbscan="results/domain_rollup/hdbscan/final",
        force=False,
    )

    print("\nDone. RAW + DOMAIN final runs completed.")


if __name__ == "__main__":
    main()