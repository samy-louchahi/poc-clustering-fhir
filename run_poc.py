"""
Single entry point for the full POC pipeline.

Run:
  python -u run_poc.py
"""

from workflows.pre_process import run_preprocess
from workflows.tune_kmeans import run_kmeans_tuning
from workflows.tune_hdbscan import run_hdbscan_tuning
from workflows.run_final import run_final
from workflows.terminology_layer import build_terminology_layer


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