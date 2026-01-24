"""
Single entry point for the full POC pipeline.

Run:
  python -u run_poc.py
"""

from workflows.pre_process import run_preprocess
from workflows.tune_kmeans import run_kmeans_tuning
from workflows.tune_hdbscan import run_hdbscan_tuning

from workflows.run_final import run_final


def main():
    artifacts = run_preprocess(force=False)
    chosen_k = run_kmeans_tuning(artifacts, force=False)
    chosen_hdb = run_hdbscan_tuning(artifacts, force=False)

    print("\n=== Decisions ===")
    print("KMeans:", chosen_k)
    print("HDBSCAN:", chosen_hdb)

    run_final(artifacts, chosen_k, chosen_hdb)


if __name__ == "__main__":
    main()