"""
Compare the impact of different n_components values on
FHIR patient clustering quality and stability.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, adjusted_rand_score

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "results/n_components_comparison"

N_COMPONENTS_LIST = [10, 20, 30, 50, 75, 100]
N_CLUSTERS = 3


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chargement des patients FHIR...")
    patients = FHIRParser.load_directory(DATA_DIR)

    if not patients:
        print("Aucun patient trouvé.")
        return

    results = []
    previous_labels = None

    print("\nComparaison des valeurs de n_components...\n")

    for n_components in N_COMPONENTS_LIST:
        print(f"→ n_components = {n_components}")

        pipeline = FHIRClusteringPipeline(
            include_systems=[
                CodeSystem.SNOMED,
                CodeSystem.LOINC,
                CodeSystem.RXNORM,
            ],
            apply_tfidf=True,
            dimensionality_reduction="svd",
            n_components=n_components,
            clustering_method="kmeans",
            n_clusters=N_CLUSTERS,
        )

        pipeline.fit(patients)

        X_reduced = pipeline.reduced_data
        labels = pipeline.cluster_labels

        silhouette = silhouette_score(X_reduced, labels)
        explained_variance = pipeline.dim_reducer.get_cumulative_variance()
        explained_curve = pipeline.dim_reducer.get_cumulative_variance()
        explained_final = float(explained_curve[-1])

        ari = None
        if previous_labels is not None:
            ari = adjusted_rand_score(previous_labels, labels)

        results.append({
            "n_components": n_components,
            "explained_variance": explained_final,
            "silhouette_score": silhouette,
            "ari_stability": ari,
        })

        previous_labels = labels

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False)

    print("\nRésultats sauvegardés dans :", csv_path)
    print(df)

    generate_plots(df)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def generate_plots(df: pd.DataFrame):
    # Variance expliquée
    plt.figure()
    plt.plot(df["n_components"], df["explained_variance"], marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Explained variance")
    plt.title("Explained variance vs n_components")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/explained_variance.png")
    plt.close()

    # Silhouette score
    plt.figure()
    plt.plot(df["n_components"], df["silhouette_score"], marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs n_components")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/silhouette_score.png")
    plt.close()

    # Stabilité (ARI)
    plt.figure()
    plt.plot(df["n_components"], df["ari_stability"], marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Adjusted Rand Index (stability)")
    plt.title("Cluster stability vs n_components")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/stability_ari.png")
    plt.close()

    print("Graphiques sauvegardés dans", OUTPUT_DIR)


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()