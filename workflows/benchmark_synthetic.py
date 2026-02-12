"""
Benchmark visuel sur données synthétiques (Sanity Check).
Objectif : Montrer la capacité des algos à comprendre la géométrie des données.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import hdbscan

def run_synthetic_benchmark(output_dir="results/benchmarks"):
    os.makedirs(output_dir, exist_ok=True)
    n_samples = 1500
    random_state = 170
    
    # --- 1. Génération des Datasets ---
    # Cas A : Blobs simples (Gaussiennes) - Terrain de jeu idéal pour K-Means
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    
    # Cas B : Varied density (Densités variables) - Difficile pour DBSCAN standard
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    
    # Cas C : Anisotropic (Allongés) - K-Means échoue ici (il cherche des sphères)
    X_aniso, y_aniso = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    aniso = (X_aniso, y_aniso)
    
    # Cas D : Noisy Moons (Non-linéaire) - Impossible pour K-Means
    moons = datasets.make_moons(n_samples=n_samples, noise=0.05)

    datasets_list = [
        (blobs, "Blobs (Sphériques)"),
        (varied, "Densités Variables"),
        (aniso, "Anisotropique (Allongé)"),
        (moons, "Non-linéaire (Croissants)")
    ]

    # --- 2. Configuration des Algos ---
    # On fixe les paramètres pour l'exemple visuel
    kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42)
    dbscan = DBSCAN(eps=0.3, min_samples=10) 
    hdb = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)

    algos = [
        ("KMeans", kmeans),
        ("DBSCAN", dbscan),
        ("HDBSCAN", hdb)
    ]

    # --- 3. Exécution et Plotting ---
    print("Génération du benchmark synthétique...")
    fig, axes = plt.subplots(len(datasets_list), len(algos), figsize=(15, 12))

    for i, (dataset, dataset_name) in enumerate(datasets_list):
        X, y = dataset
        # Important : Normalisation pour les algos de densité
        X = StandardScaler().fit_transform(X)

        for j, (algo_name, algorithm) in enumerate(algos):
            # Clonage ou réinstanciation si nécessaire (ici simple fit)
            # Pour DBSCAN sur moons, on ajuste un peu l'eps sinon c'est un seul cluster
            if algo_name == "DBSCAN" and "Croissants" in dataset_name:
                algorithm = DBSCAN(eps=0.3, min_samples=5)
            
            try:
                if hasattr(algorithm, 'fit_predict'):
                    y_pred = algorithm.fit_predict(X)
                else:
                    algorithm.fit(X)
                    y_pred = algorithm.labels_
            except Exception as e:
                print(f"Erreur {algo_name} sur {dataset_name}: {e}")
                y_pred = np.zeros(len(X))

            # Plot
            ax = axes[i, j]
            # Les points non classés (-1) en gris, les autres en couleur
            colors = np.array([x for x in 'bgrcmykbgrcmyk'])
            colors = np.hstack([colors] * 20)
            
            ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap='viridis')
            
            if i == 0:
                ax.set_title(algo_name, size=14, weight='bold')
            if j == 0:
                ax.set_ylabel(dataset_name, size=12, weight='bold')
            
            ax.set_xticks(())
            ax.set_yticks(())

    plt.tight_layout()
    output_path = os.path.join(output_dir, "synthetic_geometry_benchmark.png")
    plt.savefig(output_path, dpi=300)
    print(f"Graphique sauvegardé : {output_path}")

if __name__ == "__main__":
    run_synthetic_benchmark()