"""
Benchmark de sensibilité pour DBSCAN.
Génère une Heatmap montrant le Score Silhouette en fonction des hyperparamètres.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline

def run_sensitivity_analysis(data_dir="data", output_dir="results/benchmarks"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Chargement et Réduction de dimension (SVD)
    # On utilise le pipeline juste pour préparer les données (pas pour clusteriser tout de suite)
    print("Chargement des données pour analyse de sensibilité...")
    patients = FHIRParser.load_directory(data_dir, use_cache=True)
    if not patients:
        print("Aucun patient trouvé.")
        return

    # --- CORRECTION ICI ---
    # On configure tout dans le constructeur (__init__)
    pipeline = FHIRClusteringPipeline(
        include_systems=None, # All
        apply_tfidf=True,
        dimensionality_reduction='svd',
        n_components=50,
        clustering_method='kmeans', # Défini ici
        n_clusters=2                # Défini ici (juste pour que le fit fonctionne)
    )
    
    # On appelle fit sans arguments conflictuels
    pipeline.fit(patients) 
    # ----------------------

    X_reduced = pipeline.reduced_data
    
    if X_reduced is None:
        print("Erreur lors de la réduction de dimension.")
        return

    # 2. Grid Search pour DBSCAN
    print("Lancement du Grid Search DBSCAN...")
    
    # Plages de paramètres à tester
    eps_range = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    min_samples_range = [5, 10, 20, 30, 50, 100]
    
    results = []
    
    total_iter = len(eps_range) * len(min_samples_range)
    curr = 0
    
    for eps in eps_range:
        for ms in min_samples_range:
            curr += 1
            print(f"Test {curr}/{total_iter}: eps={eps}, min_samples={ms}", end="\r")
            
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_reduced)
            
            # Calcul des métriques
            unique_labels = set(labels)
            # Compter les clusters (hors bruit -1)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Silhouette n'est valide que si > 1 cluster et < N points
            score = -1.0
            if n_clusters > 1 and n_clusters < len(X_reduced):
                # On échantillonne pour aller plus vite si dataset énorme
                sample_size = min(10000, len(X_reduced))
                try:
                    score = silhouette_score(X_reduced, labels, sample_size=sample_size, random_state=42)
                except Exception:
                    score = -1.0
            
            results.append({
                "eps": eps,
                "min_samples": ms,
                "silhouette": score,
                "n_clusters": n_clusters,
                "noise_ratio": n_noise / len(X_reduced)
            })

    print("\nGénération de la Heatmap...")
    df = pd.DataFrame(results)
    
    # 3. Plotting
    if df.empty:
        print("Aucun résultat généré.")
        return

    # Pivot pour la heatmap
    pivot_table = df.pivot(index="min_samples", columns="eps", values="silhouette")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Sensibilité DBSCAN (Score Silhouette)\nZone jaune = Clusters bien définis")
    plt.ylabel("Min Samples")
    plt.xlabel("Epsilon (Distance)")
    
    output_path = os.path.join(output_dir, "dbscan_sensitivity_heatmap.png")
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap sauvegardée : {output_path}")
    
    # Sauvegarde CSV des résultats bruts
    df.to_csv(os.path.join(output_dir, "sensitivity_results.csv"), index=False)

if __name__ == "__main__":
    run_sensitivity_analysis()