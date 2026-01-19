import os
import pandas as pd
from fhir_clustering.fhir_parser import FHIRParser  # Le nouveau fichier ci-dessus
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem
from fhir_clustering.visualization import save_all_plots

# 1. Configuration
DATA_DIR = "data"  # Dossier contenant vos JSON Synthea
OUTPUT_DIR = "results"
N_CLUSTERS = 3     # Nombre de groupes à trouver (à ajuster selon nombre de fichiers)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # --- CHARGEMENT DES DONNÉES ---
    if not os.path.exists(DATA_DIR):
        print(f"Créez un dossier '{DATA_DIR}' et placez-y vos fichiers JSON Synthea.")
        return

    patients = FHIRParser.load_directory(DATA_DIR)
    
    if not patients:
        print("Aucun patient trouvé.")
        return

    print(f"\nExemple de patient chargé : {patients[0].patient_id}")
    print(f"Codes trouvés : {[str(c) for c in list(patients[0].codes)[:5]]} ...")

    # --- PIPELINE IA (Moteur A) ---
    print("\nLancement du Pipeline de Clustering...")
    pipeline = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=True,              # Réduit le poids des codes trop fréquents (ex: "Routine checkup")
        dimensionality_reduction='svd',# Indispensable pour la sparsité
        n_components=30,               # Réduire à 30 dimensions
        clustering_method='kmeans',    # Ou 'dbscan' pour détection auto
        n_clusters=N_CLUSTERS,
        min_df=0.01,                   # Filtrer les codes présents dans moins de 1% des patients
        max_df=0.9                     # Filtrer les codes présents dans plus de 90
    )

    # Entraînement
    pipeline.fit(patients)

    # --- RÉSULTATS & INTERPRÉTATION ---
    print("\n--- Analyse des Clusters (Top Codes) ---")
    
    # On récupère les codes les plus "distinctifs" pour chaque groupe
    # 'distinctiveness' est mieux que 'frequency' car il ignore les codes communs à tous
    top_codes = pipeline.get_top_codes_per_cluster(top_n=5, method='distinctiveness')

    for cluster_id, codes in top_codes.items():
        print(f"\nCluster {cluster_id}:")
        for code_name, score in codes:
            # Le format est "SYSTEM:Code", on peut lire le display dans le parser si on veut mieux
            print(f"  - {code_name} (Score: {score:.4f})")

    # Sauvegarde des stats
    summary = pipeline.get_cluster_summary()
    print("\nRésumé statistique :")
    print(summary)

    # Sauvegarde des assignations (quel patient est dans quel groupe)
    assignments = pipeline.get_patient_assignments()
    assignments.to_csv(f"{OUTPUT_DIR}/assignments.csv", index=False)
    print(f"\nAssignations sauvegardées dans {OUTPUT_DIR}/assignments.csv")

    # Génération des graphiques
    print("Génération des graphiques...")
    save_all_plots(pipeline, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()