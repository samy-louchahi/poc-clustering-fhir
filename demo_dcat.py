import os
import pandas as pd
# On importe le module (attention au nom sans .py)
from workflows.dcat_export import CohortSelector, DCATGenerator

def main():
    # Chemins vers les résultats (Adaptez si vous utilisez hdbscan)
    BASE_RES_DIR = "results/k_mean/final"
    TOP_CODES_PATH = os.path.join(BASE_RES_DIR, "top_codes_distinctiveness.csv")
    SUMMARY_PATH = os.path.join(BASE_RES_DIR, "summary.csv")
    ASSIGNMENTS_PATH = os.path.join(BASE_RES_DIR, "assignments.csv") # <--- NOUVEAU
    
    if not os.path.exists(TOP_CODES_PATH):
        print(f"Erreur: Résultats introuvables dans {BASE_RES_DIR}. Lancez run_poc.py d'abord.")
        return

    # 1. Scénario Utilisateur
    query_description = "Patients Diabétiques sous Metformine"
    # Remplacez par des codes présents dans vos top_codes_distinctiveness.csv pour tester !
    # Exemple générique: On cherche juste "Diabète"
    target_codes = ["44054006"] 
    
    print(f"--- Requête : {query_description} ---")

    # 2. Analyse
    # On passe maintenant ASSIGNMENTS_PATH au constructeur
    selector = CohortSelector(TOP_CODES_PATH, SUMMARY_PATH, ASSIGNMENTS_PATH)
    best_cluster_id = selector.find_best_cluster(target_codes)
    
    if best_cluster_id is None:
        print("Aucun cluster pertinent trouvé. Utilisation du cluster 0 par défaut pour la démo.")
        best_cluster_id = 0
    
    print(f"✅ Cluster Match : #{best_cluster_id}")
    
    # 3. Export des Données (La liste des patients)
    OUTPUT_DIR = "results/dcat"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # C'est ici qu'on génère le vrai fichier CSV des IDs
    ids_filename = f"cohort_{best_cluster_id}_ids.csv"
    ids_path = os.path.join(OUTPUT_DIR, ids_filename)
    selector.export_patient_ids(best_cluster_id, ids_path)

    # 4. Génération des Métadonnées DCAT
    stats = selector.get_cohort_stats(best_cluster_id)
    df_codes = selector.top_codes
    top_features = df_codes[df_codes['cluster_id'] == best_cluster_id]['code'].head(5).tolist()

    generator = DCATGenerator()
    generator.generate_dataset_metadata(
        cohort_id=best_cluster_id,
        stats=stats,
        query_description=query_description,
        top_features=top_features
    )
    
    ttl_filename = f"cohort_{best_cluster_id}_metadata.ttl"
    generator.export_ttl(os.path.join(OUTPUT_DIR, ttl_filename))

    print("\n--- Terminé ---")
    print(f"1. Données brutes : {ids_path}")
    print(f"2. Métadonnées RDF : {os.path.join(OUTPUT_DIR, ttl_filename)}")

if __name__ == "__main__":
    main()