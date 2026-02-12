"""
Module d'export DCAT-AP Health.
"""
import os
import datetime
from typing import List, Dict, Optional
import pandas as pd
from rdflib import Graph, Literal, BNode, Namespace, URIRef
from rdflib.namespace import DCAT, DCTERMS, RDF, FOAF, XSD

# Namespaces
H_DCAT = Namespace("http://www.w3.org/ns/dcat#") 
PROV = Namespace("http://www.w3.org/ns/prov#")

class CohortSelector:
    """
    Analyse les résultats pour trouver le cluster et extraire les patients.
    """
    def __init__(self, top_codes_csv: str, summary_csv: str, assignments_csv: str):
        # Vérification
        for f in [top_codes_csv, summary_csv, assignments_csv]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Fichier introuvable: {f}")

        self.top_codes = pd.read_csv(top_codes_csv)
        self.summary = pd.read_csv(summary_csv)
        self.assignments = pd.read_csv(assignments_csv) # <--- NOUVEAU
    
    def find_best_cluster(self, target_codes: List[str]) -> Optional[int]:
        # (Cette méthode reste inchangée, voir code précédent)
        # ... [Garder le code existant de find_best_cluster] ...
        best_cluster = -1
        max_score = 0
        targets = [t.split(':')[-1] if ':' in t else t for t in target_codes]
        unique_clusters = self.top_codes['cluster_id'].unique()
        
        for cid in unique_clusters:
            cluster_codes = self.top_codes[self.top_codes['cluster_id'] == cid]
            score = 0
            found_count = 0
            for target in targets:
                match = cluster_codes[cluster_codes['code'].astype(str).str.contains(target, regex=False)]
                if not match.empty:
                    score += match.iloc[0]['score']
                    found_count += 1
            
            if found_count == len(targets) and found_count > 0:
                score *= 2  
            
            if score > max_score:
                max_score = score
                best_cluster = cid
                
        return int(best_cluster) if max_score > 0 else None

    def get_cohort_stats(self, cluster_id: int) -> Dict:
        """Récupère les stats."""
        row = self.summary[self.summary['cluster_id'] == cluster_id].iloc[0]
        return {
            "size": int(row['n_patients']),
            "id": int(cluster_id)
        }

    def export_patient_ids(self, cluster_id: int, output_path: str):
        """
        Extrait la liste des IDs patients du cluster et sauvegarde en CSV.
        """
        # Détection de la colonne cluster (kmeans_cluster ou hdbscan_cluster ou cluster_id)
        col_name = None
        for candidate in ['kmeans_cluster', 'hdbscan_cluster', 'cluster_id']:
            if candidate in self.assignments.columns:
                col_name = candidate
                break
        
        if not col_name:
            raise ValueError("Impossible de trouver la colonne de cluster dans assignments.csv")

        # Filtrage
        cohort_patients = self.assignments[self.assignments[col_name] == cluster_id]
        
        # Sauvegarde seulement l'ID
        cohort_patients[['patient_id']].to_csv(output_path, index=False)
        print(f"Liste des {len(cohort_patients)} patients exportée vers : {output_path}")


class DCATGenerator:
    # (Cette classe reste inchangée, voir code précédent)
    def __init__(self, base_uri="http://data.conecsio.health/cohorts/"):
        self.base_uri = base_uri
        self.g = Graph()
        self.g.bind("dcat", DCAT)
        self.g.bind("dct", DCTERMS)
        self.g.bind("foaf", FOAF)
        self.g.bind("prov", PROV)

    def generate_dataset_metadata(self, cohort_id: int, stats: Dict, query_description: str, top_features: List[str]):
        dataset_uri = URIRef(f"{self.base_uri}{cohort_id}")
        now = datetime.datetime.now().isoformat()
        
        self.g.add((dataset_uri, RDF.type, DCAT.Dataset))
        
        title = f"Cohorte #{cohort_id}: {query_description}"
        desc = (f"Cohorte de {stats['size']} patients générée par clustering IA non-supervisé. "
                f"Identifiée par la présence conjointe de : {', '.join(str(f) for f in top_features[:3])}.")
        
        self.g.add((dataset_uri, DCTERMS.title, Literal(title, lang="fr")))
        self.g.add((dataset_uri, DCTERMS.description, Literal(desc, lang="fr")))
        
        self.g.add((dataset_uri, DCAT.keyword, Literal("FHIR")))
        self.g.add((dataset_uri, DCAT.keyword, Literal("Cohorte")))
        for feat in top_features[:3]:
            self.g.add((dataset_uri, DCAT.keyword, Literal(str(feat))))
            
        self.g.add((dataset_uri, DCTERMS.issued, Literal(now, datatype=XSD.dateTime)))
        self.g.add((dataset_uri, DCTERMS.modified, Literal(now, datatype=XSD.dateTime)))
        
        publisher = BNode()
        self.g.add((publisher, RDF.type, FOAF.Agent))
        self.g.add((publisher, FOAF.name, Literal("Conecsio AI Lab")))
        self.g.add((dataset_uri, DCTERMS.publisher, publisher))
        
        activity = BNode()
        self.g.add((activity, RDF.type, PROV.Activity))
        self.g.add((activity, DCTERMS.title, Literal("Clustering Process")))
        self.g.add((dataset_uri, PROV.wasGeneratedBy, activity))
        
        # Distribution (Le lien vers le CSV qu'on va générer)
        dist = BNode()
        self.g.add((dist, RDF.type, DCAT.Distribution))
        self.g.add((dist, DCTERMS.title, Literal("Liste des IDs Patients (CSV)")))
        self.g.add((dist, DCAT.mediaType, Literal("text/csv")))
        # Note : Dans un vrai système, ce serait une URL HTTP accessible
        self.g.add((dist, DCAT.downloadURL, URIRef(f"{self.base_uri}{cohort_id}/ids.csv")))
        self.g.add((dataset_uri, DCAT.distribution, dist))

    def export_ttl(self, output_path: str):
        self.g.serialize(destination=output_path, format="turtle")
        print(f"Fichier DCAT-AP généré : {output_path}")