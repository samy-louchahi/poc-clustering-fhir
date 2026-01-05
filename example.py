"""
Example demonstrating the FHIR clustering pipeline with synthetic data.
"""

import numpy as np
import pandas as pd
from fhir_clustering.data_structures import PatientRecord, MedicalCode, CodeSystem
from fhir_clustering.data_loader import FHIRDataLoader
from fhir_clustering.pipeline import FHIRClusteringPipeline


def generate_synthetic_data(n_patients: int = 100, 
                            n_conditions_per_patient: tuple = (3, 15)) -> pd.DataFrame:
    """
    Generate synthetic FHIR patient data for demonstration.
    
    Creates patients with different medical profiles:
    - Cardiovascular patients
    - Diabetic patients
    - Respiratory patients
    - Mixed conditions
    
    Args:
        n_patients: Number of patients to generate
        n_conditions_per_patient: Min/max conditions per patient
        
    Returns:
        DataFrame with patient_id, code, system, display
    """
    np.random.seed(42)
    
    # Define code pools for different condition groups
    cardiovascular_codes = [
        ('49601007', 'SNOMED', 'Disorder of cardiovascular system'),
        ('56265001', 'SNOMED', 'Heart disease'),
        ('38341003', 'SNOMED', 'Hypertension'),
        ('22298006', 'SNOMED', 'Myocardial infarction'),
        ('84114007', 'SNOMED', 'Heart failure'),
        ('2708-6', 'LOINC', 'Oxygen saturation'),
        ('8867-4', 'LOINC', 'Heart rate'),
        ('8480-6', 'LOINC', 'Systolic blood pressure'),
    ]
    
    diabetic_codes = [
        ('73211009', 'SNOMED', 'Diabetes mellitus'),
        ('44054006', 'SNOMED', 'Diabetes mellitus type 2'),
        ('46635009', 'SNOMED', 'Diabetes mellitus type 1'),
        ('2339-0', 'LOINC', 'Glucose'),
        ('4548-4', 'LOINC', 'Hemoglobin A1c'),
        ('2345-7', 'LOINC', 'Glucose [Mass/volume] in Serum or Plasma'),
        ('314684', 'RxNorm', 'Metformin'),
        ('253182', 'RxNorm', 'Insulin glargine'),
    ]
    
    respiratory_codes = [
        ('195967001', 'SNOMED', 'Asthma'),
        ('13645005', 'SNOMED', 'Chronic obstructive pulmonary disease'),
        ('233604007', 'SNOMED', 'Pneumonia'),
        ('49727002', 'SNOMED', 'Cough'),
        ('267036007', 'SNOMED', 'Dyspnea'),
        ('20564-1', 'LOINC', 'Oxygen saturation in Blood'),
        ('59408-5', 'LOINC', 'Oxygen saturation in Arterial blood'),
        ('745752', 'RxNorm', 'Albuterol'),
    ]
    
    common_codes = [
        ('386661006', 'SNOMED', 'Fever'),
        ('271807003', 'SNOMED', 'Rash'),
        ('22253000', 'SNOMED', 'Pain'),
        ('8302-2', 'LOINC', 'Body height'),
        ('29463-7', 'LOINC', 'Body weight'),
        ('8310-5', 'LOINC', 'Body temperature'),
    ]
    
    records = []
    
    # Generate patients from different groups
    for i in range(n_patients):
        patient_id = f"P{i:04d}"
        
        # Assign patient to a primary group
        if i < n_patients * 0.3:  # 30% cardiovascular
            primary_codes = cardiovascular_codes
            secondary_codes = common_codes
        elif i < n_patients * 0.6:  # 30% diabetic
            primary_codes = diabetic_codes
            secondary_codes = common_codes
        elif i < n_patients * 0.8:  # 20% respiratory
            primary_codes = respiratory_codes
            secondary_codes = common_codes
        else:  # 20% mixed
            primary_codes = (cardiovascular_codes + diabetic_codes + 
                           respiratory_codes)
            secondary_codes = common_codes
        
        # Generate random number of conditions
        n_conditions = np.random.randint(*n_conditions_per_patient)
        
        # Sample codes (with repetition possible)
        n_primary = int(n_conditions * 0.7)
        n_secondary = n_conditions - n_primary
        
        selected_codes = []
        for _ in range(n_primary):
            selected_codes.append(primary_codes[np.random.randint(len(primary_codes))])
        for _ in range(n_secondary):
            selected_codes.append(secondary_codes[np.random.randint(len(secondary_codes))])
        
        # Add to records
        for code, system, display in selected_codes:
            records.append({
                'patient_id': patient_id,
                'code': code,
                'system': system,
                'display': display
            })
    
    return pd.DataFrame(records)


def main():
    """Run the complete demonstration."""
    
    print("=" * 80)
    print("FHIR Patient Clustering POC - Demonstration")
    print("=" * 80)
    print()
    
    # Generate synthetic data
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(n_patients=150, n_conditions_per_patient=(5, 20))
    print(f"Generated {len(df)} code entries for {df['patient_id'].nunique()} patients")
    print(f"Code systems: {df['system'].unique()}")
    print()
    
    # Load data
    print("Loading patient records...")
    patients = FHIRDataLoader.from_dataframe(df)
    print(f"Loaded {len(patients)} patient records")
    print()
    
    # Run clustering pipeline
    print("-" * 80)
    print("Running Clustering Pipeline")
    print("-" * 80)
    print()
    
    pipeline = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=True,
        dimensionality_reduction='svd',
        n_components=30,
        clustering_method='kmeans',
        n_clusters=4
    )
    
    pipeline.fit(patients)
    print()
    
    # Get results
    print("-" * 80)
    print("Clustering Results")
    print("-" * 80)
    print()
    
    # Cluster summary
    print("Cluster Summary:")
    summary = pipeline.get_cluster_summary()
    print(summary.to_string(index=False))
    print()
    
    # Top codes per cluster
    print("Top 5 codes per cluster (by frequency):")
    print()
    top_codes = pipeline.get_top_codes_per_cluster(top_n=5, method='frequency')
    
    for cluster_id, codes in sorted(top_codes.items()):
        print(f"Cluster {cluster_id}:")
        for rank, (code, score) in enumerate(codes, 1):
            print(f"  {rank}. {code} (count: {score:.0f})")
        print()
    
    # Patient assignments
    print("Sample patient assignments:")
    assignments = pipeline.get_patient_assignments()
    print(assignments.head(10).to_string(index=False))
    print()
    
    # Try DBSCAN
    print("-" * 80)
    print("Alternative: DBSCAN Clustering")
    print("-" * 80)
    print()
    
    pipeline_dbscan = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=True,
        dimensionality_reduction='svd',
        n_components=30,
        clustering_method='dbscan',
    )
    
    pipeline_dbscan.fit(patients, eps=0.8, min_samples=3)
    print()
    
    summary_dbscan = pipeline_dbscan.get_cluster_summary()
    print("DBSCAN Cluster Summary:")
    print(summary_dbscan.to_string(index=False))
    print()
    
    print("=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
