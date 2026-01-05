"""
Basic tests for FHIR clustering functionality.
Run with: python -m pytest tests/
"""

import numpy as np
import pandas as pd
from fhir_clustering.data_structures import PatientRecord, MedicalCode, CodeSystem
from fhir_clustering.data_loader import FHIRDataLoader
from fhir_clustering.matrix_builder import PatientCodeMatrix
from fhir_clustering.feature_engineering import FeatureTransformer
from fhir_clustering.dimensionality_reduction import DimensionalityReducer
from fhir_clustering.clustering import PatientClusterer
from fhir_clustering.pipeline import FHIRClusteringPipeline


def create_sample_data():
    """Create sample patient data for testing."""
    data = {
        'patient_id': ['P1', 'P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P3'],
        'code': ['12345', '67890', '11111', '12345', '22222', '67890', '11111', '33333'],
        'system': ['SNOMED', 'SNOMED', 'LOINC', 'SNOMED', 'LOINC', 'SNOMED', 'LOINC', 'RxNorm'],
        'display': ['Disease A', 'Disease B', 'Test C', 'Disease A', 'Test D', 'Disease B', 'Test C', 'Med E']
    }
    return pd.DataFrame(data)


def test_data_structures():
    """Test basic data structures."""
    code = MedicalCode(code='12345', system=CodeSystem.SNOMED, display='Test')
    assert str(code) == 'SNOMED:12345'
    
    patient = PatientRecord(patient_id='P1')
    patient.add_code(code)
    assert len(patient.codes) == 1
    assert len(patient.get_unique_codes()) == 1


def test_data_loader():
    """Test data loading from DataFrame."""
    df = create_sample_data()
    patients = FHIRDataLoader.from_dataframe(df)
    
    assert len(patients) == 3
    assert patients[0].patient_id in ['P1', 'P2', 'P3']
    assert len(patients[0].codes) > 0


def test_matrix_builder():
    """Test matrix construction."""
    df = create_sample_data()
    patients = FHIRDataLoader.from_dataframe(df)
    
    builder = PatientCodeMatrix(patients)
    matrix = builder.build_matrix()
    
    assert matrix.shape[0] == 3  # 3 patients
    assert matrix.shape[1] > 0  # Some codes
    assert builder.get_matrix_stats()['sparsity'] > 0


def test_feature_transformer():
    """Test TF-IDF transformation."""
    df = create_sample_data()
    patients = FHIRDataLoader.from_dataframe(df)
    
    builder = PatientCodeMatrix(patients)
    matrix = builder.build_matrix()
    
    transformer = FeatureTransformer()
    tfidf_matrix = transformer.apply_tfidf(matrix)
    
    assert tfidf_matrix.shape == matrix.shape
    assert tfidf_matrix.nnz > 0  # Has non-zero entries


def test_dimensionality_reduction():
    """Test dimensionality reduction."""
    df = create_sample_data()
    patients = FHIRDataLoader.from_dataframe(df)
    
    builder = PatientCodeMatrix(patients)
    matrix = builder.build_matrix()
    
    reducer = DimensionalityReducer(method='svd', n_components=2)
    reduced = reducer.fit_transform(matrix)
    
    assert reduced.shape == (3, 2)  # 3 patients, 2 components


def test_clustering():
    """Test clustering."""
    # Create more data for meaningful clustering
    data = []
    for i in range(20):
        for code in ['C1', 'C2', 'C3']:
            data.append({
                'patient_id': f'P{i}',
                'code': code,
                'system': 'SNOMED',
                'display': f'Code {code}'
            })
    
    df = pd.DataFrame(data)
    patients = FHIRDataLoader.from_dataframe(df)
    
    builder = PatientCodeMatrix(patients)
    matrix = builder.build_matrix()
    
    clusterer = PatientClusterer(method='kmeans', n_clusters=2)
    labels = clusterer.fit_predict(matrix.toarray())
    
    assert len(labels) == 20
    assert clusterer.get_n_clusters() <= 2


def test_pipeline():
    """Test complete pipeline."""
    # Create more patients for meaningful clustering
    data = []
    for i in range(30):
        codes = ['C1', 'C2'] if i < 15 else ['C3', 'C4']
        for code in codes:
            data.append({
                'patient_id': f'P{i}',
                'code': code,
                'system': 'SNOMED',
                'display': f'Code {code}'
            })
    
    df = pd.DataFrame(data)
    patients = FHIRDataLoader.from_dataframe(df)
    
    pipeline = FHIRClusteringPipeline(
        apply_tfidf=True,
        dimensionality_reduction='svd',
        n_components=2,
        clustering_method='kmeans',
        n_clusters=2
    )
    
    pipeline.fit(patients)
    
    # Test results
    labels = pipeline.get_cluster_labels()
    assert len(labels) == 30
    
    summary = pipeline.get_cluster_summary()
    assert len(summary) == 2
    
    assignments = pipeline.get_patient_assignments()
    assert len(assignments) == 30
    
    top_codes = pipeline.get_top_codes_per_cluster(top_n=5)
    assert len(top_codes) == 2


if __name__ == '__main__':
    print("Running tests...")
    
    test_data_structures()
    print("✓ test_data_structures passed")
    
    test_data_loader()
    print("✓ test_data_loader passed")
    
    test_matrix_builder()
    print("✓ test_matrix_builder passed")
    
    test_feature_transformer()
    print("✓ test_feature_transformer passed")
    
    test_dimensionality_reduction()
    print("✓ test_dimensionality_reduction passed")
    
    test_clustering()
    print("✓ test_clustering passed")
    
    test_pipeline()
    print("✓ test_pipeline passed")
    
    print("\nAll tests passed!")
