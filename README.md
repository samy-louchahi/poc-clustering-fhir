# FHIR Patient Clustering POC

A Python proof-of-concept for **unsupervised clustering** of FHIR patient data based on medical codes (SNOMED, LOINC, RxNorm).

## Overview

This POC implements a complete pipeline for:
- Building patient × medical code matrices with sparse data handling
- Applying TF-IDF weighting to identify distinctive codes
- Dimensionality reduction using TruncatedSVD or PCA
- Patient clustering with KMeans, DBSCAN, or HDBSCAN
- Cluster interpretation via top characteristic medical codes

The code is **modular**, **reproducible**, and focused on **interpretability**.

## Features

- ✅ **Sparse matrix handling** using scipy.sparse for efficient memory usage
- ✅ **Multiple code systems**: SNOMED, LOINC, RxNorm (configurable)
- ✅ **TF-IDF transformation** to identify distinctive patient codes
- ✅ **Dimensionality reduction**: TruncatedSVD (sparse-friendly) or PCA
- ✅ **Multiple clustering algorithms**: KMeans, DBSCAN, HDBSCAN
- ✅ **Interpretability**: Top codes per cluster with multiple ranking methods
- ✅ **Clean, modular design**: Easy to extend and customize

## Installation

```bash
# Clone the repository
git clone https://github.com/samy-louchahi/poc-clustering-fhir.git
cd poc-clustering-fhir

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from fhir_clustering.data_loader import FHIRDataLoader
from fhir_clustering.data_structures import CodeSystem
from fhir_clustering.pipeline import FHIRClusteringPipeline
import pandas as pd

# Load your data (patient_id, code, system, display)
df = pd.read_csv('your_patient_data.csv')
patients = FHIRDataLoader.from_dataframe(df)

# Configure and run the pipeline
pipeline = FHIRClusteringPipeline(
    include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC],
    apply_tfidf=True,
    dimensionality_reduction='svd',
    n_components=50,
    clustering_method='kmeans',
    n_clusters=5
)

# Fit the pipeline
pipeline.fit(patients)

# Get results
summary = pipeline.get_cluster_summary()
top_codes = pipeline.get_top_codes_per_cluster(top_n=10)
assignments = pipeline.get_patient_assignments()
```

## Run the Demo

A complete demonstration with synthetic data:

```bash
python example.py
```

This will:
1. Generate synthetic patient data with different medical profiles
2. Build the patient-code matrix
3. Apply TF-IDF and dimensionality reduction
4. Perform clustering with KMeans and DBSCAN
5. Display cluster statistics and top codes per cluster

## Architecture

### Module Structure

```
fhir_clustering/
├── __init__.py              # Package initialization
├── data_structures.py       # PatientRecord, MedicalCode classes
├── data_loader.py           # Load data from DataFrame or dict
├── matrix_builder.py        # Build patient × code sparse matrices
├── feature_engineering.py   # TF-IDF and normalization
├── dimensionality_reduction.py  # TruncatedSVD, PCA
├── clustering.py            # KMeans, DBSCAN, HDBSCAN
├── interpretation.py        # Cluster interpretation and top codes
└── pipeline.py              # Main orchestration pipeline
```

### Data Flow

```
Patient Data (CSV/DataFrame)
    ↓
FHIRDataLoader → List[PatientRecord]
    ↓
PatientCodeMatrix → Sparse Matrix (n_patients × n_codes)
    ↓
FeatureTransformer → TF-IDF weighted matrix
    ↓
DimensionalityReducer → Reduced representation
    ↓
PatientClusterer → Cluster labels
    ↓
ClusterInterpreter → Top codes, statistics, interpretations
```

## API Reference

### Pipeline Configuration

**FHIRClusteringPipeline** parameters:

- `include_systems`: List of CodeSystem enums to include (default: all)
- `apply_tfidf`: Apply TF-IDF transformation (default: True)
- `dimensionality_reduction`: 'svd', 'pca', or None (default: 'svd')
- `n_components`: Number of dimensions to reduce to (default: 50)
- `clustering_method`: 'kmeans', 'dbscan', or 'hdbscan' (default: 'kmeans')
- `n_clusters`: Number of clusters for KMeans (required if using KMeans)

### Key Methods

```python
# Fit the pipeline
pipeline.fit(patients, **clustering_kwargs)

# Get cluster assignments
assignments = pipeline.get_patient_assignments()

# Get cluster summary statistics
summary = pipeline.get_cluster_summary()

# Get top codes characterizing each cluster
top_codes = pipeline.get_top_codes_per_cluster(
    top_n=10, 
    method='frequency'  # or 'tfidf', 'distinctiveness'
)

# Get cluster labels
labels = pipeline.get_cluster_labels()
```

## Clustering Methods

### KMeans
- Best for: Well-separated, spherical clusters
- Requires: n_clusters parameter
- Pros: Fast, deterministic, interpretable centers

### DBSCAN
- Best for: Arbitrary-shaped clusters, handling noise
- Parameters: eps, min_samples
- Pros: No need to specify cluster count, finds outliers

### HDBSCAN
- Best for: Varying density clusters
- Parameters: min_cluster_size, min_samples
- Pros: Hierarchical approach, robust to parameters

## Interpretation Methods

### Frequency
Ranks codes by total occurrence count in each cluster.

### TF-IDF
Ranks codes by average TF-IDF score in each cluster (emphasizes distinctive codes).

### Distinctiveness
Ranks codes by ratio of cluster frequency to overall frequency (highlights cluster-specific codes).

## Data Format

Input data should be in one of these formats:

### DataFrame Format
```python
df = pd.DataFrame({
    'patient_id': ['P001', 'P001', 'P002'],
    'code': ['73211009', '44054006', '195967001'],
    'system': ['SNOMED', 'SNOMED', 'SNOMED'],
    'display': ['Diabetes mellitus', 'Type 2 diabetes', 'Asthma']
})
```

### Dictionary List Format
```python
data = [
    {
        'patient_id': 'P001',
        'codes': [
            {'code': '73211009', 'system': 'SNOMED', 'display': 'Diabetes'},
            {'code': '2339-0', 'system': 'LOINC', 'display': 'Glucose'}
        ]
    }
]
```

## Technical Details

### Sparse Matrix Handling
- Uses scipy.sparse.csr_matrix for efficient storage
- Typical sparsity >99% for real patient data
- Memory-efficient for large datasets

### TF-IDF
- Term Frequency: How often a code appears for a patient
- Inverse Document Frequency: Down-weights common codes
- Helps identify distinctive patient characteristics

### Dimensionality Reduction
- TruncatedSVD: Works directly with sparse matrices
- PCA: Requires dense data, use for small datasets
- Reduces noise and computational complexity

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- hdbscan >= 0.8.27 (optional, for HDBSCAN clustering)
- matplotlib >= 3.4.0 (for visualization)
- seaborn >= 0.11.0 (for visualization)

## Use Cases

- **Patient stratification**: Identify subgroups for personalized medicine
- **Disease phenotyping**: Discover disease subtypes based on comorbidities
- **Care pathway optimization**: Group patients with similar care needs
- **Risk assessment**: Identify high-risk patient groups
- **Quality improvement**: Target interventions to specific patient clusters

## Future Enhancements

- [ ] t-SNE/UMAP integration for 2D visualization
- [ ] Cross-validation and cluster stability metrics
- [ ] Integration with FHIR servers
- [ ] Support for temporal patterns
- [ ] Automated optimal cluster number selection
- [ ] Export to standard formats (JSON, Parquet)

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.