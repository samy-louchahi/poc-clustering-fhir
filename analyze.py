"""
Utility script to run analysis with different configurations and compare results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example import generate_synthetic_data
from fhir_clustering.data_loader import FHIRDataLoader
from fhir_clustering.data_structures import CodeSystem
from fhir_clustering.pipeline import FHIRClusteringPipeline
import pandas as pd


def compare_clustering_methods():
    """Compare different clustering methods on the same data."""
    
    print("=" * 80)
    print("Comparison of Clustering Methods")
    print("=" * 80)
    print()
    
    # Generate data once
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(n_patients=100, n_conditions_per_patient=(5, 15))
    patients = FHIRDataLoader.from_dataframe(df)
    print(f"Loaded {len(patients)} patients")
    print()
    
    configs = [
        {
            'name': 'KMeans (k=3)',
            'method': 'kmeans',
            'n_clusters': 3,
        },
        {
            'name': 'KMeans (k=4)',
            'method': 'kmeans',
            'n_clusters': 4,
        },
        {
            'name': 'DBSCAN',
            'method': 'dbscan',
            'params': {'eps': 0.8, 'min_samples': 3}
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 80}\n")
        
        pipeline = FHIRClusteringPipeline(
            include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
            apply_tfidf=True,
            dimensionality_reduction='svd',
            n_components=20,
            clustering_method=config['method'],
            n_clusters=config.get('n_clusters')
        )
        
        pipeline.fit(patients, **config.get('params', {}))
        
        summary = pipeline.get_cluster_summary()
        n_clusters = len(summary[~summary['is_noise']])
        
        print("\nCluster Summary:")
        print(summary.to_string(index=False))
        
        results.append({
            'Method': config['name'],
            'N_Clusters': n_clusters,
            'Avg_Cluster_Size': summary[~summary['is_noise']]['n_patients'].mean(),
            'Has_Noise': summary['is_noise'].any()
        })
        
        print()
    
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()


def analyze_dimensionality_impact():
    """Analyze impact of different dimensionality reduction settings."""
    
    print("=" * 80)
    print("Impact of Dimensionality Reduction")
    print("=" * 80)
    print()
    
    # Generate data
    df = generate_synthetic_data(n_patients=100, n_conditions_per_patient=(5, 15))
    patients = FHIRDataLoader.from_dataframe(df)
    
    n_components_list = [10, 20, 30, None]
    results = []
    
    for n_comp in n_components_list:
        dim_method = 'svd' if n_comp else None
        name = f"SVD ({n_comp} components)" if n_comp else "No reduction"
        
        print(f"\nTesting: {name}")
        
        pipeline = FHIRClusteringPipeline(
            apply_tfidf=True,
            dimensionality_reduction=dim_method,
            n_components=n_comp if n_comp else 50,
            clustering_method='kmeans',
            n_clusters=3
        )
        
        pipeline.fit(patients)
        
        summary = pipeline.get_cluster_summary()
        
        explained_var = None
        if pipeline.dim_reducer:
            explained_var = pipeline.dim_reducer.get_cumulative_variance()[-1]
        
        results.append({
            'Configuration': name,
            'N_Components': n_comp if n_comp else 'All',
            'Explained_Variance': f"{explained_var:.2%}" if explained_var else 'N/A',
            'Avg_Cluster_Size': f"{summary['n_patients'].mean():.1f}"
        })
    
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Utility script for FHIR clustering analysis"
    )
    parser.add_argument(
        '--mode',
        choices=['compare', 'dimensionality', 'both'],
        default='both',
        help='Analysis mode to run'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['compare', 'both']:
        compare_clustering_methods()
    
    if args.mode in ['dimensionality', 'both']:
        analyze_dimensionality_impact()
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
