"""
Main pipeline orchestrating the clustering workflow.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Optional, Literal
import pandas as pd

from .data_structures import PatientRecord, CodeSystem
from .matrix_builder import PatientCodeMatrix
from .feature_engineering import FeatureTransformer
from .dimensionality_reduction import DimensionalityReducer
from .clustering import PatientClusterer
from .interpretation import ClusterInterpreter


class FHIRClusteringPipeline:
    """
    Complete pipeline for FHIR patient clustering.
    """
    
    def __init__(self,
                 include_systems: Optional[List[CodeSystem]] = None,
                 apply_tfidf: bool = True,
                 dimensionality_reduction: Optional[str] = 'svd',
                 n_components: int = 50,
                 clustering_method: str = 'kmeans',
                 n_clusters: Optional[int] = None):
        """
        Initialize clustering pipeline.
        
        Args:
            include_systems: Code systems to include (default: all)
            apply_tfidf: Whether to apply TF-IDF transformation
            dimensionality_reduction: Method for reduction ('svd', 'pca', or None)
            n_components: Number of components for dimensionality reduction
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hdbscan')
            n_clusters: Number of clusters (for KMeans)
        """
        self.include_systems = include_systems
        self.apply_tfidf = apply_tfidf
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        
        # Components
        self.matrix_builder: Optional[PatientCodeMatrix] = None
        self.feature_transformer: Optional[FeatureTransformer] = None
        self.dim_reducer: Optional[DimensionalityReducer] = None
        self.clusterer: Optional[PatientClusterer] = None
        self.interpreter: Optional[ClusterInterpreter] = None
        
        # Data
        self.original_matrix: Optional[csr_matrix] = None
        self.transformed_matrix: Optional[csr_matrix] = None
        self.reduced_data: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        
    def fit(self, patients: List[PatientRecord], **clustering_kwargs):
        """
        Fit the complete pipeline.
        
        Args:
            patients: List of patient records
            **clustering_kwargs: Additional parameters for clustering
            
        Returns:
            self
        """
        print(f"Building patient-code matrix...")
        # Step 1: Build matrix
        self.matrix_builder = PatientCodeMatrix(
            patients=patients,
            include_systems=self.include_systems
        )
        self.original_matrix = self.matrix_builder.build_matrix()
        stats = self.matrix_builder.get_matrix_stats()
        print(f"Matrix shape: {stats['n_patients']} patients × {stats['n_codes']} codes")
        print(f"Sparsity: {stats['sparsity']:.2%}")
        
        # Step 2: Feature transformation
        self.transformed_matrix = self.original_matrix
        if self.apply_tfidf:
            print(f"Applying TF-IDF transformation...")
            self.feature_transformer = FeatureTransformer()
            self.transformed_matrix = self.feature_transformer.apply_tfidf(
                self.original_matrix
            )
        
        # Step 3: Dimensionality reduction
        clustering_input = self.transformed_matrix
        if self.dimensionality_reduction:
            print(f"Reducing dimensionality using {self.dimensionality_reduction}...")
            self.dim_reducer = DimensionalityReducer(
                method=self.dimensionality_reduction,
                n_components=self.n_components
            )
            self.reduced_data = self.dim_reducer.fit_transform(self.transformed_matrix)
            explained_var = self.dim_reducer.get_cumulative_variance()
            print(f"Explained variance: {explained_var[-1]:.2%}")
            clustering_input = self.reduced_data
        
        # Step 4: Clustering
        print(f"Applying {self.clustering_method} clustering...")
        self.clusterer = PatientClusterer(
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            **clustering_kwargs
        )
        self.cluster_labels = self.clusterer.fit_predict(clustering_input)
        n_clusters = self.clusterer.get_n_clusters()
        print(f"Found {n_clusters} clusters")
        cluster_sizes = self.clusterer.get_cluster_sizes()
        print(f"Cluster sizes: {cluster_sizes}")
        
        # Step 5: Set up interpretation
        self.interpreter = ClusterInterpreter(
            matrix_builder=self.matrix_builder,
            original_matrix=self.original_matrix
        )
        
        return self
    
    def get_top_codes_per_cluster(self, top_n: int = 10, 
                                   method: str = 'frequency') -> Dict:
        """
        Get top medical codes characterizing each cluster.
        
        Args:
            top_n: Number of top codes per cluster
            method: Ranking method ('frequency', 'tfidf', 'distinctiveness')
            
        Returns:
            Dictionary mapping cluster_id to top codes
        """
        if self.interpreter is None:
            raise ValueError("Pipeline not fitted yet. Call fit() first.")
        
        return self.interpreter.get_top_codes_per_cluster(
            self.cluster_labels, top_n=top_n, method=method
        )
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary statistics for each cluster."""
        if self.interpreter is None:
            raise ValueError("Pipeline not fitted yet. Call fit() first.")
        
        return self.interpreter.get_cluster_summary(self.cluster_labels)
    
    def get_patient_assignments(self) -> pd.DataFrame:
        """Get patient-to-cluster assignments."""
        if self.interpreter is None:
            raise ValueError("Pipeline not fitted yet. Call fit() first.")
        
        return self.interpreter.get_patient_cluster_membership(self.cluster_labels)
    
    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster labels for all patients."""
        if self.cluster_labels is None:
            raise ValueError("Pipeline not fitted yet. Call fit() first.")
        return self.cluster_labels
