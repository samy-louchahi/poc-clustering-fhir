"""
Cluster interpretation and explainability.
Identify top medical codes that characterize each cluster.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple
from .matrix_builder import PatientCodeMatrix


class ClusterInterpreter:
    """
    Interpret clusters by identifying characteristic medical codes.
    """
    
    def __init__(self, matrix_builder: PatientCodeMatrix, 
                 original_matrix: csr_matrix):
        """
        Initialize interpreter.
        
        Args:
            matrix_builder: PatientCodeMatrix with code mappings
            original_matrix: Original patient-code matrix (before transformations)
        """
        self.matrix_builder = matrix_builder
        self.original_matrix = original_matrix
        
    def get_top_codes_per_cluster(self, 
                                   cluster_labels: np.ndarray,
                                   top_n: int = 10,
                                   method: str = 'frequency') -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top medical codes for each cluster.
        
        Args:
            cluster_labels: Cluster assignment for each patient
            top_n: Number of top codes to return per cluster
            method: Method to rank codes ('frequency', 'tfidf', 'distintiveness')
            
        Returns:
            Dictionary mapping cluster_id to list of (code, score) tuples
        """
        cluster_codes = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            # Get patients in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_matrix = self.original_matrix[cluster_mask]
            
            if method == 'frequency':
                # Sum code occurrences in cluster
                code_scores = np.asarray(cluster_matrix.sum(axis=0)).ravel()
                
            elif method == 'tfidf':
                # Average TF-IDF scores (if using transformed matrix)
                code_scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
                
            elif method == 'distinctiveness':
                # Compare cluster frequency to overall frequency
                cluster_freq = np.asarray(cluster_matrix.sum(axis=0)).ravel()
                cluster_size = cluster_mask.sum()
                overall_freq = np.asarray(self.original_matrix.sum(axis=0)).ravel()
                total_patients = self.original_matrix.shape[0]
                
                # Normalize frequencies
                cluster_rate = cluster_freq / cluster_size
                overall_rate = overall_freq / total_patients
                
                # Ratio of cluster rate to overall rate
                with np.errstate(divide='ignore', invalid='ignore'):
                    code_scores = np.where(overall_rate > 0, 
                                          cluster_rate / overall_rate, 
                                          0)
                    
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get top N codes
            top_indices = np.argsort(code_scores)[-top_n:][::-1]
            top_codes = []
            
            for idx in top_indices:
                code_name = self.matrix_builder.get_code_name(idx)
                score = code_scores[idx]
                if score > 0:  # Only include codes that appear
                    top_codes.append((code_name, float(score)))
            
            cluster_codes[int(cluster_id)] = top_codes
        
        return cluster_codes
    
    def get_cluster_summary(self, cluster_labels: np.ndarray) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        
        Args:
            cluster_labels: Cluster assignment for each patient
            
        Returns:
            DataFrame with cluster statistics
        """
        unique_clusters = np.unique(cluster_labels)
        summaries = []
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_matrix = self.original_matrix[cluster_mask]
            
            # Calculate statistics
            n_patients = cluster_mask.sum()
            avg_codes = cluster_matrix.sum() / n_patients if n_patients > 0 else 0
            unique_codes = (cluster_matrix.sum(axis=0) > 0).sum()
            
            summaries.append({
                'cluster_id': int(cluster_id),
                'n_patients': int(n_patients),
                'avg_codes_per_patient': float(avg_codes),
                'unique_codes': int(unique_codes),
                'is_noise': cluster_id == -1
            })
        
        return pd.DataFrame(summaries)
    
    def get_patient_cluster_membership(self, 
                                       cluster_labels: np.ndarray) -> pd.DataFrame:
        """
        Get DataFrame mapping patients to their clusters.
        
        Args:
            cluster_labels: Cluster assignment for each patient
            
        Returns:
            DataFrame with patient_id and cluster_id columns
        """
        patient_ids = [self.matrix_builder.get_patient_id(i) 
                      for i in range(len(cluster_labels))]
        
        return pd.DataFrame({
            'patient_id': patient_ids,
            'cluster_id': cluster_labels
        })
