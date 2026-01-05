"""
Dimensionality reduction techniques for high-dimensional patient data.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA
from typing import Optional, Literal


class DimensionalityReducer:
    """
    Reduce dimensionality of patient-code matrices.
    """
    
    def __init__(self, method: Literal['svd', 'pca'] = 'svd', n_components: int = 50):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('svd' for TruncatedSVD, 'pca' for PCA)
            n_components: Number of components to keep
        """
        self.method = method
        self.n_components = n_components
        self.reducer = None
        
    def fit_transform(self, matrix: csr_matrix) -> np.ndarray:
        """
        Fit reducer and transform the matrix.
        
        Args:
            matrix: Input sparse matrix (n_patients, n_codes)
            
        Returns:
            Reduced dense matrix (n_patients, n_components)
        """
        if self.method == 'svd':
            # TruncatedSVD works well with sparse matrices
            self.reducer = TruncatedSVD(
                n_components=self.n_components,
                random_state=42
            )
        elif self.method == 'pca':
            # PCA requires dense data, use with caution
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.reducer.fit_transform(matrix)
    
    def transform(self, matrix: csr_matrix) -> np.ndarray:
        """
        Transform new data using fitted reducer.
        
        Args:
            matrix: Input sparse matrix
            
        Returns:
            Reduced dense matrix
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform() first.")
        return self.reducer.transform(matrix)
    
    def get_explained_variance(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.
        
        Returns:
            Array of explained variance ratios
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform() first.")
        return self.reducer.explained_variance_ratio_
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance.
        
        Returns:
            Array of cumulative explained variance ratios
        """
        return np.cumsum(self.get_explained_variance())
    
    def get_components(self) -> np.ndarray:
        """
        Get the principal components/singular vectors.
        
        Returns:
            Components matrix (n_components, n_features)
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform() first.")
        return self.reducer.components_
