"""
Dimensionality reduction techniques for high-dimensional patient data.
Updated to include UMAP.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA
from typing import Optional, Literal, Union

# Attempt to import UMAP safely
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

class DimensionalityReducer:
    """
    Reduce dimensionality of patient-code matrices.
    Supports SVD (LSA), PCA, and UMAP.
    """
    
    def __init__(self, method: Literal['svd', 'pca', 'umap'] = 'svd', n_components: int = 50, random_state: int = 42):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('svd', 'pca', 'umap')
            n_components: Number of components to keep
            random_state: Seed for reproducibility
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
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
            # TruncatedSVD works well with sparse matrices (LSA)
            self.reducer = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state
            )
            return self.reducer.fit_transform(matrix)

        elif self.method == 'pca':
            # PCA requires dense data, use with caution on large datasets
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
            # Warning: converting sparse to dense might explode memory
            return self.reducer.fit_transform(matrix.toarray())

        elif self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("The 'umap-learn' library is not installed. Please run: pip install umap-learn")
            
            # UMAP configuration optimized for clustering sparse data (TF-IDF)
            # metric='cosine' is generally better for high-dim text-like data
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=15,    # Default balanced value
                min_dist=0.1,      # Default value
                metric='cosine',   # Effective for sparse/TF-IDF data
                random_state=self.random_state,
                transform_seed=self.random_state
            )
            # UMAP handles sparse matrices natively
            return self.reducer.fit_transform(matrix)

        else:
            raise ValueError(f"Unknown method: {self.method}")
        
    def transform(self, matrix: csr_matrix) -> np.ndarray:
        """
        Transform new data using fitted reducer.
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform() first.")
        
        if self.method == 'pca':
            return self.reducer.transform(matrix.toarray())
        
        # SVD and UMAP (if version supports it) support sparse input
        return self.reducer.transform(matrix)
    
    def get_explained_variance(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.
        Returns empty array for UMAP (concept not applicable).
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform() first.")
        
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            return self.reducer.explained_variance_ratio_
        
        # UMAP does not have explained variance
        return np.array([])
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance.
        Returns empty array for UMAP.
        """
        variance = self.get_explained_variance()
        if len(variance) == 0:
            return np.array([])
        return np.cumsum(variance)
    
    def get_components(self) -> Optional[np.ndarray]:
        """
        Get the principal components/singular vectors.
        
        Returns:
            Components matrix (n_components, n_features) or None for UMAP.
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform() first.")
        
        if hasattr(self.reducer, 'components_'):
            return self.reducer.components_
        
        # UMAP is non-linear and doesn't provide linear component vectors
        return None