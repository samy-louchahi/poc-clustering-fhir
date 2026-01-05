"""
Clustering algorithms for patient segmentation.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import Optional, Literal
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class PatientClusterer:
    """
    Apply clustering algorithms to patient data.
    """
    
    def __init__(self, 
                 method: Literal['kmeans', 'dbscan', 'hdbscan'] = 'kmeans',
                 n_clusters: Optional[int] = None,
                 **kwargs):
        """
        Initialize clusterer.
        
        Args:
            method: Clustering method ('kmeans', 'dbscan', 'hdbscan')
            n_clusters: Number of clusters (for KMeans)
            **kwargs: Additional parameters for the clustering algorithm
        """
        self.method = method
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.clusterer = None
        self.labels_ = None
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit clustering and predict cluster labels.
        
        Args:
            X: Input data matrix (n_samples, n_features)
            
        Returns:
            Cluster labels for each sample
        """
        if self.method == 'kmeans':
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for KMeans")
            
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                **self.kwargs
            )
            self.labels_ = self.clusterer.fit_predict(X)
            
        elif self.method == 'dbscan':
            eps = self.kwargs.get('eps', 0.5)
            min_samples = self.kwargs.get('min_samples', 5)
            
            # Filter out handled parameters
            extra_kwargs = {k: v for k, v in self.kwargs.items() 
                          if k not in ['eps', 'min_samples']}
            
            self.clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                **extra_kwargs
            )
            self.labels_ = self.clusterer.fit_predict(X)
            
        elif self.method == 'hdbscan':
            if not HDBSCAN_AVAILABLE:
                raise ImportError("hdbscan package not installed. Install with: pip install hdbscan")
            
            min_cluster_size = self.kwargs.get('min_cluster_size', 5)
            min_samples = self.kwargs.get('min_samples', None)
            
            # Filter out handled parameters
            extra_kwargs = {k: v for k, v in self.kwargs.items() 
                          if k not in ['min_cluster_size', 'min_samples']}
            
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                **extra_kwargs
            )
            self.labels_ = self.clusterer.fit_predict(X)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.labels_
    
    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster labels."""
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit_predict() first.")
        return self.labels_
    
    def get_n_clusters(self) -> int:
        """Get the number of clusters found (excluding noise for DBSCAN/HDBSCAN)."""
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit_predict() first.")
        return len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
    
    def get_cluster_sizes(self) -> dict:
        """Get the size of each cluster."""
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit_predict() first.")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_inertia(self) -> Optional[float]:
        """Get inertia (sum of squared distances to centers) for KMeans."""
        if self.method == 'kmeans' and self.clusterer is not None:
            return self.clusterer.inertia_
        return None
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers (for KMeans only)."""
        if self.method == 'kmeans' and self.clusterer is not None:
            return self.clusterer.cluster_centers_
        return None
