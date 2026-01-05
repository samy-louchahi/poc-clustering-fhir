"""
Feature engineering for patient data.
Includes TF-IDF transformation and normalization.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from typing import Literal


class FeatureTransformer:
    """
    Transform patient-code matrices using various techniques.
    """
    
    def __init__(self):
        self.tfidf_transformer = None
        
    def apply_tfidf(self, matrix: csr_matrix, 
                    norm: Literal['l1', 'l2'] = 'l2',
                    use_idf: bool = True,
                    smooth_idf: bool = True,
                    sublinear_tf: bool = False) -> csr_matrix:
        """
        Apply TF-IDF transformation to the matrix.
        
        TF-IDF helps identify codes that are distinctive for specific patients
        by down-weighting codes that appear in many patients.
        
        Args:
            matrix: Input sparse matrix (n_patients, n_codes)
            norm: Norm used to normalize term vectors ('l1', 'l2', or None)
            use_idf: Enable inverse-document-frequency reweighting
            smooth_idf: Smooth idf weights by adding 1 to document frequencies
            sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf))
            
        Returns:
            Transformed sparse matrix
        """
        self.tfidf_transformer = TfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf
        )
        
        transformed = self.tfidf_transformer.fit_transform(matrix)
        return transformed
    
    @staticmethod
    def normalize_matrix(matrix: csr_matrix, 
                        norm: Literal['l1', 'l2', 'max'] = 'l2') -> csr_matrix:
        """
        Normalize matrix rows.
        
        Args:
            matrix: Input sparse matrix
            norm: The norm to use ('l1', 'l2', or 'max')
            
        Returns:
            Normalized sparse matrix
        """
        return normalize(matrix, norm=norm, axis=1)
    
    @staticmethod
    def binarize_matrix(matrix: csr_matrix) -> csr_matrix:
        """
        Convert matrix to binary (presence/absence).
        
        Args:
            matrix: Input sparse matrix
            
        Returns:
            Binarized sparse matrix
        """
        binary = matrix.copy()
        binary.data = np.ones_like(binary.data)
        return binary
    
    def get_idf_scores(self) -> np.ndarray:
        """
        Get IDF scores for each code (after fitting TF-IDF).
        Higher scores indicate rarer codes.
        
        Returns:
            Array of IDF scores
        """
        if self.tfidf_transformer is None:
            raise ValueError("TF-IDF not fitted yet. Call apply_tfidf() first.")
        return self.tfidf_transformer.idf_
