"""
Feature engineering for patient data.
Includes TF-IDF transformation and normalization.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from typing import Literal, Tuple, List


class FeatureTransformer:
    """
    Transform patient-code matrices using various techniques.
    """
    
    def __init__(self):
        self.tfidf_transformer = None
        self.feature_mask = None # Masque booléen des colonnes conservées
        
    def calculate_frequency_mask(self, matrix: csr_matrix, 
                               min_df: float = 0.0, 
                               max_df: float = 1.0) -> np.ndarray:
        """
        Calcule un masque pour filtrer les features (codes) selon leur fréquence.
        Indispensable après un Roll-up pour éliminer les parents universels (ex: 'Disease').

        Args:
            matrix: Matrice sparse (patients x codes)
            min_df: Fréquence min (ex: 0.01 = présent chez au moins 1% des patients)
            max_df: Fréquence max (ex: 0.95 = présent chez moins de 95% des patients)

        Returns:
            np.ndarray: Masque booléen (True = garder la colonne, False = supprimer)
        """
        n_patients = matrix.shape[0]
        
        # Binariser pour compter les patients (et non le nombre d'occurrences total)
        binary = (matrix > 0).astype(int)
        
        # Somme sur l'axe 0 (colonnes) -> Nombre de patients par code
        # np.array(...).flatten() convertit la matrice 1xN en array 1D
        doc_freqs = np.array(binary.sum(axis=0)).flatten()
        
        # Calcul des seuils absolus
        min_count = int(min_df * n_patients) if isinstance(min_df, float) else min_df
        max_count = int(max_df * n_patients) if isinstance(max_df, float) else max_df
        
        # Création du masque
        # On garde ce qui est > min ET < max
        self.feature_mask = (doc_freqs >= min_count) & (doc_freqs <= max_count)
        
        n_kept = self.feature_mask.sum()
        print(f"Feature Selection: Keeping {n_kept}/{matrix.shape[1]} features "
              f"(min_df={min_df}, max_df={max_df})")
        
        return self.feature_mask

    def apply_feature_selection(self, matrix: csr_matrix) -> csr_matrix:
        """Applique le masque calculé précédemment pour réduire la matrice."""
        if self.feature_mask is None:
            return matrix
        return matrix[:, self.feature_mask]

    def apply_tfidf(self, matrix: csr_matrix, 
                    norm: Literal['l1', 'l2'] = 'l2',
                    use_idf: bool = True,
                    smooth_idf: bool = True,
                    sublinear_tf: bool = False) -> csr_matrix:
        """
        Apply TF-IDF transformation to the matrix.
        
        TF-IDF helps identify codes that are distinctive for specific patients
        by down-weighting codes that appear in many patients.
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
        """Normalize matrix rows."""
        return normalize(matrix, norm=norm, axis=1)
    
    @staticmethod
    def binarize_matrix(matrix: csr_matrix) -> csr_matrix:
        """Convert matrix to binary (presence/absence)."""
        binary = matrix.copy()
        binary.data = np.ones_like(binary.data)
        return binary
    
    def get_idf_scores(self) -> np.ndarray:
        """
        Get IDF scores for each code.
        Higher scores indicate rarer codes.
        """
        if self.tfidf_transformer is None:
            raise ValueError("TF-IDF not fitted yet. Call apply_tfidf() first.")
        return self.tfidf_transformer.idf_