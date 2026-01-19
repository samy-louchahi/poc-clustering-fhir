"""
Matrix construction for patient × medical code representation.
Handles sparse data efficiently using scipy.sparse matrices.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from typing import List, Tuple, Dict, Optional, Set
from .data_structures import PatientRecord, MedicalCode, CodeSystem


class PatientCodeMatrix:
    """
    Builds and manages patient × medical code matrices.
    Uses sparse matrices for efficient storage and computation.
    """
    
    def __init__(self, patients: List[PatientRecord], 
                 include_systems: Optional[List[CodeSystem]] = None):
        """
        Initialize matrix builder.
        
        Args:
            patients: List of patient records
            include_systems: List of code systems to include (default: all)
        """
        self.patients = patients
        self.include_systems = include_systems or [CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM]
        
        # Mappings
        self.patient_id_to_idx: Dict[str, int] = {}
        self.idx_to_patient_id: Dict[int, str] = {}
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_code: Dict[int, MedicalCode] = {}
        
        # The matrix
        self.matrix: Optional[csr_matrix] = None
        
    def build_matrix(self) -> csr_matrix:
        """
        Build the patient × code sparse matrix.
        
        Returns:
            Sparse matrix (CSR format) of shape (n_patients, n_codes)
        """
        # Build patient index mapping
        for idx, patient in enumerate(self.patients):
            self.patient_id_to_idx[patient.patient_id] = idx
            self.idx_to_patient_id[idx] = patient.patient_id
        
        # Collect all unique codes across patients
        all_codes: Set[MedicalCode] = set()
        for patient in self.patients:
            for code in patient.codes:
                if code.system in self.include_systems:
                    all_codes.add(code)
        
        # Build code index mapping
        for idx, code in enumerate(sorted(all_codes, key=lambda c: str(c))):
            code_key = str(code)
            self.code_to_idx[code_key] = idx
            self.idx_to_code[idx] = code
        
        # Build sparse matrix using LIL format for efficient construction
        n_patients = len(self.patients)
        n_codes = len(self.code_to_idx)
        matrix_lil = lil_matrix((n_patients, n_codes), dtype=np.int32)
        
        # Fill matrix with code counts
        for patient_idx, patient in enumerate(self.patients):
            code_counts: Dict[str, int] = {}
            for code in patient.codes:
                if code.system in self.include_systems:
                    code_key = str(code)
                    code_counts[code_key] = code_counts.get(code_key, 0) + 1
            
            for code_key, count in code_counts.items():
                code_idx = self.code_to_idx[code_key]
                matrix_lil[patient_idx, code_idx] = count
        
        # Convert to CSR format for efficient arithmetic operations
        self.matrix = matrix_lil.tocsr()
        return self.matrix
    
    def get_matrix(self) -> csr_matrix:
        """Get the built matrix."""
        if self.matrix is None:
            raise ValueError("Matrix not built yet. Call build_matrix() first.")
        return self.matrix
    
    def get_patient_vector(self, patient_id: str) -> np.ndarray:
        """Get the code vector for a specific patient."""
        if self.matrix is None:
            raise ValueError("Matrix not built yet. Call build_matrix() first.")
        patient_idx = self.patient_id_to_idx.get(patient_id)
        if patient_idx is None:
            raise ValueError(f"Patient {patient_id} not found")
        return self.matrix[patient_idx].toarray().ravel()
    
    def get_code_name(self, code_idx: int) -> str:
        """Get the code name for a given index."""
        code = self.idx_to_code.get(code_idx)
        if code is None:
            raise ValueError(f"Code index {code_idx} not found")
        return str(code)
    
    def get_patient_id(self, patient_idx: int) -> str:
        """Get patient ID for a given index."""
        return self.idx_to_patient_id.get(patient_idx, f"Unknown_{patient_idx}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert matrix to dense DataFrame (use with caution for large matrices)."""
        if self.matrix is None:
            raise ValueError("Matrix not built yet. Call build_matrix() first.")
        
        df = pd.DataFrame(
            self.matrix.toarray(),
            index=[self.idx_to_patient_id[i] for i in range(len(self.patients))],
            columns=[str(self.idx_to_code[i]) for i in range(len(self.code_to_idx))]
        )
        return df
    
    def get_matrix_stats(self) -> Dict[str, any]:
        """Get statistics about the matrix."""
        if self.matrix is None:
            raise ValueError("Matrix not built yet. Call build_matrix() first.")
        
        n_patients, n_codes = self.matrix.shape
        n_nonzero = self.matrix.nnz
        sparsity = 1 - (n_nonzero / (n_patients * n_codes))
        
        return {
            'n_patients': n_patients,
            'n_codes': n_codes,
            'n_nonzero_entries': n_nonzero,
            'sparsity': sparsity,
            'avg_codes_per_patient': n_nonzero / n_patients if n_patients > 0 else 0,
        }
    
    def filter_features(self, min_df: float = 0.0, max_df: float = 1.0):
        """
        Filtre les codes selon leur fréquence d'apparition.
        min_df: Fréquence minimale (ex: 0.01 pour 1%). Si int > 1, c'est un nombre absolu.
        max_df: Fréquence maximale (ex: 0.9 pour 90%).
        """
        if self.matrix is None:
            raise ValueError("Matrix not built yet.")

        n_patients = self.matrix.shape[0]
        
        # Conversion des seuils relatifs (%) en absolus
        min_count = int(min_df * n_patients) if isinstance(min_df, float) and min_df < 1.0 else int(min_df)
        max_count = int(max_df * n_patients) if isinstance(max_df, float) and max_df < 1.0 else int(max_df)
        
        print(f"Filtrage des features: conservation si présence entre {min_count} et {max_count} patients.")

        # Calcul de la fréquence documentaire (nombre de patients ayant le code)
        # On binarise temporairement pour ne compter qu'une fois par patient
        binary_matrix = self.matrix.copy()
        binary_matrix.data[:] = 1
        doc_freqs = np.array(binary_matrix.sum(axis=0)).flatten()
        
        # Création du masque de conservation
        mask = (doc_freqs >= min_count) & (doc_freqs <= max_count)
        
        # Application du filtre sur la matrice
        old_shape = self.matrix.shape
        self.matrix = self.matrix[:, mask]
        
        # RECONSTRUCTION DES MAPPINGS (Crucial pour l'interprétation)
        kept_indices = np.where(mask)[0]
        new_idx_to_code = {}
        new_code_to_idx = {}
        
        for new_idx, old_idx in enumerate(kept_indices):
            code_obj = self.idx_to_code[old_idx]
            new_idx_to_code[new_idx] = code_obj
            new_code_to_idx[str(code_obj)] = new_idx
            
        self.idx_to_code = new_idx_to_code
        self.code_to_idx = new_code_to_idx
        
        print(f"Features réduites : {old_shape[1]} -> {self.matrix.shape[1]}")
