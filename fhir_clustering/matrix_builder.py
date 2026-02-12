"""
Matrix construction for patient × medical code representation.
Handles sparse data efficiently using scipy.sparse matrices.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, hstack
from typing import List, Tuple, Dict, Optional, Set
from .data_structures import PatientRecord, MedicalCode, CodeSystem
from sklearn.preprocessing import MinMaxScaler


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
        Build the patient × code sparse matrix + demographic features.
        
        Returns:
            Sparse matrix (CSR format) of shape (n_patients, n_codes + 2)
        """
        # 1. Build mappings and collect codes (Logique existante)
        all_codes: Set[MedicalCode] = set()
        
        # Sort patients by ID for consistency
        sorted_patients = sorted(self.patients, key=lambda p: p.patient_id)
        self.patients = sorted_patients # Update internal list to sorted order
        
        for idx, patient in enumerate(self.patients):
            self.patient_id_to_idx[patient.patient_id] = idx
            self.idx_to_patient_id[idx] = patient.patient_id
            
            for code in patient.codes:
                if code.system in self.include_systems:
                    all_codes.add(code)
        
        # Build code index mapping
        # Tri alphabétique pour stabilité
        sorted_codes = sorted(list(all_codes), key=lambda c: str(c))
        
        self.code_to_idx = {}
        self.idx_to_code = {}
        
        for idx, code in enumerate(sorted_codes):
            code_key = str(code)
            self.code_to_idx[code_key] = idx
            self.idx_to_code[idx] = code
            
        n_codes_only = len(sorted_codes)
        n_patients = len(self.patients)
        
        # 2. Build Code Matrix (LIL format)
        matrix_lil = lil_matrix((n_patients, n_codes_only), dtype=np.float32)
        
        for patient_idx, patient in enumerate(self.patients):
            # Count codes for this patient
            for code in patient.codes:
                if code.system in self.include_systems:
                    code_key = str(code)
                    if code_key in self.code_to_idx:
                        code_idx = self.code_to_idx[code_key]
                        # On peut pondérer ici (ex: Condition=1.5, Observation=1.0)
                        # Pour l'instant on garde le count simple
                        matrix_lil[patient_idx, code_idx] += code.weight
        
        code_matrix_csr = matrix_lil.tocsr()
        
        # 3. Build Demographic Matrix (Dense -> Sparse)
        ages = []
        genders = []
        
        for p in self.patients:
            # Sécurité si age/gender manquent dans PatientRecord
            ages.append(getattr(p, 'age', 0.0))
            genders.append(getattr(p, 'gender', 0))
            
        ages_arr = np.array(ages).reshape(-1, 1)
        genders_arr = np.array(genders).reshape(-1, 1)
        
        # Normalisation Age (0-1)
        scaler = MinMaxScaler()
        ages_norm = scaler.fit_transform(ages_arr)
        
        # Pondération (Weights)
        AGE_WEIGHT = 2.0
        SEX_WEIGHT = 1.0
        
        ages_final = ages_norm * AGE_WEIGHT
        genders_final = genders_arr * SEX_WEIGHT
        
        # Création matrice démographique sparse
        demo_matrix = csr_matrix(np.hstack([ages_final, genders_final]))
        
        # 4. Merge (Horizontal Stack)
        self.matrix = hstack([code_matrix_csr, demo_matrix]).tocsr()
        
        # 5. Update Metadata (Add artificial codes for interpretation)
        # L'indexation continue après les codes médicaux
        age_idx = n_codes_only
        gender_idx = n_codes_only + 1
        
        # On injecte des "Faux" codes médicaux pour que l'interpréteur puisse afficher "DEMO_AGE"
        # Il faut que MedicalCode supporte des strings arbitraires, ou on triche avec un système custom
        # Ici on suppose que idx_to_code peut stocker des objets ou strings. 
        # Si interpretation.py attend un MedicalCode, il faudra créer un MedicalCode artificiel.
        
        from .data_structures import MedicalCode, CodeSystem
        
        code_age = MedicalCode(system=CodeSystem.SNOMED, code="DEMO_AGE", display="Age (Normalized)")
        code_gender = MedicalCode(system=CodeSystem.SNOMED, code="DEMO_GENDER_MALE", display="Gender (Male=1)")
        
        self.idx_to_code[age_idx] = code_age
        self.code_to_idx[str(code_age)] = age_idx
        
        self.idx_to_code[gender_idx] = code_gender
        self.code_to_idx[str(code_gender)] = gender_idx
        
        return self.matrix
    
    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names (codes + demographics) ordered by column index."""
        names = []
        # On itère jusqu'au max index connu
        max_idx = max(self.idx_to_code.keys()) if self.idx_to_code else -1
        
        for i in range(max_idx + 1):
            obj = self.idx_to_code.get(i, f"Unknown_{i}")
            # Si c'est un MedicalCode, on le stringify proprement
            # Si c'est déjà un str (ex: DEMO_AGE), on le garde
            names.append(str(obj))
        return names
    
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
