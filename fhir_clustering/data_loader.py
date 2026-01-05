"""
Data loading and parsing utilities for FHIR patient data.
"""

import pandas as pd
from typing import List, Dict, Any
from .data_structures import PatientRecord, MedicalCode, CodeSystem


class FHIRDataLoader:
    """Load and parse FHIR patient data into structured format."""
    
    @staticmethod
    def from_dataframe(df: pd.DataFrame, 
                       patient_id_col: str = 'patient_id',
                       code_col: str = 'code',
                       system_col: str = 'system',
                       display_col: str = 'display') -> List[PatientRecord]:
        """
        Load patient records from a pandas DataFrame.
        
        Expected DataFrame columns:
        - patient_id: unique patient identifier
        - code: medical code value
        - system: code system (SNOMED, LOINC, RxNorm)
        - display: human-readable description (optional)
        
        Args:
            df: Input DataFrame with patient and code data
            patient_id_col: Name of patient ID column
            code_col: Name of code column
            system_col: Name of system column
            display_col: Name of display column
            
        Returns:
            List of PatientRecord objects
        """
        patients: Dict[str, PatientRecord] = {}
        
        for _, row in df.iterrows():
            pid = str(row[patient_id_col])
            code_val = str(row[code_col])
            system_val = row[system_col]
            display_val = row.get(display_col, '') if display_col in df.columns else ''
            
            # Parse system
            try:
                if isinstance(system_val, str):
                    system = CodeSystem[system_val.upper()]
                else:
                    system = system_val
            except (KeyError, AttributeError):
                continue  # Skip invalid systems
            
            # Create or get patient record
            if pid not in patients:
                patients[pid] = PatientRecord(patient_id=pid)
            
            # Add code to patient
            medical_code = MedicalCode(
                code=code_val,
                system=system,
                display=display_val
            )
            patients[pid].add_code(medical_code)
        
        return list(patients.values())
    
    @staticmethod
    def from_dict_list(data: List[Dict[str, Any]]) -> List[PatientRecord]:
        """
        Load patient records from a list of dictionaries.
        
        Each dictionary should have:
        - patient_id: unique patient identifier
        - codes: list of dicts with 'code', 'system', 'display'
        
        Args:
            data: List of patient data dictionaries
            
        Returns:
            List of PatientRecord objects
        """
        patients = []
        
        for item in data:
            patient = PatientRecord(patient_id=str(item['patient_id']))
            
            for code_data in item.get('codes', []):
                try:
                    system = CodeSystem[code_data['system'].upper()]
                    medical_code = MedicalCode(
                        code=str(code_data['code']),
                        system=system,
                        display=code_data.get('display', '')
                    )
                    patient.add_code(medical_code)
                except (KeyError, AttributeError):
                    continue
            
            patients.append(patient)
        
        return patients
