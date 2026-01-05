"""
Data structures for FHIR patient data representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from enum import Enum


class CodeSystem(Enum):
    """Medical code systems supported."""
    SNOMED = "SNOMED"
    LOINC = "LOINC"
    RXNORM = "RxNorm"


@dataclass
class MedicalCode:
    """Represents a single medical code."""
    code: str
    system: CodeSystem
    display: str = ""
    
    def __hash__(self):
        return hash((self.code, self.system))
    
    def __eq__(self, other):
        if not isinstance(other, MedicalCode):
            return False
        return self.code == other.code and self.system == other.system
    
    def __str__(self):
        return f"{self.system.value}:{self.code}"


@dataclass
class PatientRecord:
    """Represents a patient with their medical codes."""
    patient_id: str
    codes: List[MedicalCode] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_code(self, code: MedicalCode):
        """Add a medical code to this patient."""
        self.codes.append(code)
    
    def get_unique_codes(self) -> Set[MedicalCode]:
        """Get unique codes for this patient."""
        return set(self.codes)
    
    def get_codes_by_system(self, system: CodeSystem) -> List[MedicalCode]:
        """Get all codes from a specific system."""
        return [code for code in self.codes if code.system == system]
