import json
import glob
import os
from datetime import datetime
from typing import List, Dict
from .data_structures import PatientRecord, MedicalCode, CodeSystem

class FHIRParser:
    """
    Parses FHIR JSON Bundles (Synthea format) into PatientRecords.
    """

    @staticmethod
    def parse_bundle(file_path: str) -> PatientRecord:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries = data.get('entry', [])
        
        # 1. Identifier le patient (Ressource Patient)
        patient_resource = next((e['resource'] for e in entries if e['resource']['resourceType'] == 'Patient'), None)
        patient_id = patient_resource['id'] if patient_resource else os.path.basename(file_path)
        
        record = PatientRecord(patient_id=patient_id)
        
        gender = patient_resource.get('gender')
        if gender:
            record.add_code(MedicalCode(
                code=f"GENDER_{gender.upper()}",
                system=CodeSystem.DEMOGRAPHICS,
                display=f"Gender: {gender}"
            ))
        
        birth_date = patient_resource.get('birthDate')
        if birth_date:
            try:
                birth_year = int(birth_date.split('-')[0])
                # On fixe l'année de référence (ex: 2026) pour que le résultat soit reproductible
                age = 2026 - birth_year
                
                # Création de buckets d'âge (0-10, 10-20, etc.)
                age_bucket = (age // 10) * 10
                record.add_code(MedicalCode(
                    code=f"AGE_{age_bucket}_{age_bucket+10}",
                    system=CodeSystem.DEMOGRAPHICS,
                    display=f"Age Group: {age_bucket}-{age_bucket+10}"
                ))
            except Exception:
                pass
        # 2. Scanner les ressources cliniques pour extraire les codes
        for entry in entries:
            resource = entry.get('resource', {})
            res_type = resource.get('resourceType')
            
            # --- CONDITIONS & PROCEDURES (SNOMED) ---
            if res_type in ['Condition', 'Procedure', 'Encounter']:
                FHIRParser._extract_codes(resource, record, CodeSystem.SNOMED, ['snomed.info'])
                
            # --- OBSERVATIONS (LOINC) ---
            elif res_type == 'Observation':
                FHIRParser._extract_codes(resource, record, CodeSystem.LOINC, ['loinc.org'])
                
            # --- MEDICATIONS (RxNorm) ---
            elif res_type == 'Medication':
                # Synthea met souvent le code directement dans la ressource Medication
                FHIRParser._extract_codes(resource, record, CodeSystem.RXNORM, ['rxnorm', 'nlm.nih.gov'])
                
            # Note: MedicationRequest pointe souvent vers une Medication, 
            # mais scanner 'Medication' suffit généralement pour avoir l'inventaire.

        return record

    @staticmethod
    def _extract_codes(resource: Dict, record: PatientRecord, target_system: CodeSystem, url_keywords: List[str]):
        """Helper to extract codings matching a specific system."""
        # Les codes sont généralement sous 'code' -> 'coding' -> liste
        # Parfois sous 'type' (Encounter) ou 'vaccineCode' (Immunization)
        
        code_element = resource.get('code') or resource.get('type') or resource.get('vaccineCode')
        
        # Cas spécial Encounter (c'est une liste parfois)
        if isinstance(code_element, list):
            code_element = code_element[0] if code_element else None
            
        if not code_element or 'coding' not in code_element:
            return

        for coding in code_element['coding']:
            system_url = coding.get('system', '').lower()
            # Vérifier si l'URL du système correspond (ex: "http://loinc.org" contient "loinc")
            if any(k in system_url for k in url_keywords):
                code_val = coding.get('code')
                display = coding.get('display', 'Unknown')
                
                if code_val:
                    record.add_code(MedicalCode(
                        code=str(code_val),
                        system=target_system,
                        display=display
                    ))

    @staticmethod
    def load_directory(directory_path: str) -> List[PatientRecord]:
        """Loads all JSON files from a directory."""
        records = []
        files = glob.glob(os.path.join(directory_path, "*.json"))
        print(f"Chargement de {len(files)} fichiers patients depuis {directory_path}...")
        
        for f in files:
            try:
                record = FHIRParser.parse_bundle(f)
                records.append(record)
            except Exception as e:
                print(f"Erreur sur le fichier {f}: {e}")
                
        return records