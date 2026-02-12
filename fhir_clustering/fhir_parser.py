from datetime import datetime
import json
import glob
import os
import time
import pickle
from typing import List, Dict, Optional

from .data_structures import PatientRecord, MedicalCode, CodeSystem


class FHIRParser:
    """
    Parses FHIR JSON Bundles (Synthea format) into PatientRecords.
    """

    @staticmethod
    def parse_bundle(file_path: str) -> PatientRecord:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entry", [])

        # 1. Find Patient resource safely
        patient_resource = None
        for e in entries:
            res = e.get("resource")
            if res and res.get("resourceType") == "Patient":
                patient_resource = res
                break
        
        # 2. Extract Demographics (ID, Age, Gender)
        patient_id = os.path.basename(file_path) # Valeur par défaut
        age = 0.0
        gender = 0

        if patient_resource:
            patient_id = patient_resource.get("id", patient_id)
            
            # Extraction des données brutes
            birth_date = patient_resource.get("birthDate", "")
            gender_str = patient_resource.get("gender", "")
            
            # Conversion via les utilitaires
            age = FHIRParser._calculate_age(birth_date)
            gender = FHIRParser._parse_gender(gender_str)

        # 3. Initialize Record with new fields
        # Note: Assurez-vous d'avoir mis à jour data_structures.py pour accepter age/gender
        record = PatientRecord(patient_id=patient_id, age=age, gender=gender)

        # 4. Extract codes from relevant resource types (Code existant inchangé)
        for entry in entries:
            resource = entry.get("resource", {})
            res_type = resource.get("resourceType")

            # CONDITIONS & PROCEDURES (SNOMED)
            if res_type in ["Condition", "Procedure", "Encounter"]:
                FHIRParser._extract_codes(resource, record, CodeSystem.SNOMED, ["snomed.info"], weight=1.5)

            # OBSERVATIONS (LOINC)
            elif res_type == "Observation":
                FHIRParser._extract_codes(resource, record, CodeSystem.LOINC, ["loinc.org"], weight=0.8)

            # MEDICATIONS (RxNorm)
            elif res_type == "Medication":
                FHIRParser._extract_codes(resource, record, CodeSystem.RXNORM, ["rxnorm", "nlm.nih.gov"], weight=1.0)

        return record

    @staticmethod
    def _extract_codes(resource: Dict, record: PatientRecord, target_system: CodeSystem, url_keywords: List[str], weight: float = 1.0):
        """
        Extract codings matching a specific system.
        Codes can be under 'code', or sometimes 'type'/'vaccineCode'.
        """
        code_element = resource.get("code") or resource.get("type") or resource.get("vaccineCode")

        # Sometimes Encounter 'type' is a list
        if isinstance(code_element, list):
            code_element = code_element[0] if code_element else None

        if not code_element or "coding" not in code_element:
            return

        for coding in code_element["coding"]:
            system_url = (coding.get("system") or "").lower()
            if any(k in system_url for k in url_keywords):
                code_val = coding.get("code")
                display = coding.get("display", "Unknown")
                if code_val:
                    m_code = MedicalCode(code=str(code_val), system=target_system, display=display, weight=weight)
                    record.add_code(m_code)
    

    @staticmethod
    def _calculate_age(birth_date_str: str) -> float:
        """Calcule l'âge approximatif à partir de la date de naissance."""
        if not birth_date_str:
            return 0.0
        try:
            # Format FHIR standard : YYYY-MM-DD
            birth = datetime.strptime(birth_date_str, "%Y-%m-%d")
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_gender(gender_str: str) -> int:
        """Encode le genre : Male=1, Autre=0."""
        return 1 if gender_str and gender_str.lower() == 'male' else 0

    @staticmethod
    def load_directory(directory_path: str, use_cache: bool = True) -> List[PatientRecord]:
        """
        Loads all patient JSON bundles from a directory.

        use_cache: if True, saves/loads parsed PatientRecord list to speed up reruns.
        """
        files = sorted(glob.glob(os.path.join(directory_path, "*.json")))
        print(f"Chargement de {len(files)} fichiers patients depuis {directory_path}...", flush=True)

        # Cache (highly recommended)
        cache_path = os.path.join(directory_path, "_patients_cache.pkl")
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cache: {cache_path}", flush=True)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        records: List[PatientRecord] = []
        t0 = time.perf_counter()

        for i, fp in enumerate(files, start=1):
            try:
                record = FHIRParser.parse_bundle(fp)
                records.append(record)
            except Exception as e:
                print(f"[ERROR] fichier={fp} -> {e}", flush=True)

            if i % 25 == 0 or i == len(files):
                dt = time.perf_counter() - t0
                print(f"  parsed {i}/{len(files)} files ({dt:.1f}s)", flush=True)

        print(f"Done in {time.perf_counter() - t0:.1f}s", flush=True)

        if use_cache:
            print(f"Saving cache: {cache_path}", flush=True)
            with open(cache_path, "wb") as f:
                pickle.dump(records, f)

        return records