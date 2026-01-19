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

        # Find Patient resource safely
        patient_resource = None
        for e in entries:
            res = e.get("resource")
            if res and res.get("resourceType") == "Patient":
                patient_resource = res
                break

        patient_id = patient_resource.get("id") if patient_resource else os.path.basename(file_path)
        record = PatientRecord(patient_id=patient_id)

        # Extract codes from relevant resource types
        for entry in entries:
            resource = entry.get("resource", {})
            res_type = resource.get("resourceType")

            # CONDITIONS & PROCEDURES (SNOMED)
            if res_type in ["Condition", "Procedure", "Encounter"]:
                FHIRParser._extract_codes(resource, record, CodeSystem.SNOMED, ["snomed.info"])

            # OBSERVATIONS (LOINC)
            elif res_type == "Observation":
                FHIRParser._extract_codes(resource, record, CodeSystem.LOINC, ["loinc.org"])

            # MEDICATIONS (RxNorm)
            elif res_type == "Medication":
                FHIRParser._extract_codes(resource, record, CodeSystem.RXNORM, ["rxnorm", "nlm.nih.gov"])

        return record

    @staticmethod
    def _extract_codes(resource: Dict, record: PatientRecord, target_system: CodeSystem, url_keywords: List[str]):
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
                    record.add_code(MedicalCode(code=str(code_val), system=target_system, display=display))

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