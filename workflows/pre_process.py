"""
Preprocess FHIR data once:
- Load patients (uses cache in data/_patients_cache.pkl)
- Build patient-code matrix
- Apply TF-IDF
- Apply SVD (reduced representation)

Saves artifacts to: results/pre_process/artifacts/

Run:
  python -u preprocess.py

Optional:
  python -u preprocess.py --n_components 30 --data_dir data
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
import pickle
from typing import Literal, Tuple, Dict

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.pipeline import FHIRClusteringPipeline
from fhir_clustering.data_structures import CodeSystem


def run_preprocess_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="results/pre_process/artifacts")
    parser.add_argument("--n_components", type=int, default=30)
    parser.add_argument("--apply_tfidf", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load patients (cached)
    patients = FHIRParser.load_directory(args.data_dir, use_cache=True)
    if not patients:
        print("No patients found.")
        return
    print(f"Patients loaded: {len(patients)}")

    # 2) Preprocess with pipeline (we use a dummy clustering config; clustering result is ignored)
    pipe = FHIRClusteringPipeline(
        include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
        apply_tfidf=args.apply_tfidf,
        dimensionality_reduction="svd",
        n_components=args.n_components,
        clustering_method="kmeans",
        n_clusters=2,  # dummy, just to pass pipeline requirements
    )
    pipe.fit(patients)

    if pipe.matrix_builder is None or pipe.original_matrix is None:
        raise RuntimeError("Pipeline did not build the original matrix properly.")

    if pipe.transformed_matrix is None:
        raise RuntimeError("Pipeline did not produce transformed_matrix.")

    if pipe.reduced_data is None:
        raise RuntimeError("Pipeline did not produce reduced_data (SVD).")

    # 3) Save artifacts
    # 3.1 Reduced features
    X_reduced = np.asarray(pipe.reduced_data, dtype=np.float32)
    np.save(os.path.join(args.out_dir, "X_reduced.npy"), X_reduced)
    print(f"Saved: {args.out_dir}/X_reduced.npy  shape={X_reduced.shape}")

    # 3.2 Patient ids (ordered exactly like matrix rows)
    patient_ids = [pipe.matrix_builder.get_patient_id(i) for i in range(X_reduced.shape[0])]
    pd.DataFrame({"patient_id": patient_ids}).to_csv(os.path.join(args.out_dir, "patient_ids.csv"), index=False)
    print(f"Saved: {args.out_dir}/patient_ids.csv")

    # 3.3 Code names (ordered exactly like matrix columns)
    # Helpful later for interpretation if needed.
    n_codes = pipe.original_matrix.shape[1]
    code_names = [pipe.matrix_builder.get_code_name(j) for j in range(n_codes)]
    pd.DataFrame({"code": code_names}).to_csv(os.path.join(args.out_dir, "code_names.csv"), index=False)
    print(f"Saved: {args.out_dir}/code_names.csv  n_codes={n_codes}")

    # 3.4 Matrices (sparse) "au cas où"
    save_npz(os.path.join(args.out_dir, "original_matrix.npz"), pipe.original_matrix)
    save_npz(os.path.join(args.out_dir, "tfidf_matrix.npz"), pipe.transformed_matrix)
    print(f"Saved: {args.out_dir}/original_matrix.npz")
    print(f"Saved: {args.out_dir}/tfidf_matrix.npz")

    # 3.5 SVD explained variance
    if pipe.dim_reducer is not None:
        explained = np.asarray(pipe.dim_reducer.get_explained_variance(), dtype=np.float64)
        cumulative = np.cumsum(explained)
        np.save(os.path.join(args.out_dir, "svd_explained_variance.npy"), explained)
        np.save(os.path.join(args.out_dir, "svd_cumulative_variance.npy"), cumulative)
        pd.DataFrame({
            "component": np.arange(1, len(explained) + 1),
            "explained_variance": explained,
            "cumulative": cumulative,
        }).to_csv(os.path.join(args.out_dir, "svd_variance.csv"), index=False)
        print(f"Saved: {args.out_dir}/svd_variance.csv")

    config = {
        "data_dir": args.data_dir,
        "n_patients": int(X_reduced.shape[0]),
        "n_codes": int(n_codes),
        "apply_tfidf": bool(args.apply_tfidf),
        "dimensionality_reduction": "svd",
        "n_components": int(args.n_components),
        "include_systems": ["SNOMED", "LOINC", "RXNORM"],
    }
    with open(os.path.join(args.out_dir, "preprocessing_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {args.out_dir}/preprocessing_config.json")

    print("\nPreprocess complete.")

FHIR_SNOMED_SYSTEM = "http://snomed.info/sct"
FHIR_LOINC_SYSTEM = "http://loinc.org"
FHIR_RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"


def _load_terminology_layer(pkl_path: str = "results/terminology/terminology.pkl"):
    if not os.path.exists(pkl_path):
        raise RuntimeError(
            f"Terminology layer not found at {pkl_path}. "
            "Run build_terminology_layer() first."
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _build_domain_rollup_matrix(patients, terminology_layer) -> Tuple[np.ndarray, list]:
    """
    Build X_domain: (n_patients, n_domains + 1) dense matrix.
    Features: domain proportions + log1p(total_codes)
    Returns (X, feature_names)
    """
    # Collect domains present in mapping to define stable feature space
    # We build domain counts per patient first.
    patient_domain_counts = []
    all_domains = set()

    for p in patients:
        domain_counts: Dict[str, int] = {}
        total_mapped = 0

        for c in p.codes:
            # map to FHIR system string
            if c.system == CodeSystem.SNOMED:
                sys = FHIR_SNOMED_SYSTEM
            elif c.system == CodeSystem.LOINC:
                sys = FHIR_LOINC_SYSTEM
            elif c.system == CodeSystem.RXNORM:
                sys = FHIR_RXNORM_SYSTEM
            else:
                continue

            key = (sys, str(c.code))
            cid = terminology_layer.concept_id_by_key.get(key)
            if cid is None:
                continue  # unmapped
            domain = terminology_layer.domain_by_concept_id.get(cid)
            if not domain:
                continue

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            total_mapped += 1
            all_domains.add(domain)

        patient_domain_counts.append((domain_counts, total_mapped))

    domains_sorted = sorted(all_domains)
    feature_names = [f"domain:{d}" for d in domains_sorted] + ["log1p_total_codes"]

    # Build matrix
    n = len(patients)
    d = len(domains_sorted) + 1
    X = np.zeros((n, d), dtype=np.float32)

    for i, (counts, total) in enumerate(patient_domain_counts):
        denom = float(total) if total > 0 else 1.0
        for j, dom in enumerate(domains_sorted):
            X[i, j] = float(counts.get(dom, 0)) / denom  # proportion
        X[i, -1] = np.log1p(float(total))

    return X, feature_names

def run_preprocess(
    data_dir: str = "data",
    out_dir: str = "results/pre_process",  # <- parent dir now
    n_components: int = 30,
    apply_tfidf: bool = True,
    force: bool = False,
    feature_mode: Literal["raw_codes", "domain_rollup"] = "raw_codes",
    terminology_pkl: str = "results/terminology/terminology.pkl",
) -> dict:
    """
    Preprocess in two modes:
      - raw_codes: patient-code matrix -> TFIDF -> SVD -> X_reduced.npy
      - domain_rollup: patient -> domain proportions (+log1p total) -> X.npy

    Artifacts written to:
      results/pre_process/<feature_mode>/artifacts/

    Returns a dict with x_path, patient_ids_path, feature_names_path, config...
    """
    artifacts_dir = os.path.join(out_dir, feature_mode, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Standardized artifact names
    x_path = os.path.join(artifacts_dir, "X.npy" if feature_mode == "domain_rollup" else "X_reduced.npy")
    cfg_path = os.path.join(artifacts_dir, "preprocessing_config.json")
    patient_ids_path = os.path.join(artifacts_dir, "patient_ids.csv")
    feature_names_path = os.path.join(artifacts_dir, "feature_names.csv")

    # Cache
    if (not force) and os.path.exists(x_path) and os.path.exists(cfg_path) and os.path.exists(patient_ids_path) and os.path.exists(feature_names_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "out_dir": artifacts_dir,
            "x_path": x_path,
            "patient_ids_path": patient_ids_path,
            "feature_names_path": feature_names_path,
            "config_path": cfg_path,
            "config": cfg,
        }

    # Load patients
    patients = FHIRParser.load_directory(data_dir, use_cache=True)
    if not patients:
        raise RuntimeError("No patients found.")

    # ----------------------------
    # Mode 1: raw_codes (existing)
    # ----------------------------
    if feature_mode == "raw_codes":
        pipe = FHIRClusteringPipeline(
            include_systems=[CodeSystem.SNOMED, CodeSystem.LOINC, CodeSystem.RXNORM],
            apply_tfidf=apply_tfidf,
            dimensionality_reduction="svd",
            n_components=n_components,
            clustering_method="kmeans",
            n_clusters=2,
        )
        pipe.fit(patients)

        if pipe.matrix_builder is None or pipe.original_matrix is None or pipe.transformed_matrix is None or pipe.reduced_data is None:
            raise RuntimeError("Preprocess pipeline did not produce expected artifacts.")

        X_reduced = np.asarray(pipe.reduced_data, dtype=np.float32)
        np.save(x_path, X_reduced)

        # patient ids order
        patient_ids = [pipe.matrix_builder.get_patient_id(i) for i in range(X_reduced.shape[0])]
        pd.DataFrame({"patient_id": patient_ids}).to_csv(patient_ids_path, index=False)

        # feature names = codes
        n_codes = pipe.original_matrix.shape[1]
        code_names = [pipe.matrix_builder.get_code_name(j) for j in range(n_codes)]
        pd.DataFrame({"feature": code_names}).to_csv(feature_names_path, index=False)

        # optional legacy artifacts (keep them for now)
        save_npz(os.path.join(artifacts_dir, "original_matrix.npz"), pipe.original_matrix)
        save_npz(os.path.join(artifacts_dir, "tfidf_matrix.npz"), pipe.transformed_matrix)

        if pipe.dim_reducer is not None:
            explained = np.asarray(pipe.dim_reducer.get_explained_variance(), dtype=np.float64)
            cumulative = np.cumsum(explained)
            np.save(os.path.join(artifacts_dir, "svd_explained_variance.npy"), explained)
            np.save(os.path.join(artifacts_dir, "svd_cumulative_variance.npy"), cumulative)
            pd.DataFrame({
                "component": np.arange(1, len(explained) + 1),
                "explained_variance": explained,
                "cumulative": cumulative,
            }).to_csv(os.path.join(artifacts_dir, "svd_variance.csv"), index=False)

        config = {
            "feature_mode": "raw_codes",
            "data_dir": data_dir,
            "n_patients": int(X_reduced.shape[0]),
            "n_features": int(n_codes),
            "apply_tfidf": bool(apply_tfidf),
            "dimensionality_reduction": "svd",
            "n_components": int(n_components),
            "include_systems": ["SNOMED", "LOINC", "RXNORM"],
        }

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return {
            "out_dir": artifacts_dir,
            "x_path": x_path,
            "patient_ids_path": patient_ids_path,
            "feature_names_path": feature_names_path,
            "config_path": cfg_path,
            "config": config,
        }

    # --------------------------------
    # Mode 2: domain_rollup (new)
    # --------------------------------
    if feature_mode == "domain_rollup":
        layer = _load_terminology_layer(terminology_pkl)
        X, feature_names = _build_domain_rollup_matrix(patients, layer)

        np.save(x_path, X)

        # patient ids in FHIRParser order (patients list)
        patient_ids = [p.patient_id for p in patients]
        pd.DataFrame({"patient_id": patient_ids}).to_csv(patient_ids_path, index=False)
        pd.DataFrame({"feature": feature_names}).to_csv(feature_names_path, index=False)

        config = {
            "feature_mode": "domain_rollup",
            "data_dir": data_dir,
            "n_patients": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "apply_tfidf": False,               # not used here
            "dimensionality_reduction": None,   # not used here (for now)
            "include_systems": ["SNOMED", "LOINC", "RXNORM"],
            "terminology_pkl": terminology_pkl,
        }

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return {
            "out_dir": artifacts_dir,
            "x_path": x_path,
            "patient_ids_path": patient_ids_path,
            "feature_names_path": feature_names_path,
            "config_path": cfg_path,
            "config": config,
        }

    raise ValueError(f"Unknown feature_mode: {feature_mode}")