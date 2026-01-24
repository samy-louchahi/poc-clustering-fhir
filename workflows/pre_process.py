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


def run_preprocess(
    data_dir: str = "data",
    out_dir: str = "results/pre_process/artifacts",
    n_components: int = 30,
    apply_tfidf: bool = True,
    force: bool = False,
) -> dict:
    """
    Builds matrix, TF-IDF, SVD.
    Saves artifacts and returns paths + metadata.
    If force=False and X_reduced.npy exists, reuse it.
    """
    os.makedirs(out_dir, exist_ok=True)

    x_path = os.path.join(out_dir, "X_reduced.npy")
    cfg_path = os.path.join(out_dir, "preprocessing_config.json")

    if (not force) and os.path.exists(x_path) and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "out_dir": out_dir,
            "x_path": x_path,
            "patient_ids_path": os.path.join(out_dir, "patient_ids.csv"),
            "code_names_path": os.path.join(out_dir, "code_names.csv"),
            "original_matrix_path": os.path.join(out_dir, "original_matrix.npz"),
            "tfidf_matrix_path": os.path.join(out_dir, "tfidf_matrix.npz"),
            "svd_variance_path": os.path.join(out_dir, "svd_variance.csv"),
            "config_path": cfg_path,
            "config": cfg,
        }

    patients = FHIRParser.load_directory(data_dir, use_cache=True)
    if not patients:
        raise RuntimeError("No patients found.")

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

    patient_ids = [pipe.matrix_builder.get_patient_id(i) for i in range(X_reduced.shape[0])]
    pd.DataFrame({"patient_id": patient_ids}).to_csv(os.path.join(out_dir, "patient_ids.csv"), index=False)

    n_codes = pipe.original_matrix.shape[1]
    code_names = [pipe.matrix_builder.get_code_name(j) for j in range(n_codes)]
    pd.DataFrame({"code": code_names}).to_csv(os.path.join(out_dir, "code_names.csv"), index=False)

    save_npz(os.path.join(out_dir, "original_matrix.npz"), pipe.original_matrix)
    save_npz(os.path.join(out_dir, "tfidf_matrix.npz"), pipe.transformed_matrix)

    if pipe.dim_reducer is not None:
        explained = np.asarray(pipe.dim_reducer.get_explained_variance(), dtype=np.float64)
        cumulative = np.cumsum(explained)
        np.save(os.path.join(out_dir, "svd_explained_variance.npy"), explained)
        np.save(os.path.join(out_dir, "svd_cumulative_variance.npy"), cumulative)
        pd.DataFrame({
            "component": np.arange(1, len(explained) + 1),
            "explained_variance": explained,
            "cumulative": cumulative,
        }).to_csv(os.path.join(out_dir, "svd_variance.csv"), index=False)

    config = {
        "data_dir": data_dir,
        "n_patients": int(X_reduced.shape[0]),
        "n_codes": int(n_codes),
        "apply_tfidf": bool(apply_tfidf),
        "dimensionality_reduction": "svd",
        "n_components": int(n_components),
        "include_systems": ["SNOMED", "LOINC", "RXNORM"],
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "out_dir": out_dir,
        "x_path": x_path,
        "patient_ids_path": os.path.join(out_dir, "patient_ids.csv"),
        "code_names_path": os.path.join(out_dir, "code_names.csv"),
        "original_matrix_path": os.path.join(out_dir, "original_matrix.npz"),
        "tfidf_matrix_path": os.path.join(out_dir, "tfidf_matrix.npz"),
        "svd_variance_path": os.path.join(out_dir, "svd_variance.csv"),
        "config_path": cfg_path,
        "config": config,
    }