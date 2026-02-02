"""
Build a local "terminology layer" from OMOP/Athena CSVs, trimmed to the dataset codes.

Inputs:
- FHIR patients (via FHIRParser.load_directory)
- OMOP vocabulary CSVs located in ./terminology_omop/

Outputs in results/terminology/:
- concept_min.csv            (filtered concepts used by dataset)
- concept_map.csv            (system, code -> concept_id)
- ancestor_min.csv           (filtered SNOMED ancestor relationships for dataset concepts)
- terminology.pkl            (pickled dicts for fast lookup)
- terminology_summary.json   (stats)

This module is called by run_poc.py (no CLI needed).
"""

from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional, List
import csv
import logging

import pandas as pd
import networkx as nx  # <--- AJOUT MAJEUR POUR LA CLÔTURE TRANSITIVE

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.data_structures import CodeSystem
from pandas.errors import ParserError


# --- Constants for FHIR systems ---
FHIR_SNOMED_SYSTEM = "http://snomed.info/sct"
FHIR_LOINC_SYSTEM = "http://loinc.org"
FHIR_RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"

# --- OMOP vocab ids ---
OMOP_SNOMED = "SNOMED"
OMOP_LOINC = "LOINC"
OMOP_RXNORM = "RxNorm"

logger = logging.getLogger(__name__)

@dataclass
class TerminologyLayer:
    # system+code -> concept_id
    concept_id_by_key: Dict[Tuple[str, str], int]
    # concept_id -> domain_id/name/vocabulary_id
    domain_by_concept_id: Dict[int, str]
    name_by_concept_id: Dict[int, str]
    vocab_by_concept_id: Dict[int, str]
    code_by_concept_id: Dict[int, str]

    # SNOMED: descendant_concept_id -> list of (ancestor_concept_id, min_levels_of_separation)
    ancestors_by_descendant: Dict[int, List[Tuple[int, int]]]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _detect_sep(path: str) -> str:
    # Heuristique simple : on compte les séparateurs sur la 1ère ligne
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        candidates = [",", "\t", ";", "|"]
        counts = {c: head.count(c) for c in candidates}
        # choisir celui qui apparaît le plus
        sep = max(counts, key=counts.get)
        return sep
    except FileNotFoundError:
        return ","


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip + lower + remove BOM if present
    df.columns = [
        str(c).replace("\ufeff", "").strip().lower()
        for c in df.columns
    ]
    return df


def _load_concept_csv(concept_path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(concept_path):
        logger.warning(f"File not found: {concept_path}")
        return pd.DataFrame()

    sep = _detect_sep(concept_path)

    def _read(quoting_mode: int, on_bad_lines: str):
        df = pd.read_csv(
            concept_path,
            dtype=str,
            sep=sep,
            engine="python",
            quoting=quoting_mode,
            escapechar="\\",
            on_bad_lines=on_bad_lines,   # "error" | "warn" | "skip"
        )
        df = _normalize_columns(df)
        return df

    # 1) Try normal CSV parsing (quotes respected)
    try:
        df = _read(csv.QUOTE_MINIMAL, on_bad_lines="error")
    except ParserError:
        # 2) Fallback: ignore quotes entirely
        try:
            df = _read(csv.QUOTE_NONE, on_bad_lines="warn")
        except ParserError:
            df = _read(csv.QUOTE_NONE, on_bad_lines="skip")

    if usecols is not None:
        wanted = [c.strip().lower() for c in usecols]
        # Only select columns that actually exist
        available = [c for c in wanted if c in df.columns]
        if len(available) < len(wanted):
            missing = set(wanted) - set(available)
            logger.warning(f"Missing columns in {concept_path}: {missing}")
        df = df[available]

    return df


def _select_best_concept_row(df_same_code: pd.DataFrame) -> pd.Series:
    """
    Prefer active concepts (invalid_reason is NaN/empty).
    If multiple, prefer standard_concept == 'S' if available.
    Fallback: first row.
    """
    d = df_same_code.copy()

    # normalize empties
    for col in ["invalid_reason", "standard_concept"]:
        if col in d.columns:
            d[col] = d[col].fillna("").astype(str)

    # active first
    if "invalid_reason" in d.columns:
        active = d[d["invalid_reason"] == ""]
    else:
        active = d # Assume active if no status column

    if not active.empty:
        # prefer standard
        if "standard_concept" in active.columns:
            standard = active[active["standard_concept"] == "S"]
            if not standard.empty:
                return standard.iloc[0]
        return active.iloc[0]

    return d.iloc[0]


def _extract_codes_from_patients(patients) -> Dict[str, Set[str]]:
    """
    Returns codes_by_fhir_system used in the dataset.
    """
    codes_by_system: Dict[str, Set[str]] = {
        FHIR_SNOMED_SYSTEM: set(),
        FHIR_LOINC_SYSTEM: set(),
        FHIR_RXNORM_SYSTEM: set(),
    }

    for p in patients:
        for c in p.codes:
            if c.system == CodeSystem.SNOMED:
                codes_by_system[FHIR_SNOMED_SYSTEM].add(str(c.code))
            elif c.system == CodeSystem.LOINC:
                codes_by_system[FHIR_LOINC_SYSTEM].add(str(c.code))
            elif c.system == CodeSystem.RXNORM:
                codes_by_system[FHIR_RXNORM_SYSTEM].add(str(c.code))

    return codes_by_system


# --- NOUVELLE FONCTION POUR LA CLÔTURE TRANSITIVE ---
def _build_transitive_closure_from_relations(
    concept_ids: Set[int], 
    omop_dir: str,
    max_distance: int
) -> pd.DataFrame:
    """
    Construit la hiérarchie ancêtre/descendant en utilisant CONCEPT_RELATIONSHIP.csv
    si CONCEPT_ANCESTOR.csv est absent ou incomplet.
    Utilise NetworkX pour calculer la clôture transitive.
    """
    rel_path = os.path.join(omop_dir, "CONCEPT_RELATIONSHIP.csv")
    if not os.path.exists(rel_path):
        logger.warning(f"Neither CONCEPT_ANCESTOR nor CONCEPT_RELATIONSHIP found in {omop_dir}")
        return pd.DataFrame(columns=["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"])

    logger.info("Building transitive closure from CONCEPT_RELATIONSHIP.csv...")
    
    # 1. Charger les relations "Is a" (ID: 418048695 dans OMOP pour SNOMED Is-a)
    # Note: Dans OMOP, 'Is a' est souvent relationship_id='Is a'
    cols = ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"]
    df_rel = _load_concept_csv(rel_path, usecols=cols)
    
    # Filtre: Active relationships only + Type 'Is a'
    # concept_id_1 (Child) -> Is a -> concept_id_2 (Parent)
    mask = (df_rel["relationship_id"] == "Is a") & (df_rel["invalid_reason"].isna() | (df_rel["invalid_reason"] == ""))
    df_isa = df_rel[mask]

    # 2. Construire le graphe
    G = nx.DiGraph()
    # Ajout des arêtes: Enfant -> Parent
    edges = list(zip(df_isa["concept_id_1"].astype(int), df_isa["concept_id_2"].astype(int)))
    G.add_edges_from(edges)

    # 3. Calculer les ancêtres pour nos concepts d'intérêt
    rows = []
    
    # On ne calcule que pour les concepts présents dans le dataset (pour gagner du temps)
    # Mais on doit traverser tout le graphe
    nodes_of_interest = [cid for cid in concept_ids if cid in G]

    for node in nodes_of_interest:
        # Ajout de soi-même (distance 0)
        rows.append({"ancestor_concept_id": node, "descendant_concept_id": node, "min_levels_of_separation": 0})
        
        # Récupération de tous les ancêtres (BFS pour avoir la distance minimale)
        # cutoff limite la profondeur de recherche
        ancestors = nx.single_source_shortest_path_length(G, node, cutoff=max_distance)
        
        for ancestor, dist in ancestors.items():
            if dist > 0: # Déjà ajouté 0 au dessus
                rows.append({
                    "ancestor_concept_id": ancestor,
                    "descendant_concept_id": node,
                    "min_levels_of_separation": dist
                })

    return pd.DataFrame(rows)


def build_terminology_layer(
    data_dir: str = "data",
    omop_dir: str = "terminology_omop",
    out_dir: str = "results/terminology",
    force: bool = False,
    max_ancestor_distance: Optional[int] = 3,
) -> dict:
    """
    Build and save a trimmed terminology layer.
    Returns paths + stats (dict).
    """
    _ensure_dir(out_dir)

    pkl_path = os.path.join(out_dir, "terminology.pkl")
    summary_path = os.path.join(out_dir, "terminology_summary.json")

    if (not force) and os.path.exists(pkl_path) and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return {"out_dir": out_dir, "pkl_path": pkl_path, "summary": summary}

    # 1) Load patients (cache ok)
    patients = FHIRParser.load_directory(data_dir, use_cache=True)
    if not patients:
        raise RuntimeError("No patients found; cannot build terminology layer.")

    codes_by_system = _extract_codes_from_patients(patients)

    n_snomed = len(codes_by_system[FHIR_SNOMED_SYSTEM])
    n_loinc = len(codes_by_system[FHIR_LOINC_SYSTEM])
    n_rxnorm = len(codes_by_system[FHIR_RXNORM_SYSTEM])

    # 2) Load CONCEPT and filter
    concept_path = os.path.join(omop_dir, "CONCEPT.csv")
    if not os.path.exists(concept_path):
        raise FileNotFoundError(f"CONCEPT.csv not found in {omop_dir}")

    needed_cols = [
        "concept_id", "concept_name", "domain_id", "vocabulary_id",
        "concept_class_id", "standard_concept", "concept_code",
        "valid_start_date", "valid_end_date", "invalid_reason"
    ]
    concept_df = _load_concept_csv(concept_path, usecols=needed_cols)

    concept_df["vocabulary_id"] = concept_df["vocabulary_id"].astype(str)
    concept_df["concept_code"] = concept_df["concept_code"].astype(str)

    # Build code sets
    snomed_codes = codes_by_system[FHIR_SNOMED_SYSTEM]
    loinc_codes = codes_by_system[FHIR_LOINC_SYSTEM]
    rxnorm_codes = codes_by_system[FHIR_RXNORM_SYSTEM]

    # Filter
    df_snomed = concept_df[(concept_df["vocabulary_id"] == OMOP_SNOMED) & (concept_df["concept_code"].isin(snomed_codes))]
    df_loinc = concept_df[(concept_df["vocabulary_id"] == OMOP_LOINC) & (concept_df["concept_code"].isin(loinc_codes))]
    df_rxnorm = concept_df[(concept_df["vocabulary_id"] == OMOP_RXNORM) & (concept_df["concept_code"].isin(rxnorm_codes))]

    # 3) Resolve duplicates
    def resolve_per_code(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        rows = []
        for code, g in df.groupby("concept_code"):
            best = _select_best_concept_row(g)
            rows.append(best)
        return pd.DataFrame(rows)

    df_snomed_best = resolve_per_code(df_snomed)
    df_loinc_best = resolve_per_code(df_loinc)
    df_rxnorm_best = resolve_per_code(df_rxnorm)

    concept_min = pd.concat([df_snomed_best, df_loinc_best, df_rxnorm_best], ignore_index=True)

    # Build mapping
    concept_id_by_key: Dict[Tuple[str, str], int] = {}

    def add_map(df: pd.DataFrame, fhir_system: str):
        for _, r in df.iterrows():
            code = str(r["concept_code"])
            cid = int(r["concept_id"])
            concept_id_by_key[(fhir_system, code)] = cid

    add_map(df_snomed_best, FHIR_SNOMED_SYSTEM)
    add_map(df_loinc_best, FHIR_LOINC_SYSTEM)
    add_map(df_rxnorm_best, FHIR_RXNORM_SYSTEM)

    # 4) Dicts
    domain_by_concept_id = {int(r["concept_id"]): str(r["domain_id"]) for _, r in concept_min.iterrows()}
    name_by_concept_id = {int(r["concept_id"]): str(r["concept_name"]) for _, r in concept_min.iterrows()}
    vocab_by_concept_id = {int(r["concept_id"]): str(r["vocabulary_id"]) for _, r in concept_min.iterrows()}
    code_by_concept_id = {int(r["concept_id"]): str(r["concept_code"]) for _, r in concept_min.iterrows()}

    # 5) Ancestors (Roll-up Logic)
    ancestors_by_descendant: Dict[int, List[Tuple[int, int]]] = {}
    ancestor_min_path = os.path.join(out_dir, "ancestor_min.csv")

    if not df_snomed_best.empty:
        descendant_ids = set(int(x) for x in df_snomed_best["concept_id"].tolist())
        anc_path = os.path.join(omop_dir, "CONCEPT_ANCESTOR.csv")
        
        # --- LOGIQUE HYBRIDE ---
        if os.path.exists(anc_path):
            # Cas A : Le fichier précalculé existe (OMOP Full)
            logger.info("Using existing CONCEPT_ANCESTOR.csv")
            anc_cols = ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"]
            sep = _detect_sep(anc_path)
            anc_df = pd.read_csv(anc_path, usecols=anc_cols, sep=sep, engine="python")
            anc_df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in anc_df.columns]
            anc_df = anc_df[anc_df["descendant_concept_id"].isin(descendant_ids)]
            if max_ancestor_distance is not None:
                anc_df = anc_df[anc_df["min_levels_of_separation"] <= int(max_ancestor_distance)]
        else:
            # Cas B : On doit le calculer via networkx et les relations parent/enfant directes
            logger.info("CONCEPT_ANCESTOR.csv not found. Calculating closure from relationships...")
            anc_df = _build_transitive_closure_from_relations(descendant_ids, omop_dir, max_ancestor_distance or 5)
        # -----------------------

        # Build dict
        for desc_id, g in anc_df.groupby("descendant_concept_id"):
            pairs = list(zip(g["ancestor_concept_id"].astype(int).tolist(),
                             g["min_levels_of_separation"].astype(int).tolist()))
            ancestors_by_descendant[int(desc_id)] = pairs

        anc_df.to_csv(ancestor_min_path, index=False)
    else:
        pd.DataFrame(columns=["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"]).to_csv(ancestor_min_path, index=False)

    # 6) Save outputs
    concept_min_path = os.path.join(out_dir, "concept_min.csv")
    concept_map_path = os.path.join(out_dir, "concept_map.csv")
    concept_min.to_csv(concept_min_path, index=False)

    map_rows = []
    for (sys, code), cid in concept_id_by_key.items():
        map_rows.append({"system": sys, "code": code, "concept_id": cid})
    pd.DataFrame(map_rows).to_csv(concept_map_path, index=False)

    layer = TerminologyLayer(
        concept_id_by_key=concept_id_by_key,
        domain_by_concept_id=domain_by_concept_id,
        name_by_concept_id=name_by_concept_id,
        vocab_by_concept_id=vocab_by_concept_id,
        code_by_concept_id=code_by_concept_id,
        ancestors_by_descendant=ancestors_by_descendant,
    )

    with open(pkl_path, "wb") as f:
        pickle.dump(layer, f)

    summary = {
        "data_dir": data_dir,
        "omop_dir": omop_dir,
        "n_patients": len(patients),
        "dataset_codes": {"SNOMED": n_snomed, "LOINC": n_loinc, "RxNorm": n_rxnorm},
        "mapped_concepts": {
            "SNOMED": int(len(df_snomed_best)),
            "total": int(len(concept_min)),
        },
        "max_ancestor_distance": max_ancestor_distance,
        "outputs": {
            "pkl": pkl_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"out_dir": out_dir, "pkl_path": pkl_path, "summary": summary}