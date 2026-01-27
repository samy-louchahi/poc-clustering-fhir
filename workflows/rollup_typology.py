"""
Roll-up + typology + naming (ontology-based) for clustering outputs.

Goal:
- For a given assignments.csv (patient_id -> cluster_id), compute:
  * domain distribution per cluster (from OMOP domain_id)
  * top SNOMED ancestors per cluster (distinctiveness = cluster_rate / global_rate)
  * intensity levels (low/mid/high) based on mapped code counts (quantiles)
- Produce:
  labels/cluster_typology.csv
  labels/cluster_labels.json
  labels/assignments_labeled.csv

Works for:
- KMeans outputs (macro)
- HDBSCAN global outputs

Assumptions:
- terminology layer built at results/terminology/terminology.pkl
- OMOP CONCEPT.csv available at terminology_omop/CONCEPT.csv
- patient cache exists (FHIRParser.load_directory uses it)
"""

from __future__ import annotations

import os
import json
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import csv
from pandas.errors import ParserError

from fhir_clustering.fhir_parser import FHIRParser
from fhir_clustering.data_structures import CodeSystem

FHIR_SNOMED_SYSTEM = "http://snomed.info/sct"
FHIR_LOINC_SYSTEM = "http://loinc.org"
FHIR_RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_terminology_layer(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise RuntimeError(f"Missing terminology layer: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _build_patient_rollups(
    patients,
    layer,
    max_ancestor_distance: int = 3,
) -> Tuple[Dict[str, Counter], Dict[str, Counter], Dict[str, int]]:
    """
    Returns:
      - patient_domain_counts: patient_id -> Counter(domain_id)
      - patient_ancestor_counts: patient_id -> Counter(ancestor_concept_id)
      - patient_intensity: patient_id -> int (number of mapped codes)
    """
    patient_domain_counts: Dict[str, Counter] = {}
    patient_ancestor_counts: Dict[str, Counter] = {}
    patient_intensity: Dict[str, int] = {}

    for p in patients:
        dom = Counter()
        anc = Counter()
        mapped = 0

        for c in p.codes:
            if c.system == CodeSystem.SNOMED:
                sys = FHIR_SNOMED_SYSTEM
            elif c.system == CodeSystem.LOINC:
                sys = FHIR_LOINC_SYSTEM
            elif c.system == CodeSystem.RXNORM:
                sys = FHIR_RXNORM_SYSTEM
            else:
                continue

            key = (sys, str(c.code))
            cid = layer.concept_id_by_key.get(key)
            if cid is None:
                continue

            mapped += 1

            domain = layer.domain_by_concept_id.get(cid)
            if domain:
                dom[domain] += 1

            # SNOMED ancestors only
            if c.system == CodeSystem.SNOMED:
                pairs = layer.ancestors_by_descendant.get(cid, [])
                for anc_id, dist in pairs:
                    if dist <= max_ancestor_distance:
                        anc[int(anc_id)] += 1

        patient_domain_counts[p.patient_id] = dom
        patient_ancestor_counts[p.patient_id] = anc
        patient_intensity[p.patient_id] = mapped

    return patient_domain_counts, patient_ancestor_counts, patient_intensity


def _detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    candidates = [",", "\t", ";", "|"]
    return max(candidates, key=lambda c: head.count(c))


def _load_concept_names_for_ids(concept_csv_path: str, concept_ids: List[int]) -> Dict[int, str]:
    """
    Load concept_name for a list of OMOP concept_ids from CONCEPT.csv.
    Robust to separators (tab/semicolon/comma), BOM and malformed quotes.
    """
    if not concept_ids:
        return {}

    wanted = set(int(x) for x in concept_ids)
    usecols = ["concept_id", "concept_name"]

    sep = _detect_sep(concept_csv_path)

    def _iter_chunks(quoting_mode: int, on_bad_lines: str):
        return pd.read_csv(
            concept_csv_path,
            usecols=usecols,
            chunksize=250_000,
            sep=sep,
            engine="python",
            quoting=quoting_mode,
            escapechar="\\",
            on_bad_lines=on_bad_lines,
            dtype=str,
        )

    def _consume(chunks) -> Dict[int, str]:
        name_map: Dict[int, str] = {}
        for chunk in chunks:
            chunk.columns = [str(c).replace("\ufeff", "").strip().lower() for c in chunk.columns]
            if "concept_id" not in chunk.columns or "concept_name" not in chunk.columns:
                continue
            # some rows may be NaN if skipped/bad
            chunk = chunk.dropna(subset=["concept_id", "concept_name"])
            # cast
            chunk["concept_id"] = chunk["concept_id"].astype(int, errors="ignore")
            chunk = chunk[pd.to_numeric(chunk["concept_id"], errors="coerce").notna()]
            chunk["concept_id"] = chunk["concept_id"].astype(int)

            sub = chunk[chunk["concept_id"].isin(wanted)]
            for _, r in sub.iterrows():
                name_map[int(r["concept_id"])] = str(r["concept_name"])
            if len(name_map) == len(wanted):
                break
        return name_map

    # 1) normal parsing (quotes handled)
    try:
        return _consume(_iter_chunks(csv.QUOTE_MINIMAL, "error"))
    except (ParserError, csv.Error):
        pass

    # 2) ignore quotes (warn/skip malformed)
    try:
        return _consume(_iter_chunks(csv.QUOTE_NONE, "warn"))
    except (ParserError, csv.Error):
        pass

    # 3) last resort: ignore quotes + skip bad lines
    return _consume(_iter_chunks(csv.QUOTE_NONE, "skip"))


def _quantile_levels(values_by_id: Dict[str, float]) -> Dict[str, str]:
    """
    Map each id to low/mid/high based on 33/66% quantiles.
    """
    vals = np.array(list(values_by_id.values()), dtype=float)
    if len(vals) == 0:
        return {}

    q33 = float(np.quantile(vals, 0.33))
    q66 = float(np.quantile(vals, 0.66))

    out = {}
    for k, v in values_by_id.items():
        if v <= q33:
            out[k] = "low"
        elif v <= q66:
            out[k] = "mid"
        else:
            out[k] = "high"
    return out


def _dominant_domain(domain_counts: Counter) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    """
    Returns dominant domain and top domain distribution as (domain, proportion).
    """
    total = sum(domain_counts.values())
    if total <= 0:
        return None, []
    items = [(d, c / total) for d, c in domain_counts.most_common()]
    dom = items[0][0] if items else None
    return dom, items[:5]


def _ancestor_distinctiveness(
    cluster_anc_counts: Counter,
    global_anc_counts: Counter,
    cluster_size: int,
    total_patients: int,
    top_n: int = 5,
) -> List[Tuple[int, float]]:
    """
    Distinctiveness for ancestors:
      score = (cluster_rate) / (global_rate)
    where cluster_rate = (#patients in cluster with this ancestor) / cluster_size
    and global_rate = (#patients overall with this ancestor) / total_patients
    We use "presence" per patient rather than raw occurrence counts to avoid intensity bias.
    """
    if cluster_size <= 0 or total_patients <= 0:
        return []

    scores = {}
    for anc_id, cluster_presence in cluster_anc_counts.items():
        global_presence = global_anc_counts.get(anc_id, 0)
        if global_presence <= 0:
            continue
        cr = cluster_presence / cluster_size
        gr = global_presence / total_patients
        if gr > 0:
            scores[anc_id] = cr / gr

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(int(a), float(s)) for a, s in top]


def build_typology_and_labels(
    assignments_csv: str,
    out_dir: str,
    data_dir: str = "data",
    terminology_pkl: str = "results/terminology/terminology.pkl",
    omop_concept_csv: str = "terminology_omop/CONCEPT.csv",
    max_ancestor_distance: int = 3,
    top_n_ancestors: int = 5,
) -> dict:
    """
    Main entry:
    - reads assignments_csv containing patient_id and a cluster column
    - writes labels/ artifacts in out_dir
    Returns: dict of output paths
    """
    _ensure_dir(out_dir)
    labels_dir = os.path.join(out_dir, "labels")
    _ensure_dir(labels_dir)

    df = pd.read_csv(assignments_csv)

    # detect cluster column name
    cluster_col = None
    for cand in ["kmeans_cluster", "hdbscan_cluster", "cluster_id"]:
        if cand in df.columns:
            cluster_col = cand
            break
    if cluster_col is None:
        raise RuntimeError(f"No cluster column found in {assignments_csv}")

    df["patient_id"] = df["patient_id"].astype(str)
    df[cluster_col] = df[cluster_col].astype(int)

    # Load patients + terminology
    patients = FHIRParser.load_directory(data_dir, use_cache=True)
    if not patients:
        raise RuntimeError("No patients found (needed for roll-up).")
    layer = _load_terminology_layer(terminology_pkl)

    patient_domain, patient_anc, patient_int = _build_patient_rollups(
        patients=patients,
        layer=layer,
        max_ancestor_distance=max_ancestor_distance,
    )

    # global ancestor presence: count patients where ancestor appears at least once
    global_anc_presence = Counter()
    for pid, anc_counter in patient_anc.items():
        for anc_id in anc_counter.keys():
            global_anc_presence[int(anc_id)] += 1

    total_patients = len(patient_anc)

    # cluster aggregation
    cluster_domain_counts = defaultdict(Counter)
    cluster_anc_presence = defaultdict(Counter)
    cluster_intensity_mean = {}

    for cid, grp in df.groupby(cluster_col):
        pids = grp["patient_id"].tolist()

        # domain aggregation (raw counts then proportions)
        dom = Counter()
        for pid in pids:
            dom += patient_domain.get(pid, Counter())
        cluster_domain_counts[int(cid)] = dom

        # ancestor presence per cluster (count patients)
        anc_presence = Counter()
        for pid in pids:
            anc_keys = patient_anc.get(pid, Counter()).keys()
            for a in anc_keys:
                anc_presence[int(a)] += 1
        cluster_anc_presence[int(cid)] = anc_presence

        # intensity mean
        ints = [patient_int.get(pid, 0) for pid in pids]
        cluster_intensity_mean[int(cid)] = float(np.mean(ints)) if ints else 0.0

    # intensity levels (cluster-level)
    intensity_level = _quantile_levels({str(cid): v for cid, v in cluster_intensity_mean.items()})

    # collect all ancestor ids we might name
    all_top_anc_ids = set()
    cluster_top_anc = {}
    for cid in cluster_anc_presence.keys():
        top = _ancestor_distinctiveness(
            cluster_anc_counts=cluster_anc_presence[cid],
            global_anc_counts=global_anc_presence,
            cluster_size=int((df[cluster_col] == cid).sum()),
            total_patients=total_patients,
            top_n=top_n_ancestors,
        )
        cluster_top_anc[cid] = top
        for a, _ in top:
            all_top_anc_ids.add(int(a))

    # load ancestor names
    anc_name = _load_concept_names_for_ids(omop_concept_csv, sorted(all_top_anc_ids))

    # build typology rows + labels
    rows = []
    labels_json = {}

    for cid in sorted(cluster_domain_counts.keys()):
        n_pat = int((df[cluster_col] == cid).sum())

        dom_name, dom_top = _dominant_domain(cluster_domain_counts[cid])
        dom_name = dom_name or "UnknownDomain"

        top_anc = cluster_top_anc.get(cid, [])
        if top_anc:
            best_anc_id, best_anc_score = top_anc[0]
            best_anc_name = anc_name.get(best_anc_id, f"SNOMED_ANC_{best_anc_id}")
        else:
            best_anc_id, best_anc_score = None, None
            best_anc_name = "NoAncestorSignal"

        inten = intensity_level.get(str(cid), "mid")

        label_short = f"{dom_name} | {best_anc_name} | {inten}"

        # long label: include top domains and top ancestors
        dom_str = ", ".join([f"{d}:{p:.2f}" for d, p in dom_top]) if dom_top else "NA"
        anc_str = ", ".join([f"{anc_name.get(a, a)}:{s:.2f}" for a, s in top_anc]) if top_anc else "NA"

        label_long = (
            f"{label_short} — n={n_pat} | "
            f"domains(top): {dom_str} | "
            f"ancestors(top distinctiveness): {anc_str} | "
            f"avg_mapped_codes={cluster_intensity_mean.get(cid, 0.0):.1f}"
        )

        rows.append({
            "cluster_id": int(cid),
            "n_patients": n_pat,
            "dominant_domain": dom_name,
            "intensity_level": inten,
            "avg_mapped_codes": float(cluster_intensity_mean.get(cid, 0.0)),
            "top_domains": dom_str,
            "top_snomed_ancestors": anc_str,
            "label_short": label_short,
            "label_long": label_long,
        })

        labels_json[str(int(cid))] = {
            "cluster_id": int(cid),
            "label_short": label_short,
            "label_long": label_long,
            "dominant_domain": dom_name,
            "intensity_level": inten,
        }

    df_typo = pd.DataFrame(rows)
    typology_csv = os.path.join(labels_dir, "cluster_typology.csv")
    df_typo.to_csv(typology_csv, index=False)

    labels_path = os.path.join(labels_dir, "cluster_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_json, f, indent=2)

    # labeled assignments
    df_labeled = df[["patient_id", cluster_col]].copy()
    df_labeled["label_short"] = df_labeled[cluster_col].astype(str).map(lambda x: labels_json.get(x, {}).get("label_short", "NA"))
    labeled_csv = os.path.join(labels_dir, "assignments_labeled.csv")
    df_labeled.to_csv(labeled_csv, index=False)

    return {
        "labels_dir": labels_dir,
        "typology_csv": typology_csv,
        "labels_json": labels_path,
        "assignments_labeled_csv": labeled_csv,
        "cluster_col": cluster_col,
    }