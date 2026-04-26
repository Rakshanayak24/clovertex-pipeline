"""
Raw data loaders for all source files.

Each loader returns a plain DataFrame without any cleaning applied so that
cleaning logic can be tracked and audited separately.
"""

import json
from pathlib import Path

import pandas as pd

from pipeline.utils.logging_utils import log_info


def load_site_alpha(data_dir: Path) -> pd.DataFrame:
    """Load site_alpha_patients.csv – CSV with US-style dates."""
    path = data_dir / "site_alpha_patients.csv"
    log_info(f"Loading {path.name}")
    df = pd.read_csv(path, dtype=str)
    df["_source_file"] = path.name
    return df


def load_site_beta(data_dir: Path) -> pd.DataFrame:
    """
    Load site_beta_patients.json – nested JSON where name, encounter,
    and contact are sub-objects.  Flatten them into a tabular structure.
    """
    path = data_dir / "site_beta_patients.json"
    log_info(f"Loading {path.name}")
    raw = json.loads(path.read_text())

    rows = []
    for rec in raw:
        rows.append(
            {
                "patient_id": rec.get("patientID"),
                "first_name": rec.get("name", {}).get("given"),
                "last_name": rec.get("name", {}).get("family"),
                "date_of_birth": rec.get("birthDate"),
                "sex": rec.get("gender"),
                "blood_group": rec.get("bloodType"),
                "admission_dt": rec.get("encounter", {}).get("admissionDate"),
                "discharge_dt": rec.get("encounter", {}).get("dischargeDate"),
                "contact_phone": rec.get("contact", {}).get("phone"),
                "contact_email": rec.get("contact", {}).get("email"),
                "site": rec.get("encounter", {}).get("facility"),
            }
        )
    df = pd.DataFrame(rows, dtype=object).astype(str)
    df["_source_file"] = path.name
    return df


def load_site_gamma_labs(data_dir: Path) -> pd.DataFrame:
    """Load site_gamma_lab_results.parquet."""
    path = data_dir / "site_gamma_lab_results.parquet"
    log_info(f"Loading {path.name}")
    df = pd.read_parquet(path)
    df["_source_file"] = path.name
    return df


def load_diagnoses(data_dir: Path) -> pd.DataFrame:
    """Load diagnoses_icd10.csv."""
    path = data_dir / "diagnoses_icd10.csv"
    log_info(f"Loading {path.name}")
    df = pd.read_csv(path, dtype=str)
    df["_source_file"] = path.name
    return df


def load_medications(data_dir: Path) -> pd.DataFrame:
    """Load medications_log.json – a flat list of medication records."""
    path = data_dir / "medications_log.json"
    log_info(f"Loading {path.name}")
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw).astype(str)
    df["_source_file"] = path.name
    return df


def load_clinical_notes(data_dir: Path) -> pd.DataFrame:
    """Load clinical_notes_metadata.csv."""
    path = data_dir / "clinical_notes_metadata.csv"
    log_info(f"Loading {path.name}")
    df = pd.read_csv(path, dtype=str)
    df["_source_file"] = path.name
    return df


def load_genomics(data_dir: Path) -> pd.DataFrame:
    """Load genomics_variants.parquet."""
    path = data_dir / "genomics_variants.parquet"
    log_info(f"Loading {path.name}")
    df = pd.read_parquet(path)
    df["_source_file"] = path.name
    return df


# ── Reference data ───────────────────────────────────────────────────────────

def load_lab_reference(data_dir: Path) -> dict:
    """Return the lab test reference ranges dict."""
    path = data_dir / "reference" / "lab_test_ranges.json"
    return json.loads(path.read_text())


def load_gene_reference(data_dir: Path) -> dict:
    """Return the oncology gene metadata dict."""
    path = data_dir / "reference" / "gene_reference.json"
    return json.loads(path.read_text())


def load_icd10_chapters(data_dir: Path) -> pd.DataFrame:
    """Return the ICD-10 chapter-to-code-range mapping."""
    path = data_dir / "reference" / "icd10_chapters.csv"
    return pd.read_csv(path)
