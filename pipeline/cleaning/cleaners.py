"""
Data cleaning functions for each raw dataset.

Design philosophy
─────────────────
• Every cleaner receives a raw DataFrame and returns a cleaned DataFrame
  plus an `issues` dict that is logged and included in data_quality_report.json.
• Cleaning is deterministic so running the pipeline twice yields the same output
  (idempotency requirement from the assignment).
• We never drop valid rows silently – every removal is counted and justified.
"""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

from pipeline.utils.logging_utils import log_info, log_warn


# ── Date normalization ────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%Y-%m-%d",    # ISO-8601 (most common)
    "%m/%d/%Y",    # US format  e.g. 07/15/1945
    "%d-%m-%Y",    # EU format  e.g. 29-05-2023
    "%d/%m/%Y",    # EU slash
]


def _parse_date_series(series: pd.Series) -> pd.Series:
    """
    Try parsing a string date series through a priority list of formats.
    Falls back to pd.NaT for unparseable values.
    We avoid pd.to_datetime with infer_datetime_format because it silently
    mis-parses ambiguous values like 01/02/2023 (Jan 2 vs Feb 1).
    """
    result = pd.Series([pd.NaT] * len(series), dtype="datetime64[ns]")
    remaining_mask = series.notna() & series.astype(str).str.strip().ne("")

    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(series.where(remaining_mask), format=fmt, errors="coerce")
        newly_parsed = parsed.notna() & remaining_mask
        result = result.where(~newly_parsed, parsed)
        remaining_mask = remaining_mask & ~newly_parsed

    return result


def _normalize_sex(val: str) -> str:
    """Map varied gender strings to M / F / Unknown."""
    if not isinstance(val, str):
        return "Unknown"
    v = val.strip().lower()
    if v in ("m", "male"):
        return "M"
    if v in ("f", "female"):
        return "F"
    return "Unknown"


# ── Patient cleaners ──────────────────────────────────────────────────────────

def clean_site_alpha(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    issues: dict = {}

    df = df.copy()

    # 1. Strip whitespace from all string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # 2. Deduplicate on patient_id (keep first occurrence)
    dupes = df.duplicated(subset=["patient_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["patient_id"], keep="first")
    issues["duplicates_removed"] = int(dupes)

    # 3. Parse dates
    df["date_of_birth"] = _parse_date_series(df["date_of_birth"])
    df["admission_dt"] = _parse_date_series(df["admission_dt"])
    df["discharge_dt"] = _parse_date_series(df["discharge_dt"])

    # 4. Validate: discharge must not precede admission
    bad_dates = (df["discharge_dt"] < df["admission_dt"]).sum()
    df.loc[df["discharge_dt"] < df["admission_dt"], "discharge_dt"] = pd.NaT
    issues["discharge_before_admission_nulled"] = int(bad_dates)

    # 5. Normalize sex
    null_sex = df["sex"].isna().sum()
    df["sex"] = df["sex"].apply(_normalize_sex)
    issues["nulls_handled"] = int(null_sex)

    # 6. Add site_id for provenance
    df["site_id"] = "ALPHA"

    log_info(f"  site_alpha: {rows_in} → {len(df)} rows, issues={issues}")
    return df, issues


def clean_site_beta(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    issues: dict = {}

    df = df.copy()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    dupes = df.duplicated(subset=["patient_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["patient_id"], keep="first")
    issues["duplicates_removed"] = int(dupes)

    df["date_of_birth"] = _parse_date_series(df["date_of_birth"])
    df["admission_dt"] = _parse_date_series(df["admission_dt"])
    df["discharge_dt"] = _parse_date_series(df["discharge_dt"])

    bad_dates = (df["discharge_dt"].notna() & df["admission_dt"].notna() &
                 (df["discharge_dt"] < df["admission_dt"])).sum()
    df.loc[
        df["discharge_dt"].notna() & df["admission_dt"].notna() &
        (df["discharge_dt"] < df["admission_dt"]),
        "discharge_dt",
    ] = pd.NaT
    issues["discharge_before_admission_nulled"] = int(bad_dates)

    null_sex = df["sex"].isna().sum()
    df["sex"] = df["sex"].apply(_normalize_sex)
    issues["nulls_handled"] = int(null_sex)

    df["site_id"] = "BETA"
    log_info(f"  site_beta: {rows_in} → {len(df)} rows, issues={issues}")
    return df, issues


def clean_gamma_labs(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    issues: dict = {}

    df = df.copy()

    # Rename to a consistent schema used by downstream analytics
    df = df.rename(columns={"patient_ref": "patient_id"})

    # Deduplicate on lab_result_id
    dupes = df.duplicated(subset=["lab_result_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["lab_result_id"], keep="first")
    issues["duplicates_removed"] = int(dupes)

    # Parse collection_date
    df["collection_date"] = _parse_date_series(df["collection_date"].astype(str))

    # Null out non-positive test values (physically impossible for most panels)
    negative_vals = (df["test_value"] <= 0).sum()
    df.loc[df["test_value"] <= 0, "test_value"] = float("nan")
    issues["negative_test_values_nulled"] = int(negative_vals)

    # Normalize test_name to lowercase for consistent look-ups
    df["test_name"] = df["test_name"].str.lower().str.strip()

    null_vals = df["test_value"].isna().sum()
    issues["nulls_handled"] = int(null_vals)

    log_info(f"  gamma_labs: {rows_in} → {len(df)} rows, issues={issues}")
    return df, issues


def clean_diagnoses(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    issues: dict = {}

    df = df.copy()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    dupes = df.duplicated(subset=["diagnosis_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["diagnosis_id"], keep="first")
    issues["duplicates_removed"] = int(dupes)

    # Parse diagnosis_date
    df["diagnosis_date"] = _parse_date_series(df["diagnosis_date"])

    # Standardize ICD-10 code to uppercase with dot (e.g., E119 → E11.9)
    def _normalize_icd(code: str) -> str:
        if not isinstance(code, str):
            return code
        code = code.strip().upper().replace(" ", "")
        # Insert dot if missing and code is longer than 3 chars without one
        if "." not in code and len(code) > 3:
            code = code[:3] + "." + code[3:]
        return code

    df["icd10_code"] = df["icd10_code"].apply(_normalize_icd)

    null_count = df["patient_id"].isna().sum()
    issues["nulls_handled"] = int(null_count + df["icd10_code"].isna().sum())

    log_info(f"  diagnoses: {rows_in} → {len(df)} rows, issues={issues}")
    return df, issues


def clean_medications(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    issues: dict = {}

    df = df.copy()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    dupes = df.duplicated(subset=["medication_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["medication_id"], keep="first")
    issues["duplicates_removed"] = int(dupes)

    df["start_date"] = _parse_date_series(df["start_date"])
    df["end_date"] = _parse_date_series(df["end_date"])

    # Flag end_date before start_date as a data entry error and null end_date
    bad_dates = (
        df["end_date"].notna() & df["start_date"].notna() &
        (df["end_date"] < df["start_date"])
    ).sum()
    df.loc[
        df["end_date"].notna() & df["start_date"].notna() &
        (df["end_date"] < df["start_date"]),
        "end_date",
    ] = pd.NaT
    issues["end_before_start_nulled"] = int(bad_dates)

    null_count = df[["medication_name", "patient_id"]].isna().any(axis=1).sum()
    issues["nulls_handled"] = int(null_count)

    log_info(f"  medications: {rows_in} → {len(df)} rows, issues={issues}")
    return df, issues


def clean_clinical_notes(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    rows_in = len(df)
    issues: dict = {}

    df = df.copy()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    dupes = df.duplicated(subset=["note_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["note_id"], keep="first")
    issues["duplicates_removed"] = int(dupes)

    df["note_date"] = _parse_date_series(df["note_date"])

    null_count = df[["patient_id", "note_category"]].isna().any(axis=1).sum()
    issues["nulls_handled"] = int(null_count)

    log_info(f"  clinical_notes: {rows_in} → {len(df)} rows, issues={issues}")
    return df, issues


def clean_genomics(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean and filter genomics variants.

    Filtering criteria for 'reliable' variant calls (documented decisions):
    ─────────────────────────────────────────────────────────────────────────
    1. read_depth >= 20 : Low coverage (<20x) means the variant call is
       statistically unreliable.  Industry standard minimum for somatic
       variant calling is 20–30x.
    2. allele_frequency > 0.0 : AF of exactly 0 is a placeholder / failed
       call; there is no biological variant at AF=0.
    3. allele_frequency <= 1.0 : AF > 1 is a data error (impossible).
    4. clinical_significance is not NaN/empty : Variants without a ClinVar
       significance assignment cannot contribute to clinical decision-making
       and are excluded from the analysis layer (retained in raw zone).
    """
    rows_in = len(df)
    issues: dict = {}

    df = df.copy()

    # Rename for consistency
    df = df.rename(columns={"patient_ref": "patient_id"})

    dupes = df.duplicated(subset=["variant_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["variant_id"], keep="first")
    issues["duplicates_removed"] = int(dupes)

    # Coerce numeric columns
    df["read_depth"] = pd.to_numeric(df["read_depth"], errors="coerce")
    df["allele_frequency"] = pd.to_numeric(df["allele_frequency"], errors="coerce")

    df["sample_date"] = _parse_date_series(df["sample_date"].astype(str))

    # Apply reliability filters
    before_filter = len(df)
    low_depth = (df["read_depth"] < 20).sum()
    bad_af = ((df["allele_frequency"] <= 0) | (df["allele_frequency"] > 1)).sum()
    no_sig = df["clinical_significance"].isna() | df["clinical_significance"].str.strip().eq("")

    reliable_mask = (
        (df["read_depth"] >= 20) &
        (df["allele_frequency"] > 0) &
        (df["allele_frequency"] <= 1) &
        (~no_sig)
    )
    df_reliable = df[reliable_mask].copy()

    issues["low_read_depth_filtered"] = int(low_depth)
    issues["invalid_allele_frequency_filtered"] = int(bad_af)
    issues["missing_clinical_significance_filtered"] = int(no_sig.sum())
    issues["unreliable_calls_filtered"] = int(before_filter - len(df_reliable))
    issues["reliable_calls_retained"] = int(len(df_reliable))
    issues["nulls_handled"] = int(df["read_depth"].isna().sum() + df["allele_frequency"].isna().sum())

    log_info(f"  genomics: {rows_in} raw → {len(df_reliable)} reliable calls, issues={issues}")
    return df_reliable, issues
