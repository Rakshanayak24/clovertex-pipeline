"""
Unification layer: merge patient records from all three hospital sites
into a single canonical patients table, then join supplementary datasets.

Schema decisions
────────────────
• patient_id is the primary key – each site uses a unique prefix
  (ALPHA-, BETA-, GAMMA-) so there is no cross-site collision risk.
• We standardise column names to snake_case throughout.
• Dates are stored as datetime64[ns] in the unified table.
• PII columns (contact_phone, contact_email) are retained because this
  is an internal clinical platform, not a public-facing analytics layer.
  In a production setting these would live in a separate access-controlled
  table and be excluded from the analytics zone.
"""

from __future__ import annotations

import pandas as pd

from pipeline.utils.logging_utils import log_info


PATIENT_SCHEMA = [
    "patient_id",
    "first_name",
    "last_name",
    "date_of_birth",
    "sex",
    "blood_group",
    "admission_dt",
    "discharge_dt",
    "contact_phone",
    "contact_email",
    "site",
    "site_id",
]


def compute_age(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.Series:
    """Compute age in years from date_of_birth relative to a reference date."""
    ref = reference_date or pd.Timestamp("today")
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    age = (ref - dob).dt.days / 365.25
    return age.round(1)


def unify_patients(
    alpha: pd.DataFrame,
    beta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate Alpha and Beta patient tables into a single unified patients
    table.  Gamma site does not provide a patient demographics file – only
    lab results with patient_ref IDs – so Gamma patients appear in the
    unified table only when they also have Alpha/Beta records.

    Decision: rather than fabricating empty Gamma patient rows (which would
    introduce large amounts of NaN data and mislead downstream analytics),
    we keep only the sites that provide full demographic records.  Gamma
    patient IDs are still joinable via the lab_results table.
    """
    # Ensure both frames have the same schema
    for df in (alpha, beta):
        for col in PATIENT_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA

    alpha_sub = alpha[PATIENT_SCHEMA].copy()
    beta_sub = beta[PATIENT_SCHEMA].copy()

    unified = pd.concat([alpha_sub, beta_sub], ignore_index=True)

    # Compute age
    unified["age_years"] = compute_age(unified)

    # Flag patients with no DOB (cannot compute age – flag rather than drop)
    unified["dob_missing"] = unified["date_of_birth"].isna()

    log_info(f"  unified patients: {len(unified)} rows ({len(alpha_sub)} Alpha + {len(beta_sub)} Beta)")
    return unified


def join_supplementary(
    patients: pd.DataFrame,
    lab_results: pd.DataFrame,
    diagnoses: pd.DataFrame,
    medications: pd.DataFrame,
    genomics: pd.DataFrame,
    clinical_notes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Return a dict of analytics-ready DataFrames.  We do NOT create one giant
    denormalised table because the relationships are one-to-many at every join
    (one patient → many diagnoses, many labs, etc.).  A single flat join would
    explode row counts and make aggregations incorrect without careful
    deduplication.  Instead we keep normalised tables and join only the
    patient_id dimension.
    """
    patient_ids = set(patients["patient_id"].dropna())

    # Identify orphan records (supplementary records whose patient_id does not
    # appear in the unified patient table).  We retain them rather than drop –
    # they may belong to Gamma-only patients – and flag them.
    def _flag_orphans(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.copy()
        df["orphan_record"] = ~df["patient_id"].isin(patient_ids)
        n_orphan = df["orphan_record"].sum()
        if n_orphan:
            log_info(f"  {name}: {n_orphan} orphan records (patient not in unified table) – retained and flagged")
        return df

    lab_results = _flag_orphans(lab_results, "lab_results")
    diagnoses = _flag_orphans(diagnoses, "diagnoses")
    medications = _flag_orphans(medications, "medications")
    genomics = _flag_orphans(genomics, "genomics")
    clinical_notes = _flag_orphans(clinical_notes, "clinical_notes")

    return {
        "patients": patients,
        "lab_results": lab_results,
        "diagnoses": diagnoses,
        "medications": medications,
        "genomics": genomics,
        "clinical_notes": clinical_notes,
    }
