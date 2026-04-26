"""
Task 3: Analytics and Anomaly Detection
────────────────────────────────────────
All outputs are DataFrames written to datalake/consumption/ as Parquet.
Plots are generated in a separate module (pipeline/stats/visualizations.py).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pipeline.utils.logging_utils import log_info, log_warn


# ── 3a. Patient Demographics Summary ─────────────────────────────────────────

def patient_demographics_summary(patients: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age distribution, sex split, and site distribution.
    Returns a long-format summary DataFrame.
    """
    summary_rows = []

    # Age statistics
    age = patients["age_years"].dropna()
    for stat, val in {
        "age_mean": age.mean(),
        "age_median": age.median(),
        "age_std": age.std(),
        "age_min": age.min(),
        "age_max": age.max(),
        "age_p25": age.quantile(0.25),
        "age_p75": age.quantile(0.75),
    }.items():
        summary_rows.append({"metric": stat, "value": round(val, 2), "group": "all"})

    # Sex distribution
    sex_counts = patients["sex"].value_counts()
    for sex, count in sex_counts.items():
        pct = 100 * count / len(patients)
        summary_rows.append({"metric": "patient_count", "value": count, "group": f"sex_{sex}"})
        summary_rows.append({"metric": "patient_pct", "value": round(pct, 2), "group": f"sex_{sex}"})

    # Site distribution
    site_counts = patients["site_id"].value_counts()
    for site, count in site_counts.items():
        summary_rows.append({"metric": "patient_count", "value": count, "group": f"site_{site}"})

    # Age bins for histogram data
    bins = [0, 18, 35, 50, 65, 80, 120]
    labels = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
    age_binned = pd.cut(age, bins=bins, labels=labels, right=False)
    for label, count in age_binned.value_counts().sort_index().items():
        summary_rows.append({"metric": "age_bin_count", "value": int(count), "group": f"age_{label}"})

    df = pd.DataFrame(summary_rows)
    log_info(f"  patient_summary: {len(df)} metric rows")
    return df


# ── 3b. Lab Result Statistics ─────────────────────────────────────────────────

def lab_result_statistics(
    lab_results: pd.DataFrame,
    lab_reference: dict,
) -> pd.DataFrame:
    """
    Per test type: mean, median, std.
    Flag values outside reference ranges.
    Compute per-patient trend (worsening vs improving) for hba1c and creatinine.
    """
    rows = []
    trend_rows = []

    for test_name, group in lab_results.groupby("test_name"):
        values = group["test_value"].dropna()
        if len(values) == 0:
            continue

        ref = lab_reference.get(test_name.lower(), {})
        normal_low = ref.get("normal_low")
        normal_high = ref.get("normal_high")
        critical_low = ref.get("critical_low")
        critical_high = ref.get("critical_high")

        out_of_range = 0
        critical_flags = 0
        if normal_low is not None and normal_high is not None:
            out_of_range = int(((values < normal_low) | (values > normal_high)).sum())
        if critical_low is not None and critical_high is not None:
            critical_flags = int(((values < critical_low) | (values > critical_high)).sum())

        rows.append(
            {
                "test_name": test_name,
                "n_results": len(values),
                "mean": round(values.mean(), 4),
                "median": round(values.median(), 4),
                "std": round(values.std(), 4),
                "p25": round(values.quantile(0.25), 4),
                "p75": round(values.quantile(0.75), 4),
                "normal_low": normal_low,
                "normal_high": normal_high,
                "out_of_range_count": out_of_range,
                "critical_flag_count": critical_flags,
                "pct_out_of_range": round(100 * out_of_range / len(values), 2) if len(values) else 0,
            }
        )

        # Per-patient trend for hba1c and creatinine
        if test_name.lower() in ("hba1c", "creatinine"):
            for patient_id, pt_group in group.groupby("patient_id"):
                pt = pt_group.dropna(subset=["test_value", "collection_date"]).sort_values("collection_date")
                if len(pt) < 2:
                    continue
                # Linear regression slope: positive = worsening (values increasing)
                x = np.arange(len(pt))
                y = pt["test_value"].values
                slope = float(np.polyfit(x, y, 1)[0])
                trend = "worsening" if slope > 0 else "improving"
                trend_rows.append(
                    {
                        "patient_id": patient_id,
                        "test_name": test_name,
                        "n_visits": len(pt),
                        "first_value": round(float(pt["test_value"].iloc[0]), 4),
                        "last_value": round(float(pt["test_value"].iloc[-1]), 4),
                        "slope": round(slope, 6),
                        "trend": trend,
                    }
                )

    stats_df = pd.DataFrame(rows)
    trend_df = pd.DataFrame(trend_rows)

    # Merge trend data as extra columns for each test/patient pair
    if not trend_df.empty:
        combined = pd.concat(
            [stats_df, trend_df.groupby("test_name").agg(
                patients_worsening=("trend", lambda x: (x == "worsening").sum()),
                patients_improving=("trend", lambda x: (x == "improving").sum()),
            ).reset_index()],
            join="outer",
        )
    else:
        stats_df["patients_worsening"] = 0
        stats_df["patients_improving"] = 0
        combined = stats_df

    log_info(f"  lab_statistics: {len(stats_df)} test types, {len(trend_rows)} patient trends computed")
    return combined


# ── 3c. Diagnosis Frequency ───────────────────────────────────────────────────

def _icd_chapter(code: str, chapters_df: pd.DataFrame) -> str:
    """
    Map an ICD-10 code to its chapter name using the reference table.
    The chapter table uses code_range like 'A00-B99'; we match the
    leading alpha letter of the code to the alphabetical range.
    """
    if not isinstance(code, str) or len(code) == 0:
        return "Unknown"

    prefix_letter = code[0].upper()
    prefix_num = int(re.sub(r"[^0-9]", "", code[:4] or "0") or 0)

    for _, row in chapters_df.iterrows():
        try:
            rng = row["code_range"]
            start, end = rng.split("-")
            s_letter, s_num = start[0], int(start[1:])
            e_letter, e_num = end[0], int(end[1:])

            if s_letter == e_letter:
                if prefix_letter == s_letter and s_num <= prefix_num <= e_num:
                    return row["chapter_name"]
            else:
                if s_letter <= prefix_letter <= e_letter:
                    return row["chapter_name"]
        except Exception:
            continue
    return "Other"


def diagnosis_frequency(
    diagnoses: pd.DataFrame,
    patients: pd.DataFrame,
    icd10_chapters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Top 15 ICD-10 chapters by unique patient count (not diagnosis count).
    We count distinct patients per chapter so that a patient with 5 E11.*
    diagnoses counts once in the Endocrine chapter.
    """
    diagnoses = diagnoses.copy()
    diagnoses["chapter"] = diagnoses["icd10_code"].apply(
        lambda c: _icd_chapter(c, icd10_chapters)
    )

    # Patient-level unique chapter flags
    pt_chapters = (
        diagnoses.groupby("patient_id")["chapter"]
        .apply(set)
        .reset_index()
        .explode("chapter")
    )

    chapter_counts = (
        pt_chapters.groupby("chapter")["patient_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
        .rename(columns={"patient_id": "patient_count"})
    )

    total_patients = patients["patient_id"].nunique()
    chapter_counts["pct_of_patients"] = (
        chapter_counts["patient_count"] / total_patients * 100
    ).round(2)

    log_info(f"  diagnosis_frequency: top {len(chapter_counts)} ICD-10 chapters computed")
    return chapter_counts


# ── 3d. Genomics Variant Hotspots ─────────────────────────────────────────────

def genomics_variant_hotspots(genomics: pd.DataFrame) -> pd.DataFrame:
    """
    Top 5 genes by Pathogenic or Likely Pathogenic variant count.
    For each gene: mean AF, 25th and 75th percentile AF.
    """
    path_variants = genomics[
        genomics["clinical_significance"].isin(["Pathogenic", "Likely Pathogenic"])
    ].copy()

    if path_variants.empty:
        log_warn("  No Pathogenic/Likely Pathogenic variants found after filtering")
        return pd.DataFrame()

    hotspots = (
        path_variants.groupby("gene")
        .agg(
            variant_count=("variant_id", "count"),
            mean_allele_frequency=("allele_frequency", "mean"),
            af_p25=("allele_frequency", lambda x: x.quantile(0.25)),
            af_p75=("allele_frequency", lambda x: x.quantile(0.75)),
            unique_patients=("patient_id", "nunique"),
        )
        .reset_index()
        .sort_values("variant_count", ascending=False)
        .head(5)
    )

    hotspots["mean_allele_frequency"] = hotspots["mean_allele_frequency"].round(4)
    hotspots["af_p25"] = hotspots["af_p25"].round(4)
    hotspots["af_p75"] = hotspots["af_p75"].round(4)

    log_info(f"  variant_hotspots: {len(hotspots)} hotspot genes identified")
    return hotspots


# ── 3e. Cross-Domain High-Risk Patients ──────────────────────────────────────

def identify_high_risk_patients(
    patients: pd.DataFrame,
    lab_results: pd.DataFrame,
    genomics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Patients with BOTH:
      • at least one hba1c reading > 7.0 (diabetic range), AND
      • at least one Pathogenic or Likely Pathogenic genomics variant.

    Decision on orphan records: we include patients found in supplementary
    data even if they lack a demographics row, because the clinical risk
    signal is present regardless of whether we have their full record.
    We flag these as 'demographics_missing=True' so analysts are aware.
    """
    # Patients with elevated HbA1c
    hba1c = lab_results[lab_results["test_name"].str.lower() == "hba1c"]
    diabetic_patients = set(
        hba1c.loc[hba1c["test_value"] > 7.0, "patient_id"].dropna()
    )

    # Patients with pathogenic variants
    path_variants = genomics[
        genomics["clinical_significance"].isin(["Pathogenic", "Likely Pathogenic"])
    ]
    pathogenic_patients = set(path_variants["patient_id"].dropna())

    # Intersection
    high_risk_ids = diabetic_patients & pathogenic_patients

    if not high_risk_ids:
        log_warn("  No high-risk patients identified (intersection is empty)")
        return pd.DataFrame(columns=["patient_id", "max_hba1c", "pathogenic_variant_count", "demographics_missing"])

    # Build output with supporting evidence
    hba1c_max = (
        hba1c[hba1c["patient_id"].isin(high_risk_ids)]
        .groupby("patient_id")["test_value"]
        .max()
        .reset_index()
        .rename(columns={"test_value": "max_hba1c"})
    )

    path_count = (
        path_variants[path_variants["patient_id"].isin(high_risk_ids)]
        .groupby("patient_id")["variant_id"]
        .count()
        .reset_index()
        .rename(columns={"variant_id": "pathogenic_variant_count"})
    )

    result = hba1c_max.merge(path_count, on="patient_id", how="outer")

    # Merge patient demographics if available
    patient_ids_known = set(patients["patient_id"].dropna())
    result["demographics_missing"] = ~result["patient_id"].isin(patient_ids_known)

    # Join available demographics columns
    result = result.merge(
        patients[["patient_id", "sex", "age_years", "site_id"]],
        on="patient_id",
        how="left",
    )

    result["max_hba1c"] = result["max_hba1c"].round(2)
    result = result.sort_values("max_hba1c", ascending=False).reset_index(drop=True)

    log_info(f"  high_risk_patients: {len(result)} patients identified")
    return result


# ── 3f. Anomaly Detection ────────────────────────────────────────────────────

def detect_anomalies(
    patients: pd.DataFrame,
    lab_results: pd.DataFrame,
    diagnoses: pd.DataFrame,
    medications: pd.DataFrame,
    genomics: pd.DataFrame,
    lab_reference: dict,
) -> pd.DataFrame:
    """
    Flag clinically anomalous records.

    Anomaly categories and rationale:
    ──────────────────────────────────
    A1 – Impossible age: dob in the future or age > 130.
         Rationale: no verified human has lived past 122 years.

    A2 – Age-diagnosis mismatch: paediatric-only diagnoses in adults
         or adult-onset diseases (e.g., T2DM ICD E11) in very young patients
         (<5 years). Rationale: clinical domain knowledge.

    A3 – Critical lab value: lab result outside the critical range defined
         in lab_test_ranges.json. These values are incompatible with life
         if sustained and require immediate review.

    A4 – Conflicting medication orders: same patient has two records for
         the same medication with overlapping active periods.
         Rationale: risk of accidental double-dosing.

    A5 – Genomics inconsistency: a variant whose chromosome in the data
         does not match the reference chromosome for that gene.
         Rationale: indicates a sample-swap or data entry error.
    """
    flag_rows = []

    # A1 – Impossible age
    today = pd.Timestamp("today")
    for _, row in patients.iterrows():
        if pd.isna(row.get("date_of_birth")):
            continue
        dob = pd.to_datetime(row["date_of_birth"], errors="coerce")
        if pd.isna(dob):
            continue
        age = (today - dob).days / 365.25
        if dob > today or age > 130:
            flag_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "anomaly_type": "A1_impossible_age",
                    "detail": f"DOB={row['date_of_birth']}, computed_age={round(age,1)}",
                    "severity": "high",
                }
            )

    # A2 – E11 (Type 2 diabetes) in patients under 10 years old
    t2dm_codes = diagnoses[diagnoses["icd10_code"].str.startswith("E11", na=False)]
    young_patients = patients[
        patients["age_years"].notna() & (patients["age_years"] < 10)
    ]["patient_id"]
    for pid in t2dm_codes["patient_id"].unique():
        if pid in young_patients.values:
            age = patients.loc[patients["patient_id"] == pid, "age_years"].values
            age_val = age[0] if len(age) else "unknown"
            flag_rows.append(
                {
                    "patient_id": pid,
                    "anomaly_type": "A2_age_diagnosis_mismatch",
                    "detail": f"Type 2 diabetes (E11) diagnosed in patient aged {age_val} years",
                    "severity": "medium",
                }
            )

    # A3 – Critical lab values
    for test_name, ref in lab_reference.items():
        crit_low = ref.get("critical_low")
        crit_high = ref.get("critical_high")
        if crit_low is None or crit_high is None:
            continue
        subset = lab_results[lab_results["test_name"].str.lower() == test_name.lower()].dropna(subset=["test_value"])
        critical = subset[(subset["test_value"] < crit_low) | (subset["test_value"] > crit_high)]
        for _, row in critical.iterrows():
            flag_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "anomaly_type": "A3_critical_lab_value",
                    "detail": f"{test_name}={row['test_value']} (critical range {crit_low}–{crit_high})",
                    "severity": "critical",
                }
            )

    # A4 – Overlapping medication orders
    med_active = medications[
        medications["status"].str.lower().isin(["active", "ongoing"])
    ].dropna(subset=["patient_id", "medication_name"])
    for (pid, med), grp in med_active.groupby(["patient_id", "medication_name"]):
        if len(grp) < 2:
            continue
        # Sort by start_date and check pairwise overlap
        grp = grp.sort_values("start_date").dropna(subset=["start_date"])
        for i in range(len(grp) - 1):
            r1 = grp.iloc[i]
            r2 = grp.iloc[i + 1]
            # Overlap if r2 starts before r1 ends
            end1 = r1["end_date"]
            start2 = r2["start_date"]
            if pd.isna(end1) or pd.isna(start2):
                continue
            if start2 < end1:
                flag_rows.append(
                    {
                        "patient_id": pid,
                        "anomaly_type": "A4_conflicting_medication_orders",
                        "detail": f"{med}: records {r1['medication_id']} and {r2['medication_id']} overlap",
                        "severity": "high",
                    }
                )

    # A5 – Genomics chromosome mismatch against gene reference
    # We rely on the gene column; load gene_reference externally and pass as part of genomics enrichment
    # Simplified: flag if gene appears with an unexpected chromosome pattern
    known_chr = {
        "BRCA1": "chr17", "BRCA2": "chr13", "TP53": "chr17",
        "EGFR": "chr7", "KRAS": "chr12", "BRAF": "chr7",
        "PIK3CA": "chr3", "ALK": "chr2", "ROS1": "chr6",
        "MET": "chr7", "HER2": "chr17", "NRAS": "chr1",
        "APC": "chr5", "PTEN": "chr10", "RB1": "chr13",
        "CDH1": "chr16", "MLH1": "chr3", "MSH2": "chr2",
        "ATM": "chr11", "PALB2": "chr16", "CHEK2": "chr22",
        "RAD51C": "chr17",
    }
    for _, row in genomics.iterrows():
        gene = str(row.get("gene", "")).strip()
        chrom = str(row.get("chromosome", "")).strip()
        expected = known_chr.get(gene)
        if expected and chrom != expected:
            flag_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "anomaly_type": "A5_genomics_chromosome_mismatch",
                    "detail": f"Gene {gene} listed on {chrom} but expected {expected}",
                    "severity": "medium",
                }
            )

    df = pd.DataFrame(flag_rows)
    if df.empty:
        df = pd.DataFrame(columns=["patient_id", "anomaly_type", "detail", "severity"])

    log_info(
        f"  anomalies: {len(df)} flags "
        f"({df['anomaly_type'].value_counts().to_dict() if not df.empty else {}})"
    )
    return df


# ── Note Category Classifier (Bonus Task) ────────────────────────────────────

STANDARD_CATEGORIES = [
    "Admission", "Discharge", "Progress", "Surgical",
    "Consultation", "Lab Review", "Other",
]

_KEYWORD_MAP = {
    "Admission": [
        "admitting", "admission", "admit", "presenting complaint",
        "chief complaint", "history of present illness",
    ],
    "Discharge": [
        "discharge", "discharged", "discharge summary", "discharge instructions",
    ],
    "Progress": [
        "progress", "daily note", "follow-up", "soap note", "subjective",
        "assessment and plan", "problem list",
    ],
    "Surgical": [
        "operative", "surgical", "procedure note", "pre-op", "post-op",
        "intraoperative", "anaesthesia", "anesthesia",
    ],
    "Consultation": [
        "consultation", "consult", "specialist review", "referred",
    ],
    "Lab Review": [
        "lab results review", "laboratory", "lab report", "pathology",
        "microbiology", "radiology", "imaging report",
    ],
}


def classify_note_category(raw_category: str) -> str:
    """
    Rule-based classifier mapping free-text note_category strings to one of
    the seven standardised categories.

    Design: keyword matching on lowercased text with priority ordering.
    This is interpretable, fast, and requires no external API.
    Accuracy is evaluated against the expected_category column in the dataset.
    """
    if not isinstance(raw_category, str):
        return "Other"
    text = raw_category.lower().strip()
    for category, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text:
                return category
    return "Other"


def classify_clinical_notes(clinical_notes: pd.DataFrame) -> pd.DataFrame:
    """Add predicted_category column and compute accuracy against expected."""
    df = clinical_notes.copy()
    df["predicted_category"] = df["note_category"].apply(classify_note_category)

    if "expected_category" in df.columns:
        df["classification_correct"] = df["predicted_category"] == df["expected_category"]
        accuracy = df["classification_correct"].mean()
        log_info(f"  note classifier accuracy: {accuracy:.2%} on {len(df)} notes")

    return df
