"""
Clovertex Clinical Data Pipeline – main entry point.

Run order:
  1. Ingest all raw files
  2. Copy originals to datalake/raw/
  3. Clean and validate each dataset
  4. Unify patients; join supplementary data
  5. Write refined Parquet files + zone manifest
  6. Run analytics (Task 3) → write consumption Parquet files
  7. Generate visualizations (Task 4) → write PNGs
  8. Write consumption/ manifest
  9. Write data_quality_report.json
 10. Write plots_README.md

Pipeline is idempotent: re-running overwrites outputs deterministically.
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline.ingestion.loaders import (
    load_clinical_notes,
    load_diagnoses,
    load_gene_reference,
    load_genomics,
    load_icd10_chapters,
    load_lab_reference,
    load_medications,
    load_site_alpha,
    load_site_beta,
    load_site_gamma_labs,
)
from pipeline.cleaning.cleaners import (
    clean_clinical_notes,
    clean_diagnoses,
    clean_gamma_labs,
    clean_genomics,
    clean_medications,
    clean_site_alpha,
    clean_site_beta,
)
from pipeline.transformation.unify import join_supplementary, unify_patients
from pipeline.stats.analytics import (
    classify_clinical_notes,
    detect_anomalies,
    diagnosis_frequency,
    genomics_variant_hotspots,
    identify_high_risk_patients,
    lab_result_statistics,
    patient_demographics_summary,
)
from pipeline.stats.visualizations import (
    plot_data_quality_overview,
    plot_diagnosis_frequency,
    plot_genomics_scatter,
    plot_high_risk_summary,
    plot_lab_distributions,
    plot_patient_demographics,
)
from pipeline.utils.io_utils import write_manifest, write_parquet
from pipeline.utils.logging_utils import log_dataset_stats, log_error, log_info

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LAKE = ROOT / "datalake"
RAW = LAKE / "raw"
REFINED = LAKE / "refined"
CONSUMPTION = LAKE / "consumption"
PLOTS_DIR = CONSUMPTION / "plots"


def ensure_dirs() -> None:
    for d in (RAW, REFINED, CONSUMPTION, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def copy_raw_files() -> None:
    """Mirror original data files into datalake/raw/ (untouched copies)."""
    log_info("Copying raw files to datalake/raw/")
    for src in DATA_DIR.iterdir():
        if src.is_file():
            shutil.copy2(src, RAW / src.name)
    ref_dst = RAW / "reference"
    ref_dst.mkdir(exist_ok=True)
    for src in (DATA_DIR / "reference").iterdir():
        shutil.copy2(src, ref_dst / src.name)


def run_ingestion(data_dir: Path) -> dict:
    log_info("=" * 55)
    log_info("STAGE 1: Ingestion")
    log_info("=" * 55)
    return {
        "alpha_raw": load_site_alpha(data_dir),
        "beta_raw": load_site_beta(data_dir),
        "gamma_raw": load_site_gamma_labs(data_dir),
        "diagnoses_raw": load_diagnoses(data_dir),
        "medications_raw": load_medications(data_dir),
        "notes_raw": load_clinical_notes(data_dir),
        "genomics_raw": load_genomics(data_dir),
        "lab_ref": load_lab_reference(data_dir),
        "gene_ref": load_gene_reference(data_dir),
        "icd10_chapters": load_icd10_chapters(data_dir),
    }


def run_cleaning(raw: dict) -> tuple[dict, dict]:
    log_info("=" * 55)
    log_info("STAGE 2: Cleaning")
    log_info("=" * 55)

    quality_sources = []

    alpha, a_issues = clean_site_alpha(raw["alpha_raw"])
    log_dataset_stats("site_alpha_patients", len(raw["alpha_raw"]), len(alpha), a_issues)
    quality_sources.append({"dataset": "site_alpha_patients", "rows_in": len(raw["alpha_raw"]), "rows_out": len(alpha), "issues": a_issues})

    beta, b_issues = clean_site_beta(raw["beta_raw"])
    log_dataset_stats("site_beta_patients", len(raw["beta_raw"]), len(beta), b_issues)
    quality_sources.append({"dataset": "site_beta_patients", "rows_in": len(raw["beta_raw"]), "rows_out": len(beta), "issues": b_issues})

    gamma, g_issues = clean_gamma_labs(raw["gamma_raw"])
    log_dataset_stats("site_gamma_lab_results", len(raw["gamma_raw"]), len(gamma), g_issues)
    quality_sources.append({"dataset": "site_gamma_lab_results", "rows_in": len(raw["gamma_raw"]), "rows_out": len(gamma), "issues": g_issues})

    diagnoses, d_issues = clean_diagnoses(raw["diagnoses_raw"])
    log_dataset_stats("diagnoses_icd10", len(raw["diagnoses_raw"]), len(diagnoses), d_issues)
    quality_sources.append({"dataset": "diagnoses_icd10", "rows_in": len(raw["diagnoses_raw"]), "rows_out": len(diagnoses), "issues": d_issues})

    medications, m_issues = clean_medications(raw["medications_raw"])
    log_dataset_stats("medications_log", len(raw["medications_raw"]), len(medications), m_issues)
    quality_sources.append({"dataset": "medications_log", "rows_in": len(raw["medications_raw"]), "rows_out": len(medications), "issues": m_issues})

    notes, n_issues = clean_clinical_notes(raw["notes_raw"])
    log_dataset_stats("clinical_notes_metadata", len(raw["notes_raw"]), len(notes), n_issues)
    quality_sources.append({"dataset": "clinical_notes_metadata", "rows_in": len(raw["notes_raw"]), "rows_out": len(notes), "issues": n_issues})

    genomics, ge_issues = clean_genomics(raw["genomics_raw"])
    log_dataset_stats("genomics_variants", len(raw["genomics_raw"]), len(genomics), ge_issues)
    quality_sources.append({"dataset": "genomics_variants", "rows_in": len(raw["genomics_raw"]), "rows_out": len(genomics), "issues": ge_issues})

    cleaned = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "diagnoses": diagnoses,
        "medications": medications,
        "notes": notes,
        "genomics": genomics,
    }
    quality_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": quality_sources,
    }
    return cleaned, quality_report


def run_transformation(cleaned: dict) -> dict:
    log_info("=" * 55)
    log_info("STAGE 3: Transformation & Unification")
    log_info("=" * 55)

    patients = unify_patients(cleaned["alpha"], cleaned["beta"])
    tables = join_supplementary(
        patients=patients,
        lab_results=cleaned["gamma"],
        diagnoses=cleaned["diagnoses"],
        medications=cleaned["medications"],
        genomics=cleaned["genomics"],
        clinical_notes=cleaned["notes"],
    )
    return tables


def write_refined(tables: dict) -> None:
    log_info("=" * 55)
    log_info("STAGE 4: Writing refined zone")
    log_info("=" * 55)

    write_parquet(tables["patients"], REFINED / "patients.parquet")
    log_info("  wrote patients.parquet")

    write_parquet(tables["diagnoses"], REFINED / "diagnoses.parquet")
    log_info("  wrote diagnoses.parquet")

    write_parquet(tables["medications"], REFINED / "medications.parquet")
    log_info("  wrote medications.parquet")

    write_parquet(tables["clinical_notes"], REFINED / "clinical_notes.parquet")
    log_info("  wrote clinical_notes.parquet")

    write_parquet(tables["genomics"], REFINED / "genomics_reliable.parquet")
    log_info("  wrote genomics_reliable.parquet")

    # Partition lab_results by test_name for downstream query efficiency.
    # Rationale: lab analytics almost always filter by test type first
    # (e.g., "give me all HbA1c results").  Partitioning by test_name
    # means a query for hba1c reads only that partition instead of the
    # full ~2000-row file.  The tradeoff is a larger number of small files,
    # which is acceptable here because we have <20 unique test types.
    lab_results = tables["lab_results"]
    lab_results.to_parquet(
        REFINED / "lab_results",
        partition_cols=["test_name"],
        index=False,
    )
    log_info("  wrote lab_results/ (partitioned by test_name)")

    write_manifest(REFINED)
    log_info("  wrote refined/manifest.json")


def run_analytics(tables: dict, raw: dict) -> dict:
    log_info("=" * 55)
    log_info("STAGE 5: Analytics (Task 3)")
    log_info("=" * 55)

    lab_ref = raw["lab_ref"]
    icd10_chapters = raw["icd10_chapters"]

    patient_summary = patient_demographics_summary(tables["patients"])
    write_parquet(patient_summary, CONSUMPTION / "patient_summary.parquet")
    log_info("  wrote patient_summary.parquet")

    lab_stats = lab_result_statistics(tables["lab_results"], lab_ref)
    write_parquet(lab_stats, CONSUMPTION / "lab_statistics.parquet")
    log_info("  wrote lab_statistics.parquet")

    diag_freq = diagnosis_frequency(tables["diagnoses"], tables["patients"], icd10_chapters)
    write_parquet(diag_freq, CONSUMPTION / "diagnosis_frequency.parquet")
    log_info("  wrote diagnosis_frequency.parquet")

    hotspots = genomics_variant_hotspots(tables["genomics"])
    write_parquet(hotspots, CONSUMPTION / "variant_hotspots.parquet")
    log_info("  wrote variant_hotspots.parquet")

    high_risk = identify_high_risk_patients(tables["patients"], tables["lab_results"], tables["genomics"])
    write_parquet(high_risk, CONSUMPTION / "high_risk_patients.parquet")
    log_info("  wrote high_risk_patients.parquet")

    anomalies = detect_anomalies(
        patients=tables["patients"],
        lab_results=tables["lab_results"],
        diagnoses=tables["diagnoses"],
        medications=tables["medications"],
        genomics=tables["genomics"],
        lab_reference=lab_ref,
    )
    write_parquet(anomalies, CONSUMPTION / "anomaly_flags.parquet")
    log_info("  wrote anomaly_flags.parquet")

    # Bonus: note classification
    classified_notes = classify_clinical_notes(tables["clinical_notes"])
    write_parquet(classified_notes, CONSUMPTION / "classified_notes.parquet")
    log_info("  wrote classified_notes.parquet")

    return {
        "patient_summary": patient_summary,
        "lab_stats": lab_stats,
        "diag_freq": diag_freq,
        "hotspots": hotspots,
        "high_risk": high_risk,
        "anomalies": anomalies,
    }


def run_visualizations(analytics: dict, tables: dict, raw: dict, quality_report: dict) -> None:
    log_info("=" * 55)
    log_info("STAGE 6: Visualizations (Task 4)")
    log_info("=" * 55)

    plot_patient_demographics(tables["patients"], PLOTS_DIR)
    plot_diagnosis_frequency(analytics["diag_freq"], PLOTS_DIR)
    plot_lab_distributions(tables["lab_results"], raw["lab_ref"], PLOTS_DIR)
    plot_genomics_scatter(tables["genomics"], PLOTS_DIR)
    plot_high_risk_summary(analytics["high_risk"], PLOTS_DIR)
    plot_data_quality_overview(quality_report, PLOTS_DIR)

    # Write plots_README.md
    readme = """# Plot Descriptions

| Plot | File | Description |
|------|------|-------------|
| 1 | `01_patient_demographics.png` | Age distribution histogram with mean/median overlay, and sex split bar chart across the unified patient cohort (Alpha + Beta sites). |
| 2 | `02_diagnosis_frequency.png` | Horizontal bar chart showing the top 15 ICD-10 disease chapters ranked by unique patient count (not diagnosis count, to avoid inflating chapters with repeat visits). |
| 3 | `03_lab_distributions.png` | KDE + histogram distribution plots for the four most-recorded lab tests. Green bands indicate the normal reference range from `lab_test_ranges.json`. |
| 4 | `04_genomics_scatter.png` | Scatter plot of allele frequency (x) vs read depth (y) for all reliable variant calls, coloured by clinical significance. The dashed horizontal line marks the 20x read depth reliability threshold. |
| 5 | `05_high_risk_summary.png` | Two panels summarising the high-risk cohort (patients with diabetic-range HbA1c AND at least one pathogenic variant): HbA1c distribution and pathogenic variant burden distribution. |
| 6 | `06_data_quality_overview.png` | Grouped bar chart summarising pipeline quality metrics per source dataset: rows ingested, rows after cleaning, duplicates removed, and nulls handled. |
"""
    (PLOTS_DIR / "plots_README.md").write_text(readme)
    log_info("  wrote plots_README.md")


def write_quality_report(quality_report: dict) -> None:
    out = ROOT / "data_quality_report.json"
    out.write_text(json.dumps(quality_report, indent=2, default=str))
    log_info(f"  wrote data_quality_report.json")


def main() -> None:
    log_info("Clovertex Clinical Data Pipeline – starting")
    log_info(f"Root: {ROOT}")

    ensure_dirs()
    copy_raw_files()

    raw = run_ingestion(DATA_DIR)
    cleaned, quality_report = run_cleaning(raw)

    # Merge cleaned data back so analytics stages can see reference data
    raw["lab_ref"] = load_lab_reference(DATA_DIR)
    raw["gene_ref"] = load_gene_reference(DATA_DIR)
    raw["icd10_chapters"] = load_icd10_chapters(DATA_DIR)

    tables = run_transformation(cleaned)
    write_refined(tables)

    analytics = run_analytics(tables, raw)

    run_visualizations(analytics, tables, raw, quality_report)

    write_manifest(CONSUMPTION)
    log_info("  wrote consumption/manifest.json")

    write_quality_report(quality_report)

    log_info("=" * 55)
    log_info("Pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_error(f"Pipeline failed: {exc}")
        raise SystemExit(1) from exc
