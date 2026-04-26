"""
Task 4: Analytical Visualizations
───────────────────────────────────
All plots read exclusively from pipeline output (consumption/ parquet files),
not from raw data.  Output path: datalake/consumption/plots/

Plot inventory:
  01_patient_demographics.png     – age histogram + sex bar chart
  02_diagnosis_frequency.png      – horizontal bar chart top-15 ICD-10 chapters
  03_lab_distributions.png        – distribution plots for 2+ test types with reference bands
  04_genomics_scatter.png         – allele frequency vs read depth, colored by significance
  05_high_risk_summary.png        – high-risk cohort summary visualization
  06_data_quality_overview.png    – pipeline data quality metrics
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Docker/headless environments

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline.utils.logging_utils import log_info


# ── Style constants ───────────────────────────────────────────────────────────

PALETTE = {
    "primary": "#1B6CA8",
    "secondary": "#2CA87F",
    "accent": "#E85D04",
    "neutral": "#6C757D",
    "light": "#E9ECEF",
    "M": "#4A90D9",
    "F": "#E87DA8",
    "Unknown": "#9B9B9B",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log_info(f"  saved plot: {out.name}")


# ── Plot 1: Patient Demographics ─────────────────────────────────────────────

def plot_patient_demographics(
    patients: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Age distribution histogram + sex split bar chart side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Patient Demographics Overview", fontsize=14, fontweight="bold", y=1.01)

    # ── Left: Age histogram
    ax1 = axes[0]
    age_data = patients["age_years"].dropna()
    ax1.hist(age_data, bins=20, color=PALETTE["primary"], edgecolor="white", linewidth=0.6, alpha=0.88)
    ax1.axvline(age_data.mean(), color=PALETTE["accent"], linestyle="--", linewidth=1.6, label=f"Mean: {age_data.mean():.1f} yrs")
    ax1.axvline(age_data.median(), color=PALETTE["secondary"], linestyle=":", linewidth=1.6, label=f"Median: {age_data.median():.1f} yrs")
    ax1.set_xlabel("Age (years)", labelpad=8)
    ax1.set_ylabel("Number of Patients", labelpad=8)
    ax1.set_title("Age Distribution", fontweight="semibold")
    ax1.legend(frameon=False, fontsize=9)

    # ── Right: Sex distribution bar chart
    ax2 = axes[1]
    sex_counts = patients["sex"].value_counts()
    colors = [PALETTE.get(s, PALETTE["neutral"]) for s in sex_counts.index]
    bars = ax2.bar(sex_counts.index, sex_counts.values, color=colors, edgecolor="white", linewidth=0.8)
    total = sex_counts.sum()
    for bar, (label, count) in zip(bars, sex_counts.items()):
        pct = 100 * count / total
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    ax2.set_xlabel("Sex", labelpad=8)
    ax2.set_ylabel("Number of Patients", labelpad=8)
    ax2.set_title("Sex Distribution", fontweight="semibold")
    ax2.set_ylim(0, sex_counts.max() * 1.18)

    fig.tight_layout()
    _save(fig, output_dir, "01_patient_demographics.png")


# ── Plot 2: Diagnosis Frequency ───────────────────────────────────────────────

def plot_diagnosis_frequency(
    diag_freq: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Horizontal bar chart of top-15 ICD-10 chapters by unique patient count."""
    df = diag_freq.sort_values("patient_count", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = sns.color_palette("Blues_d", n_colors=len(df))
    bars = ax.barh(df["chapter"], df["patient_count"], color=colors, edgecolor="white")

    for bar, pct in zip(bars, df["pct_of_patients"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}  ({pct:.1f}%)",
            va="center",
            fontsize=8.5,
        )

    ax.set_xlabel("Unique Patients", labelpad=8)
    ax.set_title("Top 15 ICD-10 Disease Chapters by Patient Count", fontweight="bold", fontsize=13, pad=12)
    ax.set_xlim(0, df["patient_count"].max() * 1.22)

    fig.tight_layout()
    _save(fig, output_dir, "02_diagnosis_frequency.png")


# ── Plot 3: Lab Result Distributions ─────────────────────────────────────────

def plot_lab_distributions(
    lab_results: pd.DataFrame,
    lab_reference: dict,
    output_dir: Path,
    test_names: list[str] | None = None,
) -> None:
    """
    KDE + histogram for at least 2 test types with reference range bands overlaid.
    """
    if test_names is None:
        # Pick the two tests with the most data
        counts = lab_results.groupby("test_name")["test_value"].count().sort_values(ascending=False)
        test_names = counts.head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for idx, test in enumerate(test_names[:4]):
        ax = axes[idx]
        subset = lab_results[lab_results["test_name"] == test]["test_value"].dropna()

        if len(subset) == 0:
            ax.set_visible(False)
            continue

        ax.hist(subset, bins=30, color=PALETTE["primary"], alpha=0.5, edgecolor="white", density=True, label="Distribution")
        try:
            subset.plot.kde(ax=ax, color=PALETTE["primary"], linewidth=2)
        except Exception:
            pass

        ref = lab_reference.get(test, {})
        lo = ref.get("normal_low")
        hi = ref.get("normal_high")
        if lo is not None and hi is not None:
            ax.axvspan(lo, hi, alpha=0.12, color=PALETTE["secondary"], label=f"Normal ({lo}–{hi})")
            ax.axvline(lo, color=PALETTE["secondary"], linestyle="--", linewidth=1.2)
            ax.axvline(hi, color=PALETTE["secondary"], linestyle="--", linewidth=1.2)

        unit = ref.get("unit", "")
        ax.set_title(f"{test.upper()} ({unit})", fontweight="semibold")
        ax.set_xlabel(f"Value ({unit})" if unit else "Value")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Lab Result Distributions with Reference Ranges", fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, "03_lab_distributions.png")


# ── Plot 4: Genomics Scatter ──────────────────────────────────────────────────

def plot_genomics_scatter(
    genomics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Allele frequency vs read depth, colored by clinical significance.
    Includes reliability threshold lines.
    """
    significance_order = [
        "Pathogenic",
        "Likely Pathogenic",
        "Uncertain Significance",
        "Likely Benign",
        "Benign",
    ]
    color_map = {
        "Pathogenic": "#C0392B",
        "Likely Pathogenic": "#E67E22",
        "Uncertain Significance": "#8E44AD",
        "Likely Benign": "#27AE60",
        "Benign": "#2980B9",
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    for sig in significance_order:
        sub = genomics[genomics["clinical_significance"] == sig]
        if sub.empty:
            continue
        ax.scatter(
            sub["allele_frequency"],
            sub["read_depth"],
            c=color_map.get(sig, PALETTE["neutral"]),
            label=sig,
            alpha=0.55,
            s=25,
            edgecolors="none",
        )

    # Reliability threshold
    ax.axhline(20, color=PALETTE["accent"], linestyle="--", linewidth=1.4, label="Min read depth (20x)")
    ax.axvline(0.0, color=PALETTE["neutral"], linestyle=":", linewidth=1.0)

    ax.set_xlabel("Allele Frequency", labelpad=8)
    ax.set_ylabel("Read Depth (x)", labelpad=8)
    ax.set_title("Genomic Variants: Allele Frequency vs Read Depth", fontweight="bold", fontsize=13)
    ax.legend(title="Clinical Significance", frameon=True, fontsize=8.5, title_fontsize=9)

    fig.tight_layout()
    _save(fig, output_dir, "04_genomics_scatter.png")


# ── Plot 5: High-Risk Patient Summary ────────────────────────────────────────

def plot_high_risk_summary(
    high_risk: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Stacked bar and scatter summarising high-risk cohort."""
    if high_risk.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No high-risk patients identified", ha="center", va="center", fontsize=13)
        ax.axis("off")
        _save(fig, output_dir, "05_high_risk_summary.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("High-Risk Patient Cohort Summary", fontweight="bold", fontsize=14, y=1.01)

    # ── Left: HbA1c distribution in high-risk group
    ax1 = axes[0]
    hba1c_data = high_risk["max_hba1c"].dropna()
    ax1.hist(hba1c_data, bins=max(5, len(hba1c_data) // 3), color=PALETTE["accent"], edgecolor="white", alpha=0.85)
    ax1.axvline(7.0, color="#333", linestyle="--", linewidth=1.5, label="Diabetic threshold (7.0%)")
    ax1.set_xlabel("Max HbA1c (%)")
    ax1.set_ylabel("Patient Count")
    ax1.set_title("Max HbA1c in High-Risk Cohort", fontweight="semibold")
    ax1.legend(frameon=False, fontsize=8.5)

    # ── Right: Pathogenic variant burden
    ax2 = axes[1]
    var_data = high_risk["pathogenic_variant_count"].dropna()
    ax2.hist(var_data, bins=max(5, int(var_data.max())), color=PALETTE["primary"], edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Pathogenic Variant Count")
    ax2.set_ylabel("Patient Count")
    ax2.set_title("Pathogenic Variant Burden in High-Risk Cohort", fontweight="semibold")

    # Annotation
    ax2.text(
        0.97, 0.94,
        f"n = {len(high_risk)} patients",
        transform=ax2.transAxes,
        ha="right",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["light"], edgecolor=PALETTE["neutral"]),
    )

    fig.tight_layout()
    _save(fig, output_dir, "05_high_risk_summary.png")


# ── Plot 6: Data Quality Overview ────────────────────────────────────────────

def plot_data_quality_overview(
    quality_report: dict,
    output_dir: Path,
) -> None:
    """
    Grouped bar chart showing quality metrics per source dataset:
    rows_in, rows_out, nulls_handled, duplicates_removed.
    """
    datasets = []
    rows_in = []
    rows_out = []
    dupes = []
    nulls = []

    for entry in quality_report.get("sources", []):
        datasets.append(entry["dataset"])
        rows_in.append(entry["rows_in"])
        rows_out.append(entry["rows_out"])
        issues = entry.get("issues", {})
        dupes.append(issues.get("duplicates_removed", 0))
        nulls.append(issues.get("nulls_handled", 0))

    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - 1.5 * width, rows_in, width, label="Rows In", color="#4A90D9", alpha=0.88)
    ax.bar(x - 0.5 * width, rows_out, width, label="Rows Out (cleaned)", color="#27AE60", alpha=0.88)
    ax.bar(x + 0.5 * width, dupes, width, label="Duplicates Removed", color="#E67E22", alpha=0.88)
    ax.bar(x + 1.5 * width, nulls, width, label="Nulls Handled", color="#8E44AD", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=9)
    ax.set_ylabel("Record Count")
    ax.set_title("Pipeline Data Quality Metrics by Dataset", fontweight="bold", fontsize=13)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    _save(fig, output_dir, "06_data_quality_overview.png")
