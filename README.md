# Clovertex Clinical Data Pipeline

A production-grade data engineering pipeline that ingests multi-format clinical and genomics data from three hospital sites, cleans and unifies it into a data lake, computes clinical analytics, detects anomalies, and generates analytical visualizations — all containerized with Docker and tested via GitHub Actions CI.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/clovertex-pipeline.git
cd clovertex-pipeline

# 2. Run the full pipeline end-to-end
docker-compose up --build

# 3. Inspect outputs
ls datalake/refined/
ls datalake/consumption/plots/
```

Outputs persist on your host machine in `./datalake/` via a bind mount. Re-running is safe — the pipeline is idempotent.

---

## Pipeline Architecture

```
data/                          ← Raw input files (7 files, 3 sites, 4 formats)
│
└─► STAGE 1 · Ingestion        pipeline/ingestion/loaders.py
      ↓ load all formats (CSV / JSON / Parquet)
      ↓ flatten nested JSON (site_beta)
      ↓ add provenance column _source_file
      ↓
      ► datalake/raw/          ← untouched copies, SHA-256 in manifest.json
      ↓
└─► STAGE 2 · Cleaning         pipeline/cleaning/cleaners.py
      ↓ per-dataset: dedup, date normalisation, type coercion
      ↓ structured JSON stats logged to stdout for each dataset
      ↓ genomics reliability filtering (read_depth / AF / significance)
      ↓
└─► STAGE 3 · Transformation   pipeline/transformation/unify.py
      ↓ unify Alpha + Beta patients into single schema
      ↓ join supplementary tables (labs, diagnoses, meds, genomics)
      ↓ flag orphan records (patient_id not in patient table)
      ↓
      ► datalake/refined/      ← cleaned Parquet; labs partitioned by test_name
      ↓
└─► STAGE 4 · Analytics        pipeline/stats/analytics.py
      ↓ 3a patient demographics summary
      ↓ 3b lab statistics + HbA1c / creatinine trend per patient
      ↓ 3c top-15 ICD-10 chapters by unique patient count
      ↓ 3d genomics variant hotspots (top-5 pathogenic genes)
      ↓ 3e cross-domain high-risk patients
      ↓ 3f anomaly detection (5 anomaly types)
      ↓ bonus: rule-based clinical note classifier
      ↓
      ► datalake/consumption/  ← analytics-ready Parquet files
      ↓
└─► STAGE 5 · Visualizations   pipeline/stats/visualizations.py
      ↓ 6 PNG plots reading exclusively from consumption/ Parquet
      ►  datalake/consumption/plots/
```

Each stage emits human-readable logs to **stderr** and structured JSON stats to **stdout** (the required logging format). This separation means you can pipe stdout to a log aggregator without noise.

---

## Data Cleaning Decisions

### Site Alpha (CSV)
- **Date format**: US-style `MM/DD/YYYY`. Parsed via format-priority list to avoid `infer_datetime_format` ambiguity.
- **Duplicates**: 20 exact duplicate `patient_id` rows removed (keep first). These were likely re-submissions from the hospital's export system.
- **Sex normalization**: `"male"/"M"` → `M`, `"female"/"F"` → `F`, everything else → `Unknown`.

### Site Beta (JSON)
- **Flattening**: Nested sub-objects (`name`, `encounter`, `contact`) flattened to top-level columns. Field names remapped to the unified schema (e.g., `patientID` → `patient_id`, `birthDate` → `date_of_birth`).
- **Date format**: Mix of ISO-8601 and `DD-MM-YYYY`. The priority parser handles both.
- **10 duplicate patient_ids** removed.

### Site Gamma Lab Results (Parquet)
- **patient_ref** renamed to **patient_id** for schema consistency.
- **Negative test values** (73 records): nulled out. A glucose reading of `-5 mg/dL` is physically impossible and indicates a sensor error or data entry issue rather than a real patient value.
- **test_name** lowercased for consistent join keys downstream.

### Diagnoses (CSV)
- **ICD-10 normalization**: codes normalized to uppercase with dot separator (`E119` → `E11.9`). Some EHR exports omit the dot; downstream chapter mapping depends on the prefix letter.

### Medications (JSON)
- **23 duplicate medication_id** records removed.
- **end_date before start_date**: these are nulled. A medication that "ended before it started" is a data entry error; we preserve the start_date but cannot trust the end_date.

### Genomics (Parquet)
Reliability filtering is the most significant cleaning decision in the pipeline. We retain only variants meeting **all** of:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| `read_depth >= 20` | 20× | Industry-standard minimum for reliable somatic variant calling (GATK, ACMG guidelines) |
| `allele_frequency > 0` | > 0.0 | AF = 0.0 means no reads support the alt allele — the variant doesn't exist |
| `allele_frequency <= 1` | ≤ 1.0 | AF > 1.0 is a data error; physically impossible |
| `clinical_significance` not null | present | Variants without ClinVar classification cannot contribute to clinical risk stratification |

**262 of 1137 variants (23%)** were filtered as unreliable. These remain in `datalake/raw/` for audit purposes.

---

## Data Lake Design

```
datalake/
├── raw/                    ← Immutable copies of all source files + SHA-256 manifest
├── refined/                ← Cleaned, validated Parquet files
│   ├── patients.parquet    ← Unified Alpha + Beta (650 patients)
│   ├── diagnoses.parquet
│   ├── medications.parquet
│   ├── clinical_notes.parquet
│   ├── genomics_reliable.parquet
│   ├── lab_results/        ← Hive-style partitioned by test_name
│   │   ├── test_name=hba1c/
│   │   ├── test_name=creatinine/
│   │   └── ...             (16 partitions)
│   └── manifest.json
└── consumption/            ← Analytics-ready aggregates + visualizations
    ├── patient_summary.parquet
    ├── lab_statistics.parquet
    ├── diagnosis_frequency.parquet
    ├── variant_hotspots.parquet
    ├── high_risk_patients.parquet
    ├── anomaly_flags.parquet
    ├── classified_notes.parquet
    ├── manifest.json
    └── plots/
        ├── 01_patient_demographics.png
        ├── 02_diagnosis_frequency.png
        ├── 03_lab_distributions.png
        ├── 04_genomics_scatter.png
        ├── 05_high_risk_summary.png
        ├── 06_data_quality_overview.png
        └── plots_README.md
```

### Partitioning rationale

`lab_results` is partitioned by `test_name` (Hive-style directory partitioning via PyArrow).

**Why**: Lab analytics almost always begin with a test-type filter ("show me all HbA1c results for diabetic patients"). With 16 distinct test types and ~2000 records total, reading a single partition instead of the full table reduces I/O by ~94% for single-test queries. The tradeoff — more small files — is acceptable here because the total file count (16) is well below the threshold where small-file overhead becomes a problem.

An alternative partitioning by `collection_date` year/month was considered for time-series queries but rejected because most analytical questions in this dataset are test-type-first, not time-first.

### Manifest structure

Each zone's `manifest.json` records: filename, row count, column schema, ISO-8601 timestamp, and SHA-256 checksum. This enables downstream systems to detect file corruption or unexpected schema changes without running the full pipeline.

---

## Anomaly Detection Logic

We define five categories of clinically anomalous records:

| Code | Type | Detection Logic | Severity |
|------|------|-----------------|----------|
| A1 | Impossible age | `date_of_birth` in the future or computed age > 130 years | High |
| A2 | Age-diagnosis mismatch | Type 2 diabetes (ICD E11) in patients under age 10 | Medium |
| A3 | Critical lab value | Lab result outside the critical range from `lab_test_ranges.json` | Critical |
| A4 | Conflicting medication orders | Same drug with overlapping active periods for the same patient | High |
| A5 | Genomics chromosome mismatch | A gene's chromosome in the data differs from the known reference chromosome | Medium |

**Design reasoning**: we prioritize false-negative safety (flag more, miss less) because in a clinical context, an unreviewed true positive is more dangerous than a false positive that gets reviewed and dismissed. All flags include a `detail` field with enough context for a clinician or data steward to quickly evaluate whether the flag is a real issue.

---

## Genomics Filtering Criteria

See Data Cleaning Decisions → Genomics above. The filtering is documented inline in `pipeline/cleaning/cleaners.py` with citations to the GATK and ACMG standards.

---

## Assumptions

1. **Gamma site patients**: Site Gamma does not provide a demographics file. We flag Gamma patient IDs as orphan records rather than creating phantom patient rows with empty demographics. This preserves analytical integrity.

2. **Sex encoding**: The assignment does not specify a canonical encoding. We chose `M`/`F`/`Unknown` (single character) for compactness and SQL-friendliness. If the downstream system requires `male`/`female`, the mapping in `cleaners.py` is the single change point.

3. **Date ambiguity**: For dates that could be parsed as either `MM/DD/YYYY` or `DD/MM/YYYY` (e.g., `01/02/2023`), we apply `MM/DD/YYYY` first (US format, as Site Alpha is the primary source). Site Beta uses explicit ISO-8601 or `DD-MM-YYYY` format which is unambiguous.

4. **Orphan records in supplementary tables**: ~60% of lab results, diagnoses, and medications belong to Gamma patients not in the patient demographics table. We retain these records with an `orphan_record=True` flag rather than dropping them, because clinical data should not be silently discarded.

5. **High-risk patient definition**: "Diabetic range" is defined as HbA1c > 7.0% per ADA guidelines (2024). We use the maximum HbA1c per patient rather than mean, because a single confirmed diabetic-range reading is clinically significant regardless of other readings.

---

## Improvements for Production

1. **Schema registry**: Register Parquet schemas in a versioned schema registry (e.g., AWS Glue Data Catalog) so that breaking schema changes are detected before they reach the consumption layer.

2. **Incremental loading**: Replace full-reload with `UPSERT` semantics using Delta Lake or Apache Iceberg for the refined zone — critical when sites send daily incremental feeds.

3. **PII handling**: `contact_phone` and `contact_email` should be encrypted at rest (column-level encryption or tokenization) and placed in a separate access-controlled table. The analytics zone should never contain raw PII.

4. **Data lineage**: Integrate OpenLineage to track which source records contributed to each output row — required for FDA 21 CFR Part 11 audit trails in clinical settings.

5. **Alerting**: Route anomaly flags (especially `A3 Critical lab value`) to a real-time alerting system (PagerDuty / Slack) rather than only writing to Parquet.

6. **Test suite**: Add `pytest` unit tests for each cleaner function with synthetic edge-case data (empty inputs, all-null columns, mixed date formats in the same column).

---

## Repository Structure

```
clovertex-pipeline/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .dockerignore
├── pipeline/
│   ├── main.py               ← Pipeline orchestrator
│   ├── ingestion/
│   │   └── loaders.py        ← Format-specific loaders
│   ├── cleaning/
│   │   └── cleaners.py       ← Per-dataset cleaning + genomics filtering
│   ├── transformation/
│   │   └── unify.py          ← Patient unification + supplementary joins
│   ├── stats/
│   │   ├── analytics.py      ← Tasks 3a–3f + bonus classifier
│   │   └── visualizations.py ← Task 4: all 6 plots
│   └── utils/
│       ├── io_utils.py       ← Parquet writer, manifest builder, checksums
│       └── logging_utils.py  ← Structured JSON logging to stdout
├── data/
│   ├── *.csv / *.json / *.parquet   ← Source files
│   └── reference/
├── datalake/                 ← Generated outputs (git-ignored)
│   ├── raw/
│   ├── refined/
│   └── consumption/
│       └── plots/
└── .github/
    └── workflows/
        └── ci.yml            ← Lint + Docker build check
```
