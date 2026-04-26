# Plot Descriptions

| Plot | File | Description |
|------|------|-------------|
| 1 | `01_patient_demographics.png` | Age distribution histogram with mean/median overlay, and sex split bar chart across the unified patient cohort (Alpha + Beta sites). |
| 2 | `02_diagnosis_frequency.png` | Horizontal bar chart showing the top 15 ICD-10 disease chapters ranked by unique patient count (not diagnosis count, to avoid inflating chapters with repeat visits). |
| 3 | `03_lab_distributions.png` | KDE + histogram distribution plots for the four most-recorded lab tests. Green bands indicate the normal reference range from `lab_test_ranges.json`. |
| 4 | `04_genomics_scatter.png` | Scatter plot of allele frequency (x) vs read depth (y) for all reliable variant calls, coloured by clinical significance. The dashed horizontal line marks the 20x read depth reliability threshold. |
| 5 | `05_high_risk_summary.png` | Two panels summarising the high-risk cohort (patients with diabetic-range HbA1c AND at least one pathogenic variant): HbA1c distribution and pathogenic variant burden distribution. |
| 6 | `06_data_quality_overview.png` | Grouped bar chart summarising pipeline quality metrics per source dataset: rows ingested, rows after cleaning, duplicates removed, and nulls handled. |
