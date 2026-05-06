# ITCS Alumni Employment Outcomes — 2026 Retrospective

Analysis of employment outcomes for alumni of the [Industry Team Case Study (ITCS)](https://www.tdccbr.ca/itcs) program at the University of Toronto, covering cohorts 2021–2025, with a 5-year comparison to the previous period (2016–2020, Sealey et al. 2021).

## Repository contents

```
itcs_analysis_2026.py       Main analysis: QC, sector classification, 6 figures
itcs_comparison_2021.py     5-year comparison: 2016–2020 vs 2021–2025, 6 figures
itcs_cleaned_data.csv       De-duplicated, QC-validated output dataset (208 rows)
figures/                    All 12 output figures (fig1–6, figC1–6)
```

## Requirements

```
Python 3.7+
pandas >= 0.24
numpy
matplotlib >= 3.0
```

## Usage

```bash
python3 itcs_analysis_2026.py      # produces itcs_cleaned_data.csv + figures/fig1–6
python3 itcs_comparison_2021.py    # produces figures/figC1–6
```

Both scripts read `Where are Trainees Now_ - Raw Data.csv` (not included; contains personal LinkedIn data).

## Data processing rules

### De-duplication
- Each person is counted only in their **first** cohort year.
- Identity is resolved by LinkedIn URL first (normalized to `linkedin.com/in/username`, stripping `/details/experience/` and query-param suffixes), then by normalized name.
- **Ghost rows** are excluded before de-duplication: rows where `position=FALSE`, `graduated=FALSE`, and LinkedIn is blank — these are empty participant slots with no recoverable data (6 rows removed).
- **Max-graduated override**: if any row for a person across all years shows `graduated=TRUE`, that person is counted as graduated regardless of what their first-year row says.

### Trainee types
- **MHSc is tracked separately from MSc** (never combined into a generic "Master" category).
- Types: MSc, MHSc, PhD, Postdoc, PharmD, Other, Unknown.

### Sector classification
Sectors for 2021–2025:

| Sector | Description |
|---|---|
| Industry | Private-sector employment |
| Academia | University faculty, postdoc, or research staff |
| Further Schooling | Enrolled in a new professional or graduate degree (JD, MD, DDS, law school, or MSc/MHSc → PhD transition at a university) |
| Non-profit | Registered charities, hospitals, research institutes (e.g. SickKids, CIHR, MITACS, OBI) |
| Government | Federal or provincial government roles |
| Unemployed | Explicitly unemployed, seeking work, or no current position |
| Unknown | Untraced; LinkedIn not found and no sector data available |

"Other" sector entries in the raw data are resolved by keyword matching on role and company fields into Non-profit, Unemployed, or Unknown.

### Internship classification
- **Paid internships only.** Entries containing "unpaid", "not paid", or "volunteer" are excluded.
- Undergrad co-ops without a graduate-level component are excluded.
- Postdoc entries are always classified as No (not applicable).

## Key results (2021–2025)

| Metric | Value |
|---|---|
| Unique participants (de-duplicated) | 208 |
| Graduates analysed | 160 |
| Still in training | 48 |
| **Industry employment** | **92 / 160 = 57.5%** |
| Academia | 29 / 160 = 18.1% |
| Non-profit | 14 / 160 = 8.8% |
| Unemployed | 16 / 160 = 10.0% |
| Further Schooling | 7 / 160 = 4.4% |
| Government | 2 / 160 = 1.2% |
| Unknown | 0 / 160 = 0.0% |
| Had a paid internship | 62 / 160 = 39% |

### Benchmarks
- **Sealey et al. 2021** (cohorts 2016–2020, n=106 graduates): 55% industry — ITCS 2021–2025 is +2.5 pp.
- **Reithmeier et al. 2019** (10,000 PhDs Project, UofT graduates 2000–2015): ~20% industry for all disciplines, ~25% for life sciences. ITCS 2021–2025 is roughly 2.3× the life-sciences baseline.

## Figures

### itcs_analysis_2026.py

| File | Description |
|---|---|
| `fig1_cohort_composition.png` | Cohort size by year and trainee type (stacked bar) |
| `fig2_sector_by_cohort.png` | Sector distribution per cohort year — graduates only (stacked %) |
| `fig3_industry_trend.png` | Industry rate 2016–2025 with UofT PhD benchmarks |
| `fig4_internship_effect.png` | Sector outcomes split by paid internship (Yes vs No) |
| `fig5_sector_by_position.png` | Sector distribution by trainee type (MSc, MHSc, PhD, Postdoc) |
| `fig6_cohort_comparison.png` | 2016–2020 vs 2021–2025 side-by-side bar comparison |

### itcs_comparison_2021.py

| File | Description |
|---|---|
| `figC1_program_growth.png` | Participant and graduate counts 2016–2025 |
| `figC2_trainee_type_comparison.png` | Trainee type composition — pie charts for each period |
| `figC3_sector_comparison.png` | Sector stacked bars with change arrows |
| `figC4_sector_trends.png` | 4-panel sector trend lines 2016–2025 |
| `figC5_sector_delta.png` | Change in sector share (Period B − Period A, pp) |
| `figC6_industry_10yr_unified.png` | Unified 10-year industry rate with benchmarks |

## Methodology notes

### Comparison to 2016–2020
Period A (Sealey et al. 2021) reported "first employment after graduation" from a Jan 2021 snapshot.
Period B (this analysis) reports "current role" at the time of data collection (2025–2026), which may reflect career changes for earlier cohorts.

Period A combined "Academia" and "Healthcare" into one category.
Period B separates Academia, Further Schooling, and Unemployed. The apparent 10 pp drop in Academia is largely a methodology artefact. The 14 pp drop in Unknown reflects improved LinkedIn coverage.

### Data not included
The raw survey CSV (`Where are Trainees Now_ - Raw Data.csv`) is not included in this repository as it contains personal identifiers (names, LinkedIn URLs). The cleaned output (`itcs_cleaned_data.csv`) contains no LinkedIn URLs.

## References

- Sealey, D. et al. (2021). *Life Science Industry Job Simulation Program Alumni* [poster].
- Reithmeier, R. A. F. et al. (2019). A 10,000 PhDs Project at the University of Toronto: Using employment outcome data for PhDs to inform graduate education. *PLOS ONE*, 14(8), e0221346. https://doi.org/10.1371/journal.pone.0221346
