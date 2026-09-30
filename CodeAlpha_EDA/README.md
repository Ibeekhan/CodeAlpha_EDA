# CodeAlpha_EDA

**CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)**

An end-to-end exploratory analysis of the Titanic passenger dataset (891 passengers, 15 columns) using
pandas, seaborn, matplotlib and SciPy. The script follows the task checklist step by step.

## What the analysis covers

| Task requirement                         | How it is done                                                                 |
|------------------------------------------|--------------------------------------------------------------------------------|
| Ask meaningful questions first           | 7 guiding questions defined up front (see `eda_report.md`, section 1)          |
| Explore structure, variables, data types | dtype / non-null / unique tables and numeric summary                           |
| Identify trends, patterns, anomalies     | Survival by sex, class, age group, family size, port; correlation heatmap      |
| Test hypotheses and validate assumptions | 6 tests (chi-square, Mann-Whitney U) with effect sizes                         |
| Detect data issues                       | Missing values, outliers (IQR), duplicates, redundant columns                  |

## Key findings

- **Sex was the strongest factor:** 74.2% of women survived vs 18.9% of men (Cramer's V = 0.54).
- **Class mattered:** survival fell from 63.0% (1st class) to 47.3% (2nd) to 24.2% (3rd).
- **The two interact:** 96.8% of 1st-class women survived, compared with 13.5% of 3rd-class men.
- **Family size is non-linear:** small families (2-4) survived at 57.9%; those travelling alone at 30.4%; large families (5+) at only 16.1%.
- **Age alone was not a significant separator** (p = 0.16), although children under 12 survived at 58%.
- **Higher fares went with higher survival** (p < 0.001), largely because fare tracks class.
- **Data issues:** `deck` is 77% missing, `age` 20% missing, `fare` is heavily right-skewed with 116 outliers, and three columns (`alive`, `class`, `embark_town`) duplicate others.

The full write-up, generated from the data by the script, is in [`outputs/eda_report.md`](outputs/eda_report.md).
Association is not causation: sex and class are correlated with each other, so their effects overlap.

## Run it

```bash
pip install -r requirements.txt
python eda.py
```

The dataset is cached in `data/titanic.csv` (public Titanic data distributed with seaborn), so the
script also works offline. Outputs are written to `outputs/`:

```
outputs/
├── eda_report.md
├── 01_missing_values.png
├── 02_survival_breakdowns.png
├── 03_age_fare_distributions.png
├── 04_survival_sex_by_class.png
└── 05_correlation_heatmap.png
```

## Project structure

```
CodeAlpha_EDA/
├── eda.py
├── data/titanic.csv
├── outputs/
├── requirements.txt
└── README.md
```
