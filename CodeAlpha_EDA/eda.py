"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)

Dataset: Titanic passengers (891 rows, 15 columns), a classic real-world dataset
with missing values, mixed data types and outliers - ideal for EDA practice.

The script follows the task checklist:
    1. Ask meaningful questions about the dataset before analysing it
    2. Explore the data structure (variables and data types)
    3. Identify trends, patterns and anomalies
    4. Test hypotheses and validate assumptions with statistics + visualisation
    5. Detect data issues to address in further analysis

Outputs:
    outputs/*.png          - figures
    outputs/eda_report.md  - written report (all numbers are computed, not typed in)

Usage:
    python eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "titanic.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

QUESTIONS = [
    "Which variables are available, what are their types, and how complete are they?",
    "Did men and women survive at different rates?",
    "Did passenger class (a proxy for wealth) affect survival?",
    "Were survivors younger than non-survivors?",
    "Did paying a higher fare go with a higher chance of survival?",
    "Did travelling alone or with family change survival odds?",
    "Are there anomalies (outliers, missing values, redundant columns) to fix before modelling?",
]


def load_data() -> pd.DataFrame:
    """Load the dataset from data/titanic.csv, downloading + caching it on first run."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = sns.load_dataset("titanic")
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def cramers_v(table: pd.DataFrame) -> float:
    """Effect size for a chi-square test (0 = no association, 1 = perfect)."""
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def verdict(p: float) -> str:
    return "statistically significant" if p < 0.05 else "not statistically significant"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. Structure
# --------------------------------------------------------------------------- #
def explore_structure(df: pd.DataFrame, rpt: list[str]) -> None:
    rpt += ["## 2. Data structure", "", f"- **Rows:** {len(df)}  |  **Columns:** {df.shape[1]}", ""]
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non-null": df.notna().sum(),
            "unique": df.nunique(),
            "example": [df[c].dropna().iloc[0] for c in df.columns],
        }
    )
    rpt += [info.to_markdown(), ""]
    rpt += ["**Numeric summary**", "", df.describe().round(2).T.to_markdown(), ""]


# --------------------------------------------------------------------------- #
# 2. Data quality
# --------------------------------------------------------------------------- #
def check_quality(df: pd.DataFrame, rpt: list[str]) -> dict:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    miss_pct = (missing / len(df) * 100).round(1)

    dup = int(df.duplicated().sum())
    q1, q3 = df["fare"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    fare_outliers = int((df["fare"] > upper).sum())
    zero_fares = int((df["fare"] == 0).sum())

    # Columns that duplicate other columns
    redundant = []
    if (df["survived"].map({0: "no", 1: "yes"}) == df["alive"]).all():
        redundant.append("`alive` duplicates `survived`")
    if (df["pclass"].astype(str) == df["class"].map({"First": "1", "Second": "2", "Third": "3"})).all():
        redundant.append("`class` duplicates `pclass`")
    if (df["embarked"].dropna().map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
            == df.loc[df["embarked"].notna(), "embark_town"]).all():
        redundant.append("`embark_town` duplicates `embarked`")

    rpt += ["## 3. Data quality checks", "", "**Missing values**", ""]
    rpt += [pd.DataFrame({"missing": missing, "percent": miss_pct}).to_markdown(), ""]
    rpt += [
        f"- **Duplicate rows:** {dup}",
        f"- **Fare outliers (IQR rule, > {upper:.1f}):** {fare_outliers} passengers "
        f"({fare_outliers / len(df):.1%}); max fare = {df['fare'].max():.1f}",
        f"- **Zero-fare passengers:** {zero_fares}",
        f"- **Redundant columns:** {'; '.join(redundant) if redundant else 'none found'}",
        "",
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=miss_pct.values, y=miss_pct.index, color="#4C78A8", ax=ax)
    ax.set(title="Missing values by column", xlabel="% of rows missing", ylabel="")
    for i, v in enumerate(miss_pct.values):
        ax.text(v + 0.5, i, f"{v}%", va="center")
    ax.set_xlim(0, max(miss_pct.max() * 1.15, 5))
    save(fig, "01_missing_values.png")
    return {"missing": missing, "miss_pct": miss_pct, "fare_outliers": fare_outliers,
            "duplicates": dup, "redundant": redundant}


# --------------------------------------------------------------------------- #
# 3. Patterns and trends
# --------------------------------------------------------------------------- #
def explore_patterns(df: pd.DataFrame, rpt: list[str]) -> dict:
    overall = df["survived"].mean()
    by_sex = df.groupby("sex")["survived"].mean()
    by_class = df.groupby("pclass")["survived"].mean()
    by_embarked = df.groupby("embarked")["survived"].mean()

    df = df.assign(family_size=df["sibsp"] + df["parch"] + 1)
    by_alone = df.groupby("alone")["survived"].mean()
    fam_bins = pd.cut(df["family_size"], [0, 1, 4, 20], labels=["Alone", "Small (2-4)", "Large (5+)"])
    by_family = df.groupby(fam_bins, observed=True)["survived"].agg(["mean", "count"])
    sex_class = df.pivot_table(index="pclass", columns="sex", values="survived", aggfunc="mean")
    age_bins = pd.cut(df["age"], [0, 12, 18, 35, 60, 100],
                      labels=["Child (<=12)", "Teen (13-18)", "Adult (19-35)", "Middle (36-60)", "Senior (60+)"])
    by_age = df.groupby(age_bins, observed=True)["survived"].agg(["mean", "count"])

    rpt += [
        "## 4. Trends and patterns", "",
        f"- **Overall survival rate:** {overall:.1%}",
        f"- **By sex:** female {by_sex['female']:.1%} vs male {by_sex['male']:.1%}",
        "- **By class:** " + ", ".join(f"class {k} {v:.1%}" for k, v in by_class.items()),
        "- **By port of embarkation:** " + ", ".join(f"{k} {v:.1%}" for k, v in by_embarked.items()),
        f"- **Travelling alone:** {by_alone[True]:.1%} survived vs {by_alone[False]:.1%} with family",
        "",
        "**Survival rate by sex and class**", "", (sex_class * 100).round(1).to_markdown(), "",
        "**Survival rate by age group**", "",
        by_age.assign(mean=(by_age["mean"] * 100).round(1)).rename(
            columns={"mean": "survival %", "count": "passengers"}).to_markdown(), "",
        "**Survival rate by family size**", "",
        by_family.assign(mean=(by_family["mean"] * 100).round(1)).rename(
            columns={"mean": "survival %", "count": "passengers"}).to_markdown(), "",
    ]

    # Figure 2: survival rate breakdowns
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    sns.barplot(data=df, x="sex", y="survived", order=["female", "male"], hue="sex",
                palette={"female": "#E45756", "male": "#4C78A8"},
                legend=False, errorbar=("ci", 95), ax=axes[0])
    axes[0].set(title="Survival rate by sex", ylabel="Survival rate", xlabel="")
    sns.barplot(data=df, x="pclass", y="survived", color="#4C78A8", errorbar=("ci", 95), ax=axes[1])
    axes[1].set(title="Survival rate by passenger class", xlabel="Class (1 = highest)", ylabel="")
    sns.barplot(data=df, x=fam_bins, y=df["survived"], color="#4C78A8", errorbar=("ci", 95), ax=axes[2])
    axes[2].set(title="Survival rate by family size", xlabel="", ylabel="")
    for ax in axes:
        ax.axhline(overall, ls="--", color="grey", lw=1)
    axes[0].text(0.02, overall + 0.015, f"overall {overall:.0%}", color="grey", transform=axes[0].get_yaxis_transform())
    save(fig, "02_survival_breakdowns.png")

    # Figure 3: age and fare distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(data=df, x="age", hue="alive", bins=30, multiple="stack",
                 palette={"no": "#E45756", "yes": "#4C78A8"}, ax=axes[0])
    axes[0].set(title="Age distribution by outcome", xlabel="Age")
    sns.histplot(df["fare"], bins=40, color="#4C78A8", ax=axes[1])
    axes[1].set(title="Fare distribution (right-skewed)", xlabel="Fare")
    sns.boxplot(data=df, x="pclass", y="fare", color="#9ecae9", ax=axes[2])
    axes[2].set(title="Fare by class (log scale)", xlabel="Class", yscale="symlog")
    axes[2].set_ylim(bottom=0)  # avoid meaningless negative ticks on the symlog axis
    save(fig, "03_age_fare_distributions.png")

    # Figure 4: sex x class interaction
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex",
                palette={"female": "#E45756", "male": "#4C78A8"}, errorbar=("ci", 95), ax=ax)
    ax.set(title="Survival rate by class and sex", xlabel="Class (1 = highest)", ylabel="Survival rate")
    save(fig, "04_survival_sex_by_class.png")

    # Figure 5: correlation heatmap
    num = df.assign(is_female=(df["sex"] == "female").astype(int))[
        ["survived", "pclass", "is_female", "age", "sibsp", "parch", "fare"]]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix (numeric variables)")
    save(fig, "05_correlation_heatmap.png")

    strongest = corr["survived"].drop("survived").abs().sort_values(ascending=False)
    rpt += [f"- **Variables most correlated with survival:** "
            + ", ".join(f"`{k}` ({corr.loc['survived', k]:+.2f})" for k in strongest.index[:3]), ""]
    return {"overall": overall, "by_sex": by_sex, "by_class": by_class, "sex_class": sex_class}


# --------------------------------------------------------------------------- #
# 4. Hypothesis tests
# --------------------------------------------------------------------------- #
def test_hypotheses(df: pd.DataFrame, rpt: list[str]) -> None:
    rpt += ["## 5. Hypothesis tests", "",
            "All tests use a 5% significance level. Chi-square tests are used for categorical "
            "variables and the non-parametric Mann-Whitney U test for age and fare, because fare "
            "is heavily skewed and does not satisfy the normality assumption of a t-test.", ""]

    rows = []
    for col, label in [("sex", "H1: Survival depends on sex"),
                       ("pclass", "H2: Survival depends on passenger class"),
                       ("alone", "H3: Survival depends on travelling alone"),
                       ("embarked", "H4: Survival depends on port of embarkation")]:
        table = pd.crosstab(df[col], df["survived"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append({"hypothesis": label, "test": "Chi-square", "statistic": round(chi2, 2),
                     "p-value": f"{p:.2e}", "effect size (Cramer's V)": round(cramers_v(table), 3),
                     "result": verdict(p)})

    for col, label in [("age", "H5: Survivors and non-survivors differ in age"),
                       ("fare", "H6: Survivors and non-survivors differ in fare")]:
        surv = df.loc[df["survived"] == 1, col].dropna()
        died = df.loc[df["survived"] == 0, col].dropna()
        u, p = stats.mannwhitneyu(surv, died, alternative="two-sided")
        # rank-biserial correlation as effect size
        r = 1 - 2 * u / (len(surv) * len(died))
        rows.append({"hypothesis": label, "test": "Mann-Whitney U", "statistic": round(u, 1),
                     "p-value": f"{p:.2e}", "effect size (Cramer's V)": f"r = {abs(r):.3f}",
                     "result": verdict(p)})

    rpt += [pd.DataFrame(rows).rename(columns={"effect size (Cramer's V)": "effect size"}).to_markdown(index=False), ""]

    med_age_s, med_age_d = df.loc[df.survived == 1, "age"].median(), df.loc[df.survived == 0, "age"].median()
    med_fare_s, med_fare_d = df.loc[df.survived == 1, "fare"].median(), df.loc[df.survived == 0, "fare"].median()
    rpt += [f"- Median age: survivors {med_age_s:.1f} vs non-survivors {med_age_d:.1f}",
            f"- Median fare: survivors {med_fare_s:.1f} vs non-survivors {med_fare_d:.1f}", "",
            "*A significant p-value shows an association exists; it does not prove causation. "
            "Sex and class are also correlated with each other (see the correlation heatmap), so "
            "their effects overlap.*", ""]


# --------------------------------------------------------------------------- #
# 5. Issues for further analysis
# --------------------------------------------------------------------------- #
def summarise_issues(q: dict, df: pd.DataFrame, rpt: list[str]) -> None:
    rpt += ["## 6. Issues to address in further analysis", ""]
    for col, pct in q["miss_pct"].items():
        n = int(q["missing"][col])
        if col == "deck":
            rpt.append(f"- `deck` is missing for {pct}% of rows ({n}); too sparse to impute, so "
                       "drop it or use a 'Unknown' category.")
        elif col == "age":
            rpt.append(f"- `age` is missing for {pct}% of rows ({n}); impute (e.g. median by class/sex) "
                       "rather than dropping, and check whether missingness is related to survival.")
        else:
            rpt.append(f"- `{col}` has {n} missing value(s) ({pct}%); safe to impute with the mode or drop.")
    rpt.append(f"- `fare` is strongly right-skewed with {q['fare_outliers']} outliers; consider a log transform.")
    if q["redundant"]:
        rpt.append("- Redundant columns to drop before modelling: " + "; ".join(q["redundant"]) + ".")
    rpt.append(f"- Duplicate rows: {q['duplicates']} (these are unlikely to be true duplicates since "
               "no passenger id exists, so review before removing).")
    rpt += ["- Sex and class interact strongly; model them together rather than separately.", ""]


def main() -> None:
    df = load_data()
    rpt = ["# Titanic Dataset - Exploratory Data Analysis Report", "",
           "## 1. Questions asked before analysis", ""]
    rpt += [f"{i}. {q}" for i, q in enumerate(QUESTIONS, 1)] + [""]

    explore_structure(df, rpt)
    quality = check_quality(df, rpt)
    explore_patterns(df, rpt)
    test_hypotheses(df, rpt)
    summarise_issues(quality, df, rpt)

    (OUT / "eda_report.md").write_text("\n".join(rpt), encoding="utf-8")
    print(f"EDA complete. Report and figures saved to {OUT}/")
    print("\n".join(rpt[rpt.index('## 4. Trends and patterns'):rpt.index('## 5. Hypothesis tests')][:12]))


if __name__ == "__main__":
    main()
