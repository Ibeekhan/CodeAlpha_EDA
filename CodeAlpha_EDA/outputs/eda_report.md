# Titanic Dataset - Exploratory Data Analysis Report

## 1. Questions asked before analysis

1. Which variables are available, what are their types, and how complete are they?
2. Did men and women survive at different rates?
3. Did passenger class (a proxy for wealth) affect survival?
4. Were survivors younger than non-survivors?
5. Did paying a higher fare go with a higher chance of survival?
6. Did travelling alone or with family change survival odds?
7. Are there anomalies (outliers, missing values, redundant columns) to fix before modelling?

## 2. Data structure

- **Rows:** 891  |  **Columns:** 15

|             | dtype   |   non-null |   unique | example     |
|:------------|:--------|-----------:|---------:|:------------|
| survived    | int64   |        891 |        2 | 0           |
| pclass      | int64   |        891 |        3 | 3           |
| sex         | str     |        891 |        2 | male        |
| age         | float64 |        714 |       88 | 22.0        |
| sibsp       | int64   |        891 |        7 | 1           |
| parch       | int64   |        891 |        7 | 0           |
| fare        | float64 |        891 |      248 | 7.25        |
| embarked    | str     |        889 |        3 | S           |
| class       | str     |        891 |        3 | Third       |
| who         | str     |        891 |        3 | man         |
| adult_male  | bool    |        891 |        2 | True        |
| deck        | str     |        203 |        7 | C           |
| embark_town | str     |        889 |        3 | Southampton |
| alive       | str     |        891 |        2 | no          |
| alone       | bool    |        891 |        2 | False       |

**Numeric summary**

|          |   count |   mean |   std |   min |   25% |   50% |   75% |    max |
|:---------|--------:|-------:|------:|------:|------:|------:|------:|-------:|
| survived |     891 |   0.38 |  0.49 |  0    |  0    |  0    |     1 |   1    |
| pclass   |     891 |   2.31 |  0.84 |  1    |  2    |  3    |     3 |   3    |
| age      |     714 |  29.7  | 14.53 |  0.42 | 20.12 | 28    |    38 |  80    |
| sibsp    |     891 |   0.52 |  1.1  |  0    |  0    |  0    |     1 |   8    |
| parch    |     891 |   0.38 |  0.81 |  0    |  0    |  0    |     0 |   6    |
| fare     |     891 |  32.2  | 49.69 |  0    |  7.91 | 14.45 |    31 | 512.33 |

## 3. Data quality checks

**Missing values**

|             |   missing |   percent |
|:------------|----------:|----------:|
| deck        |       688 |      77.2 |
| age         |       177 |      19.9 |
| embarked    |         2 |       0.2 |
| embark_town |         2 |       0.2 |

- **Duplicate rows:** 107
- **Fare outliers (IQR rule, > 65.6):** 116 passengers (13.0%); max fare = 512.3
- **Zero-fare passengers:** 15
- **Redundant columns:** `alive` duplicates `survived`; `class` duplicates `pclass`; `embark_town` duplicates `embarked`

## 4. Trends and patterns

- **Overall survival rate:** 38.4%
- **By sex:** female 74.2% vs male 18.9%
- **By class:** class 1 63.0%, class 2 47.3%, class 3 24.2%
- **By port of embarkation:** C 55.4%, Q 39.0%, S 33.7%
- **Travelling alone:** 30.4% survived vs 50.6% with family

**Survival rate by sex and class**

|   pclass |   female |   male |
|---------:|---------:|-------:|
|        1 |     96.8 |   36.9 |
|        2 |     92.1 |   15.7 |
|        3 |     50   |   13.5 |

**Survival rate by age group**

| age            |   survival % |   passengers |
|:---------------|-------------:|-------------:|
| Child (<=12)   |         58   |           69 |
| Teen (13-18)   |         42.9 |           70 |
| Adult (19-35)  |         38.3 |          358 |
| Middle (36-60) |         40   |          195 |
| Senior (60+)   |         22.7 |           22 |

**Survival rate by family size**

| family_size   |   survival % |   passengers |
|:--------------|-------------:|-------------:|
| Alone         |         30.4 |          537 |
| Small (2-4)   |         57.9 |          292 |
| Large (5+)    |         16.1 |           62 |

- **Variables most correlated with survival:** `is_female` (+0.54), `pclass` (-0.34), `fare` (+0.26)

## 5. Hypothesis tests

All tests use a 5% significance level. Chi-square tests are used for categorical variables and the non-parametric Mann-Whitney U test for age and fare, because fare is heavily skewed and does not satisfy the normality assumption of a t-test.

| hypothesis                                     | test           |   statistic |   p-value | effect size   | result                        |
|:-----------------------------------------------|:---------------|------------:|----------:|:--------------|:------------------------------|
| H1: Survival depends on sex                    | Chi-square     |      260.72 |  1.2e-58  | 0.541         | statistically significant     |
| H2: Survival depends on passenger class        | Chi-square     |      102.89 |  4.55e-23 | 0.34          | statistically significant     |
| H3: Survival depends on travelling alone       | Chi-square     |       36    |  1.97e-09 | 0.201         | statistically significant     |
| H4: Survival depends on port of embarkation    | Chi-square     |       26.49 |  1.77e-06 | 0.173         | statistically significant     |
| H5: Survivors and non-survivors differ in age  | Mann-Whitney U |    57682    |  0.16     | r = 0.062     | not statistically significant |
| H6: Survivors and non-survivors differ in fare | Mann-Whitney U |   129952    |  4.55e-22 | r = 0.384     | statistically significant     |

- Median age: survivors 28.0 vs non-survivors 28.0
- Median fare: survivors 26.0 vs non-survivors 10.5

*A significant p-value shows an association exists; it does not prove causation. Sex and class are also correlated with each other (see the correlation heatmap), so their effects overlap.*

## 6. Issues to address in further analysis

- `deck` is missing for 77.2% of rows (688); too sparse to impute, so drop it or use a 'Unknown' category.
- `age` is missing for 19.9% of rows (177); impute (e.g. median by class/sex) rather than dropping, and check whether missingness is related to survival.
- `embarked` has 2 missing value(s) (0.2%); safe to impute with the mode or drop.
- `embark_town` has 2 missing value(s) (0.2%); safe to impute with the mode or drop.
- `fare` is strongly right-skewed with 116 outliers; consider a log transform.
- Redundant columns to drop before modelling: `alive` duplicates `survived`; `class` duplicates `pclass`; `embark_town` duplicates `embarked`.
- Duplicate rows: 107 (these are unlikely to be true duplicates since no passenger id exists, so review before removing).
- Sex and class interact strongly; model them together rather than separately.
