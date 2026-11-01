# 📊 HCL Technologies — Employee Salary Analysis

> **A data science consulting project** evaluating salary determinants for HCL Technologies employees using regression analysis, hypothesis testing, and advanced econometric modelling.

---

## 🗂️ Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Analysis Pipeline](#-analysis-pipeline)
- [Key Findings](#-key-findings)
- [Visualizations](#-visualizations)
- [Setup & Installation](#-setup--installation)
- [Technologies Used](#-technologies-used)
- [Authors](#-authors)

---

## 📌 Project Overview

This project is a **data science consulting engagement** contracted by HCL Technologies to evaluate and model the salary packages of its IT workforce. The dataset (collected in 2014) covers 526 employees with 25 variables spanning human capital, demographics, geography, and IT role specializations.

The analysis addresses the following core questions:

| Research Question | Method |
|---|---|
| What factors are primary determinants of employee salary? | Multiple Linear Regression (OLS) |
| Do education and experience exhibit diminishing/accelerating returns? | Quadratic (polynomial) terms |
| Is there a gender pay gap at HCL? | Coefficient testing & Forest plot |
| Does remote work or metro location affect salary? | Interaction & dummy variable analysis |
| Are model assumptions satisfied? | VIF, Breusch-Pagan, residual diagnostics |

---

## 📁 Dataset

| File | Description |
|---|---|
| [`IT_employees_full.csv`](IT_employees_full.csv) | Cleaned dataset — 526 employees × 25 variables |
| [`data_description.csv`](data_description.csv) | Variable codebook (names, types, descriptions) |

### Variable Summary

| # | Variable | Description | Type |
|---|---|---|---|
| 1 | `id` | Employee identification number | Identifier |
| 2 | `sal` | Average hourly salary (USD) | Numerical (continuous) |
| 3 | `edu` | Years of formal education | Numerical |
| 4 | `exper` | Years of professional experience | Numerical (continuous) |
| 5 | `ten` | Years of tenure at current company | Numerical |
| 6 | `noprev` | 1 if non-privileged employee | Nominal |
| 7 | `female` | 1 if female | Nominal |
| 8 | `married` | 1 if married | Nominal |
| 9 | `dep` | Number of dependents | Numerical |
| 10 | `metro` | 1 if lives in metro area | Nominal |
| 11 | `norcen` | 1 if north-central Indian province | Nominal |
| 12 | `south` | 1 if southern Indian province | Nominal |
| 13 | `west` | 1 if western Indian province | Nominal |
| 14 | `itreq` | 1 if IT requirements group | Nominal |
| 15 | `itprog` | 1 if IT programming group | Nominal |
| 16 | `itqual` | 1 if IT quality assurance group | Nominal |
| 17 | `itarch` | 1 if IT architecture group | Nominal |
| 18 | `itai` | 1 if IT work involves AI | Nominal |
| 19 | `itclient` | 1 if works only with IT client | Nominal |
| 20 | `itrd` | 1 if IT R&D division | Nominal |
| 21 | `ittech` | 1 if IT tech support | Nominal |
| 22 | `wrkhm` | 1 if works primarily from home | Nominal |
| 23 | `momed` | Mother's years of education | Numerical |
| 24 | `daded` | Father's years of education | Numerical |
| 25 | `sib` | Number of siblings | Numerical |

---

## 📂 Repository Structure

```
HCL_SALARY-ANALYSIS/
│
├── 📓 MP2/
│   ├── mp_2.ipynb                    # Main analysis notebook (full pipeline)
│   └── MP_2.docx                     # Formal project report
│
├── 📊 figures/                        # All generated plots and charts
│   ├── client_executive_dashboard.png
│   ├── client_chart1_salary_distribution.png
│   ├── client_chart2_human_capital_curves.png
│   ├── client_chart3_gender_gap_and_remote.png
│   ├── client_chart4_specialization_premiums.png
│   ├── plot1_salary_vs_log_salary_histograms.png
│   ├── plot2_human_capital_nonlinear_curves.png
│   ├── plot3_salary_differences_forest_plot.png
│   ├── regression_diagnostics_four_panel.png
│   ├── rec_plot1_pay_equity_action_plan.png
│   ├── rec_plot2_loyalty_penalty_vs_market.png
│   ├── rec_plot3_upskilling_and_career_ladder.png
│   ├── rec_plot4_workplace_location_policy_matrix.png
│   └── ... (23 figures total)
│
├── 📄 IT_employees_full.csv          # Clean dataset (526 employees × 25 vars)
├── 📄 data_description.csv           # Variable codebook
├── 📄 data_cleaning_description.md   # Detailed data preprocessing log
├── 🐍 generate_all_plots.py          # Script to regenerate all figures
├── 📋 requirements.txt               # Python package dependencies
└── 📖 README.md                      # This file
```

---

## ⚙️ Analysis Pipeline

### 1. Data Cleaning & Preprocessing
> Full details in [`data_cleaning_description.md`](data_cleaning_description.md)

| Step | Technique | Impact |
|---|---|---|
| Schema Alignment | Metadata mapping from `data_description.csv` | Standardized 25 column types |
| Type Coercion | `pd.to_numeric(errors='coerce')` | Removed 8 string typos across 7 columns |
| Missing Value Imputation | Mode (nominal) / Median (continuous) / Rounded Median (discrete) | Filled 30 missing values |
| Sign Rectification | Absolute value `.abs()` | Fixed 1 invalid negative value |
| Logical Consistency Audits | Domain boundary checks | Flagged 16 tenure, 285 regional, 107 client role conflicts |
| Outlier Detection | Tukey's IQR method (1.5 × IQR) | Flagged 29 extreme salary records |
| Deduplication | `df.duplicated()` | Confirmed 0 duplicates |
| Feature Scaling | Z-score standardization (`StandardScaler`) | Normalized 8 continuous variables |

### 2. Exploratory Data Analysis
- Salary distributions (raw vs. log-transformed)
- Correlation heatmaps and distributional profiling
- Group-level salary comparisons across gender, location, IT role

### 3. Regression Modelling
- **Baseline OLS Model** — linear predictors of `log(salary)`
- **Quadratic Extensions** — non-linear effects of `edu²`, `exper²`, `ten²`
- **Interaction Models** — e.g., `exper × edu`, `female × metro`
- **Model Diagnostics** — residual plots, Q-Q plots, scale-location, leverage
- **Heteroskedasticity Testing** — Breusch-Pagan test
- **Multicollinearity** — Variance Inflation Factor (VIF) analysis
- **Model Selection** — AIC/BIC comparison across specifications

### 4. Recommendations
Data-driven policy recommendations for HCL management on pay equity, tenure loyalty penalties, upskilling ladders, and remote work policies.

---

## 📈 Key Findings

| Finding | Direction | Significance |
|---|---|---|
| **Education** has diminishing returns beyond ~16 years | Concave (quadratic) | Significant |
| **Experience** shows accelerating salary growth | Convex (quadratic) | Significant |
| **Gender Pay Gap** — female employees earn less, controlling for all factors | Negative | Significant |
| **IT Architecture & AI roles** carry the highest salary premiums | Positive | Significant |
| **Metro location** boosts salary | Positive | Significant |
| **Work-from-home** is associated with salary discount | Negative | Significant |
| **Tenure loyalty penalty** — long-tenured employees underpaid vs. market | Negative | Significant |

---

## 🖼️ Visualizations

> A selection of key figures from the analysis:

| Figure | Description |
|---|---|
| `client_executive_dashboard.png` | High-level executive summary dashboard |
| `plot1_salary_vs_log_salary_histograms.png` | Salary distribution & log-normality check |
| `plot2_human_capital_nonlinear_curves.png` | Quadratic fits for education, experience, tenure |
| `plot3_salary_differences_forest_plot.png` | Forest plot of gender & group salary gaps |
| `regression_diagnostics_four_panel.png` | Full OLS diagnostic panel (4 plots) |
| `rec_plot1_pay_equity_action_plan.png` | Pay equity action plan visualization |

All figures are in [`figures/`](figures/).

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jaideep3020/HCL_SALARY-ANALYSIS.git
cd HCL_SALARY-ANALYSIS

# 2. Create and activate a virtual environment (recommended)
python -m venv env
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter to explore the analysis
jupyter notebook MP2/mp_2.ipynb
```

### Regenerate All Figures

```bash
python generate_all_plots.py
```

---

## 🔬 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and cleaning |
| `numpy` | Numerical computation |
| `matplotlib` / `seaborn` | Data visualization |
| `statsmodels` | OLS regression, hypothesis tests, diagnostics |
| `scikit-learn` | Feature scaling (`StandardScaler`) |
| `scipy` | Statistical tests |
| `formulaic` / `patsy` | Formula-based model specification |
| `openpyxl` | Excel file reading |
| `jupyter` | Interactive notebook environment |

---

## 👥 Authors

**Korrapati Jaideep** — Data Science & Analytics  
Course: DOM207 · Monsoon 2026  
Institution: Shiv Nadar University  

---

## 📜 License

This project is for academic and educational purposes. The dataset is proprietary to HCL Technologies and may not be redistributed or used for commercial purposes.

---

*Built with rigour, curiosity, and a lot of `statsmodels`*
