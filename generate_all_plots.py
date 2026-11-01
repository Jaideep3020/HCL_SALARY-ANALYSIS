"""
generate_all_plots.py
=====================
Unified Master Visualization & Econometric Graphics Suite
Course: DOM 207 - Econometric Consulting Project
Dataset: IT_employees_full.csv (Exact N = 523 Sample)

This single consolidated script merges all standalone visualization modules:
1. DESCRIPTIVE & DISTRIBUTION SUITE:
   - dist_histograms.png
   - dist_correlation_heatmap.png
   - dist_skew_kurtosis.png
   - dist_boxplots.png

2. REPORT ECONOMETRIC SUITE:
   - plot1_salary_vs_log_salary_histograms.png
   - plot2_human_capital_nonlinear_curves.png
   - plot3_salary_differences_forest_plot.png

3. REGRESSION DIAGNOSTICS SUITE:
   - regression_diagnostics_four_panel.png

4. CLIENT PRESENTATION SUITE:
   - client_chart1_salary_distribution.png
   - client_chart2_human_capital_curves.png
   - client_chart3_gender_gap_and_remote.png
   - client_chart4_specialization_premiums.png
   - client_executive_dashboard.png

5. STRATEGIC HR RECOMMENDATIONS SUITE:
   - rec_plot1_pay_equity_action_plan.png
   - rec_plot2_loyalty_penalty_vs_market.png
   - rec_plot3_upskilling_and_career_ladder.png
   - rec_plot4_workplace_location_policy_matrix.png

All generated figures are saved directly to the 'figures/' directory.
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ----------------------------------------------------------------------
# GLOBAL VISUAL STYLING SYSTEM
# ----------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#C8CFD6'
plt.rcParams['axes.linewidth'] = 0.9

NAVY = '#1B365D'
STEEL_BLUE = '#2B5B84'
TEAL = '#008080'
CORAL = '#C0392B'
AMBER = '#D97706'
SLATE = '#2C3E50'
GRAY = '#6C757D'
LIGHT_BG = '#F8F9FA'
GREEN = '#1E8449'
PURPLE = '#6F42C1'

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
ARTIFACT_DIR = r"C:\Users\jaide\.gemini\antigravity-ide\brain\4ef4dfdc-1bf1-44ed-80ed-bd6dd419c6b0"

def save_fig(fig, filename, dpi=300, **kwargs):
    """Helper to save a figure into figures/ and copy to artifact directory."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=dpi, facecolor='white', bbox_inches='tight', **kwargs)
    plt.close(fig)
    print(f"  [+] Saved: {filepath}")
    if os.path.exists(ARTIFACT_DIR):
        try:
            shutil.copy(filepath, os.path.join(ARTIFACT_DIR, filename))
        except Exception:
            pass

# ----------------------------------------------------------------------
# DATA LOAD & REPRODUCIBLE MODELING SAMPLE PIPELINE (N = 523)
# ----------------------------------------------------------------------
def load_and_preprocess_data():
    """Loads and cleans dataset matching mp_2.ipynb exactly (N = 523)."""
    desc_df = pd.read_csv("data_description.csv")
    df_raw = pd.read_csv("IT_employees_full.csv")

    df_raw.columns = df_raw.columns.str.strip().str.lower()
    desc_df.columns = desc_df.columns.str.strip().str.lower()
    desc_df["variable"] = desc_df["variable"].str.strip().str.lower()
    desc_df["type"] = desc_df["type"].str.strip().str.lower()

    df_raw = df_raw.rename(columns={"id": "id", "noprev": "nonprev"})
    desc_df = desc_df.rename(columns={"id": "id", "noprev": "nonprev"})
    col_types = dict(zip(desc_df["variable"], desc_df["type"]))
    col_types["nonprev"] = "nominal"

    df_clean = df_raw.copy()
    for col in df_clean.columns:
        if col in col_types:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    binary_cols = [
        "nonprev", "female", "married", "metro", "norcen", "south", "west",
        "itreq", "itprog", "itqual", "itarch", "itai", "itclient", "itrd", "ittech", "wrkhm"
    ]
    binary_cols = [c for c in binary_cols if c in df_clean.columns]
    for col in binary_cols:
        invalid = df_clean[col].notna() & ~df_clean[col].isin([0, 1])
        df_clean.loc[invalid, col] = np.nan

    range_rules = {
        "sal": (0, None), "edu": (0, 30), "exper": (0, None), "ten": (0, None),
        "momed": (0, 30), "daded": (0, 30), "sib": (0, 25), "dep": (0, 20)
    }
    for col, (lower, upper) in range_rules.items():
        if col in df_clean.columns:
            invalid = pd.Series(False, index=df_clean.index)
            if lower is not None: invalid |= df_clean[col] < lower
            if upper is not None: invalid |= df_clean[col] > upper
            df_clean.loc[invalid, col] = np.nan

    for col in df_clean.columns:
        if col not in col_types or df_clean[col].isna().sum() == 0: continue
        v_type = col_types[col]
        if v_type == "nominal":
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        elif v_type == "numerical":
            df_clean[col] = df_clean[col].fillna(round(df_clean[col].median()))
        elif v_type == "numerical(continuous)":
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Exact modeling filter from Cell 2
    model_df = df_clean[
        (df_clean["sal"] > 0) & (df_clean["sal"] < 100) &
        (df_clean["edu"] <= 30) & (df_clean["momed"] <= 30) &
        (df_clean["daded"] <= 30) & (df_clean["sib"] <= 25)
    ].copy().reset_index(drop=True)

    df_full = model_df.copy()
    df_full["log_sal"] = np.log(df_full["sal"])
    df_full["ln_sal"] = df_full["log_sal"]
    df_full["edu2"] = df_full["edu"] ** 2
    df_full["exp2"] = df_full["exper"] ** 2
    df_full["ten2"] = df_full["ten"] ** 2
    df_full["outside_exp"] = np.maximum(0, df_full["exper"] - df_full["ten"])
    df_full["outside_exp2"] = df_full["outside_exp"] ** 2

    # Centered features
    for col in ["edu", "exper", "ten", "momed", "daded", "dep"]:
        df_full[f"{col}_c"] = df_full[col] - df_full[col].mean()

    df_full["edu_c_sq"] = df_full["edu_c"] ** 2
    df_full["exper_c_sq"] = df_full["exper_c"] ** 2
    df_full["ten_c_sq"] = df_full["ten_c"] ** 2

    df_full["int_edu_female"] = df_full["edu_c"] * df_full["female"]
    df_full["int_edu_momed"] = df_full["edu_c"] * df_full["momed_c"]
    df_full["int_edu_daded"] = df_full["edu_c"] * df_full["daded_c"]
    df_full["int_exper_nonprev"] = df_full["exper_c"] * df_full["nonprev"]
    df_full["int_edu_nonprev"] = df_full["edu_c"] * df_full["nonprev"]
    df_full["int_dep_wrkhm"] = df_full["dep_c"] * df_full["wrkhm"]
    df_full["int_female_wrkhm"] = df_full["female"] * df_full["wrkhm"]
    df_full["int_edu_metro"] = df_full["edu_c"] * df_full["metro"]

    # Final Combined Econometric Model Matrix
    main_effects = [
        "edu_c", "exper_c", "ten_c", "female", "married", "dep_c",
        "metro", "nonprev", "momed_c", "daded_c", "itai", "itclient",
        "ittech", "itrd", "wrkhm"
    ]
    quadratic_terms = ["edu_c_sq", "exper_c_sq", "ten_c_sq"]
    interaction_terms = [
        "int_edu_female", "int_edu_momed", "int_edu_daded",
        "int_exper_nonprev", "int_edu_nonprev", "int_dep_wrkhm",
        "int_female_wrkhm", "int_edu_metro"
    ]
    all_predictors = main_effects + quadratic_terms + interaction_terms

    X_combined = sm.add_constant(df_full[all_predictors], has_constant="add")
    m_combined_ols = sm.OLS(df_full["log_sal"], X_combined).fit()
    m_combined_hc3 = sm.OLS(df_full["log_sal"], X_combined).fit(cov_type="HC3")

    return df_clean, df_full, col_types, X_combined, m_combined_ols, m_combined_hc3


# ======================================================================
# SUITE 1: DESCRIPTIVE & DISTRIBUTION VISUALS
# ======================================================================
def generate_descriptive_plots(df_clean, col_types):
    print("\n--- Generating Suite 1: Descriptive & Distribution Plots ---")
    num_cols = [c for c, t in col_types.items() if "numerical" in t and c in df_clean.columns and c != "id"]
    n = len(num_cols)
    ncols = 2
    nrows = (n + 1) // ncols

    # 1. Histograms + KDE + Normal Reference
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.8), dpi=200)
    fig.suptitle("Distribution of Numerical Features (Histogram + KDE + Normal Reference)", fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df_clean[col].dropna()
        ax.hist(data, bins=30, density=True, color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.7)
        # KDE
        kde = sp_stats.gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_grid, kde(x_grid), color="navy", lw=1.8, label="KDE")
        # Normal reference
        ax.plot(x_grid, sp_stats.norm.pdf(x_grid, data.mean(), data.std()), color="purple", linestyle="--", lw=1.6, label="Normal Ref")
        ax.axvline(data.mean(), color="red", linestyle="--", lw=1.2, label="Mean")
        ax.axvline(data.median(), color="orange", linestyle="-", lw=1.2, label="Median")
        ax.set_title(f"{col.upper()} | Skew={data.skew():.2f} Kurt={data.kurt():.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=9.5)
        ax.set_ylabel("Density", fontsize=9.5)
        ax.legend(fontsize=7.5, framealpha=0.8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "dist_histograms.png", dpi=200)

    # 2. Boxplots with Outliers
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.8), dpi=200)
    fig.suptitle("Box-Plots of Numerical Features (IQR & Outliers)", fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df_clean[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="#FADBD8", color="darkred"),
                    medianprops=dict(color="black", lw=1.5), flierprops=dict(marker="o", markerfacecolor="red", markersize=4, alpha=0.5))
        stats_txt = f"Min={data.min():.2f}  Max={data.max():.2f}\nQ1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}\nMean={data.mean():.2f}  Std={data.std():.2f}"
        ax.text(0.96, 0.96, stats_txt, transform=ax.transAxes, fontsize=7.5, va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_title(col.upper(), fontsize=11, fontweight="bold")
        ax.set_ylabel(col, fontsize=9.5)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "dist_boxplots.png", dpi=200)

    # 3. Correlation Heatmap
    corr = df_clean[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=200)
    cax = ax.imshow(np.ma.masked_array(corr, mask), cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, shrink=0.8, label="Pearson Correlation (r)")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels([c.upper() for c in num_cols], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([c.upper() for c in num_cols], fontsize=9)
    for row in range(len(num_cols)):
        for col in range(len(num_cols)):
            if not mask[row, col]:
                ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", color="black" if abs(corr.iloc[row, col]) < 0.6 else "white", fontsize=8.5)
    ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, "dist_correlation_heatmap.png", dpi=200)

    # 4. Skewness & Kurtosis
    skews = df_clean[num_cols].skew().sort_values()
    kurts = df_clean[num_cols].kurt().sort_values()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    fig.suptitle("Skewness & Kurtosis Distribution Summary", fontsize=13, fontweight="bold")
    ax1.barh([c.upper() for c in skews.index], skews.values, color=["#E74C3C" if v > 0 else "#3498DB" for v in skews], edgecolor="white")
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title("Skewness (Red = Right-skewed)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Skewness", fontsize=9.5)
    for v, patch in zip(skews.values, ax1.patches):
        offset = 0.05 if v >= 0 else -0.05
        ax1.text(v + offset, patch.get_y() + patch.get_height() / 2, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)

    ax2.barh([c.upper() for c in kurts.index], kurts.values, color=["#E74C3C" if v > 3 else "#3498DB" for v in kurts], edgecolor="white")
    ax2.axvline(3, color="black", lw=0.8, linestyle="--", label="Normal = 3")
    ax2.set_title("Kurtosis (Red = Leptokurtic / Heavy-tailed)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Kurtosis", fontsize=9.5)
    ax2.legend(fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, "dist_skew_kurtosis.png", dpi=200)


# ======================================================================
# SUITE 2: REPORT ECONOMETRIC VISUALS
# ======================================================================
def generate_report_visuals(df_full, X_combined, m_combined_hc3):
    print("\n--- Generating Suite 2: Report Econometric Visuals ---")
    params = m_combined_hc3.params
    cov_hc3 = m_combined_hc3.cov_params()
    feature_names = list(X_combined.columns)

    mean_edu = df_full["edu"].mean()
    mean_exp = df_full["exper"].mean()
    mean_ten = df_full["ten"].mean()
    mean_mom = df_full["momed"].mean()
    mean_dad = df_full["daded"].mean()
    mean_dep = df_full["dep"].mean()

    def build_vector(edu_val, exp_val, ten_val, female=0, married=1, metro=1,
                     nonprev=0, momed_val=mean_mom, daded_val=mean_dad, dep_val=mean_dep,
                     itai=0, itclient=0, ittech=0, itrd=0, wrkhm=0):
        vec = pd.Series(0.0, index=feature_names)
        vec["const"] = 1.0
        ed_c, ex_c, tn_c = edu_val - mean_edu, exp_val - mean_exp, ten_val - mean_ten
        mo_c, da_c, dp_c = momed_val - mean_mom, daded_val - mean_dad, dep_val - mean_dep
        vec["edu_c"] = ed_c
        vec["exper_c"] = ex_c
        vec["ten_c"] = tn_c
        vec["female"] = female
        vec["married"] = married
        vec["dep_c"] = dp_c
        vec["metro"] = metro
        vec["nonprev"] = nonprev
        vec["momed_c"] = mo_c
        vec["daded_c"] = da_c
        vec["itai"] = itai
        vec["itclient"] = itclient
        vec["ittech"] = ittech
        vec["itrd"] = itrd
        vec["wrkhm"] = wrkhm
        vec["edu_c_sq"] = ed_c ** 2
        vec["exper_c_sq"] = ex_c ** 2
        vec["ten_c_sq"] = tn_c ** 2
        vec["int_edu_female"] = ed_c * female
        vec["int_edu_momed"] = ed_c * mo_c
        vec["int_edu_daded"] = ed_c * da_c
        vec["int_exper_nonprev"] = ex_c * nonprev
        vec["int_edu_nonprev"] = ed_c * nonprev
        vec["int_dep_wrkhm"] = dp_c * wrkhm
        vec["int_female_wrkhm"] = female * wrkhm
        vec["int_edu_metro"] = ed_c * metro
        return vec

    # PLOT 1: SALARY VS LOG-SALARY HISTOGRAMS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)
    sal = df_full["sal"]
    log_sal = df_full["log_sal"]
    ax1.hist(sal, bins=32, density=True, color=STEEL_BLUE, edgecolor="white", linewidth=0.6, alpha=0.8, label="Empirical Density")
    sal_mu, sal_std = sal.mean(), sal.std()
    x_g1 = np.linspace(sal.min(), sal.max(), 300)
    ax1.plot(x_g1, sp_stats.norm.pdf(x_g1, sal_mu, sal_std), color=CORAL, linestyle="--", linewidth=2.0, label="Normal Reference")
    ax1.axvline(sal_mu, color=CORAL, linestyle="-", linewidth=1.8, label=f"Mean: ${sal_mu:.2f}/hr")
    ax1.axvline(sal.median(), color=AMBER, linestyle="-", linewidth=1.8, label=f"Median: ${sal.median():.2f}/hr")
    ax1.set_title("A. Untransformed Hourly Salary Distribution", fontsize=12.5, fontweight="bold", pad=10, color=SLATE)
    ax1.set_xlabel("Hourly Salary ($/hr)", fontsize=10.5, labelpad=6)
    ax1.set_ylabel("Probability Density", fontsize=10.5, labelpad=6)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.0f"))
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.8)
    stats_raw = f"$\\bf{{Sample\\ Characteristics:}}$\n• N = {len(sal)} employees\n• Skewness = +{sal.skew():.2f} (Severe Right Skew)\n• Excess Kurtosis = +{sal.kurtosis():.2f}\n• Range: ${sal.min():.2f} – ${sal.max():.2f}/hr"
    ax1.text(0.38, 0.58, stats_raw, transform=ax1.transAxes, fontsize=8.8, bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9F0", edgecolor=AMBER, alpha=0.95))

    ax2.hist(log_sal, bins=32, density=True, color=TEAL, edgecolor="white", linewidth=0.6, alpha=0.8, label="Empirical Density")
    log_mu, log_std = log_sal.mean(), log_sal.std()
    x_g2 = np.linspace(log_sal.min(), log_sal.max(), 300)
    ax2.plot(x_g2, sp_stats.norm.pdf(x_g2, log_mu, log_std), color=CORAL, linestyle="--", linewidth=2.0, label="Normal Reference")
    ax2.axvline(log_mu, color=CORAL, linestyle="-", linewidth=1.8, label=f"Mean: {log_mu:.2f}")
    ax2.axvline(log_sal.median(), color=AMBER, linestyle="-", linewidth=1.8, label=f"Median: {log_sal.median():.2f}")
    ax2.set_title(r"B. Log-Transformed Salary Distribution ($\ln(\mathrm{Salary})$)", fontsize=12.5, fontweight="bold", pad=10, color=SLATE)
    ax2.set_xlabel(r"Log Hourly Salary ($\ln(\mathrm{Salary})$)", fontsize=10.5, labelpad=6)
    ax2.set_ylabel("Probability Density", fontsize=10.5, labelpad=6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.8)
    stats_log = f"$\\bf{{Post-Transformation:}}$\n• N = {len(log_sal)} employees\n• Skewness = +{log_sal.skew():.2f} (Stabilized)\n• Excess Kurtosis = +{log_sal.kurtosis():.2f}\n• Mean $\\approx$ Median Symmetry Achieved"
    ax2.text(0.04, 0.58, stats_log, transform=ax2.transAxes, fontsize=8.8, bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F9F8", edgecolor=TEAL, alpha=0.95))
    fig.suptitle("Impact of Semi-Logarithmic Wage Transformation on Distributional Form (N = 523)", fontsize=14, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "plot1_salary_vs_log_salary_histograms.png")

    # PLOT 2: ADJUSTED HUMAN CAPITAL NONLINEAR CURVES WITH 95% CIs
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.4), dpi=300)
    base_exp, base_ten, base_edu = mean_exp, mean_ten, mean_edu

    # Education curve
    edu_grid = np.linspace(8, 20, 100)
    p_sal_edu, ci_l_edu, ci_u_edu = [], [], []
    for e in edu_grid:
        v = build_vector(edu_val=e, exp_val=base_exp, ten_val=base_ten)
        y_hat = np.dot(v, params)
        se = np.sqrt(v.T @ cov_hc3 @ v)
        p_sal_edu.append(np.exp(y_hat))
        ci_l_edu.append(np.exp(y_hat - 1.96 * se))
        ci_u_edu.append(np.exp(y_hat + 1.96 * se))
    p_sal_edu, ci_l_edu, ci_u_edu = np.array(p_sal_edu), np.array(ci_l_edu), np.array(ci_u_edu)
    ax1.plot(edu_grid, p_sal_edu, color=NAVY, linewidth=2.4, label="Predicted Hourly Salary")
    ax1.fill_between(edu_grid, ci_l_edu, ci_u_edu, color=NAVY, alpha=0.15, label="95% HC3 Robust CI")
    ax1.axvline(12, color=GRAY, linestyle=":", linewidth=1.2)
    ax1.axvline(16, color=TEAL, linestyle="--", linewidth=1.2)
    ax1.axvline(18, color=CORAL, linestyle="--", linewidth=1.2)
    ax1.text(12, p_sal_edu.min() + 0.4, "High School\n(12 yrs)", fontsize=8, ha="center", color=GRAY)
    ax1.text(16, p_sal_edu.min() + 0.9, "Bachelor's\n(16 yrs)", fontsize=8, ha="center", color=TEAL, fontweight="bold")
    ax1.text(18, p_sal_edu.min() + 1.6, "Master's\n(18 yrs)", fontsize=8, ha="center", color=CORAL, fontweight="bold")
    ax1.set_title("A. Education: Convex 'Degree Surge'", fontsize=12, fontweight="bold", pad=9, color=SLATE)
    ax1.set_xlabel("Years of Formal Education", fontsize=10, labelpad=5)
    ax1.set_ylabel("Adjusted Hourly Salary ($/hr)", fontsize=10, labelpad=5)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.1f"))
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.2)

    # Experience curve
    exp_grid = np.linspace(0, 42, 100)
    p_sal_exp, ci_l_exp, ci_u_exp = [], [], []
    for x in exp_grid:
        v = build_vector(edu_val=base_edu, exp_val=x, ten_val=base_ten)
        y_hat = np.dot(v, params)
        se = np.sqrt(v.T @ cov_hc3 @ v)
        p_sal_exp.append(np.exp(y_hat))
        ci_l_exp.append(np.exp(y_hat - 1.96 * se))
        ci_u_exp.append(np.exp(y_hat + 1.96 * se))
    p_sal_exp, ci_l_exp, ci_u_exp = np.array(p_sal_exp), np.array(ci_l_exp), np.array(ci_u_exp)
    ax2.plot(exp_grid, p_sal_exp, color=TEAL, linewidth=2.4, label="Predicted Hourly Salary")
    ax2.fill_between(exp_grid, ci_l_exp, ci_u_exp, color=TEAL, alpha=0.15, label="95% HC3 Robust CI")
    pk_idx = np.argmax(p_sal_exp)
    ax2.scatter(exp_grid[pk_idx], p_sal_exp[pk_idx], color=CORAL, s=55, zorder=5, edgecolor="black")
    ax2.annotate(f"Peak Earning Point\n~{exp_grid[pk_idx]:.0f} Yrs Exp (${p_sal_exp[pk_idx]:.2f}/hr)",
                 xy=(exp_grid[pk_idx], p_sal_exp[pk_idx]), xytext=(exp_grid[pk_idx] - 12, p_sal_exp[pk_idx] - 1.1),
                 arrowprops=dict(facecolor=CORAL, shrink=0.08, width=1.2, headwidth=5),
                 fontsize=8.5, fontweight="bold", color=CORAL, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=CORAL, alpha=0.9))
    ax2.set_title("B. Workforce Experience: Mid-Career Peak", fontsize=12, fontweight="bold", pad=9, color=SLATE)
    ax2.set_xlabel("Total Work Experience (Years)", fontsize=10, labelpad=5)
    ax2.set_ylabel("Adjusted Hourly Salary ($/hr)", fontsize=10, labelpad=5)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.1f"))
    ax2.legend(loc="lower center", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.2)

    # Tenure curve
    ten_grid = np.linspace(0, 32, 100)
    p_sal_ten, ci_l_ten, ci_u_ten = [], [], []
    for t in ten_grid:
        v = build_vector(edu_val=base_edu, exp_val=base_exp, ten_val=t)
        y_hat = np.dot(v, params)
        se = np.sqrt(v.T @ cov_hc3 @ v)
        p_sal_ten.append(np.exp(y_hat))
        ci_l_ten.append(np.exp(y_hat - 1.96 * se))
        ci_u_ten.append(np.exp(y_hat + 1.96 * se))
    p_sal_ten, ci_l_ten, ci_u_ten = np.array(p_sal_ten), np.array(ci_l_ten), np.array(ci_u_ten)
    ax3.plot(ten_grid, p_sal_ten, color=SLATE, linewidth=2.4, label="Predicted Hourly Salary")
    ax3.fill_between(ten_grid, ci_l_ten, ci_u_ten, color=SLATE, alpha=0.15, label="95% HC3 Robust CI")
    plat_idx = np.argmax(p_sal_ten)
    ax3.scatter(ten_grid[plat_idx], p_sal_ten[plat_idx], color=AMBER, s=55, zorder=5, edgecolor="black")
    ax3.annotate(f"Loyalty Ceiling / Plateau\n~{ten_grid[plat_idx]:.0f} Yrs Tenure (${p_sal_ten[plat_idx]:.2f}/hr)",
                 xy=(ten_grid[plat_idx], p_sal_ten[plat_idx]), xytext=(ten_grid[plat_idx] - 14, p_sal_ten[plat_idx] - 1.0),
                 arrowprops=dict(facecolor=AMBER, shrink=0.08, width=1.2, headwidth=5),
                 fontsize=8.5, fontweight="bold", color="#935116", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=AMBER, alpha=0.9))
    ax3.set_title("C. Company Tenure: Seniority Loyalty Ceiling", fontsize=12, fontweight="bold", pad=9, color=SLATE)
    ax3.set_xlabel("Company Tenure (Years at Current Firm)", fontsize=10, labelpad=5)
    ax3.set_ylabel("Adjusted Hourly Salary ($/hr)", fontsize=10, labelpad=5)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.1f"))
    ax3.legend(loc="lower center", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.2)

    fig.suptitle("Econometric Human Capital Trajectories: Non-Linear Returns with 95% HC3 CIs", fontsize=14, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "plot2_human_capital_nonlinear_curves.png")

    # PLOT 3: SALARY-DIFFERENCE FOREST PLOT
    def calc_contrast_stats(v1, v0, comp, ref):
        diff = v1 - v0
        delta = np.dot(diff, params)
        se = np.sqrt(diff.T @ cov_hc3 @ diff)
        return {
            "Comparison": comp, "Reference": ref,
            "Pct_Change": 100 * (np.exp(delta) - 1),
            "Pct_Low": 100 * (np.exp(delta - 1.96 * se) - 1),
            "Pct_Upp": 100 * (np.exp(delta + 1.96 * se) - 1)
        }

    c_data = [
        calc_contrast_stats(build_vector(base_edu, base_exp, base_ten, female=1, wrkhm=0),
                            build_vector(base_edu, base_exp, base_ten, female=0, wrkhm=0),
                            "Gender: Female vs. Male (Office)", "Ref: Office-based Male"),
        calc_contrast_stats(build_vector(base_edu, base_exp, base_ten, female=1, wrkhm=1),
                            build_vector(base_edu, base_exp, base_ten, female=0, wrkhm=1),
                            "Gender: Female vs. Male (Remote)", "Ref: Remote-based Male"),
        calc_contrast_stats(build_vector(base_edu, base_exp, base_ten, metro=1),
                            build_vector(base_edu, base_exp, base_ten, metro=0),
                            "Location: Metro vs. Non-Metro", "Ref: Non-Metropolitan Area"),
        calc_contrast_stats(build_vector(base_edu, base_exp, base_ten, female=0, wrkhm=1),
                            build_vector(base_edu, base_exp, base_ten, female=0, wrkhm=0),
                            "Remote Work: Remote vs. Office (Male)", "Ref: Office-based Male"),
        calc_contrast_stats(build_vector(base_edu, base_exp, base_ten, female=1, wrkhm=1),
                            build_vector(base_edu, base_exp, base_ten, female=1, wrkhm=0),
                            "Remote Work: Remote vs. Office (Female)", "Ref: Office-based Female"),
        calc_contrast_stats(build_vector(base_edu, base_exp, base_ten, itai=1),
                            build_vector(base_edu, base_exp, base_ten, itai=0),
                            "AI Work: AI Applications vs. Core IT", "Ref: Non-AI Core IT Staff"),
        calc_contrast_stats(build_vector(base_edu, base_exp, base_ten, itclient=1),
                            build_vector(base_edu, base_exp, base_ten, itclient=0),
                            "Client Role: IT Client-Only vs. Internal IT", "Ref: Internal Delivery Staff"),
    ]
    c_df = pd.DataFrame(c_data)

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
    y_pos = np.arange(len(c_df))[::-1]
    bar_colors = [GREEN if p > 0 else CORAL for p in c_df["Pct_Change"]]
    ax.axvline(0, color=SLATE, linestyle="--", linewidth=1.3, zorder=2, label="Parity (0% Salary Difference)")

    for y, row, col in zip(y_pos, c_df.itertuples(), bar_colors):
        ax.plot([row.Pct_Low, row.Pct_Upp], [y, y], color=col, linewidth=2.4, zorder=3)
        ax.scatter(row.Pct_Change, y, color=col, edgecolor="black", linewidth=1.0, s=80, zorder=4)
        ax.text(row.Pct_Upp + 1.8, y, f"{row.Pct_Change:+.1f}%  [{row.Pct_Low:+.1f}%, {row.Pct_Upp:+.1f}%]",
                va="center", fontsize=9.2, fontweight="bold", color=col)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r.Comparison}\n({r.Reference})" for r in c_df.itertuples()], fontsize=9.5, fontweight="semibold", color=SLATE)
    ax.set_title("Adjusted Salary Differentials Relative to Reference Conditions\nControlled Ceteris Paribus Estimates with 95% HC3 Confidence Intervals",
                 fontsize=13, fontweight="bold", pad=12, color=NAVY)
    ax.set_xlabel("Adjusted Salary Differential (% Difference in Geometric Median Wage)", fontsize=10.5, labelpad=7, color=SLATE)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%+.0f%%"))
    ax.set_xlim(-45, 45)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=9)
    plt.tight_layout()
    save_fig(fig, "plot3_salary_differences_forest_plot.png")


# ======================================================================
# SUITE 3: FOUR-PANEL REGRESSION DIAGNOSTIC FIGURE
# ======================================================================
def generate_four_panel_diagnostics(df_full, X_combined, m_combined_ols):
    print("\n--- Generating Suite 3: Four-Panel Regression Diagnostic Suite ---")
    n_obs = int(m_combined_ols.nobs)
    p_params = int(X_combined.shape[1])
    ordinary_residuals = m_combined_ols.resid
    fitted_vals = m_combined_ols.fittedvalues

    influence = m_combined_ols.get_influence()
    cooks_d, _ = influence.cooks_distance
    leverage = influence.hat_matrix_diag
    studentized_resid = influence.resid_studentized_internal
    sqrt_abs_studentized = np.sqrt(np.abs(studentized_resid))
    leverage_thresh = 2.0 * p_params / n_obs

    top4_order = np.argsort(cooks_d)[::-1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), dpi=300)
    fig.patch.set_facecolor("white")
    P_COL = "#2B5B84"
    LOW_COL = "#C0392B"

    # Panel 1: Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(fitted_vals, ordinary_residuals, color=P_COL, edgecolors="white", linewidths=0.4, s=32, alpha=0.72, label="Observations")
    ax1.axhline(0, color="#555555", linestyle="--", linewidth=1.1, label="Zero Reference ($e_i = 0$)")
    low1 = sm.nonparametric.lowess(ordinary_residuals, fitted_vals, frac=0.66)
    ax1.plot(low1[:, 0], low1[:, 1], color=LOW_COL, linewidth=2.0, label="LOWESS Fit")
    ax1.set_title("Residuals vs Fitted", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel(r"Fitted Values ($\widehat{\ln(\mathrm{Salary})}$)", fontsize=10, labelpad=6)
    ax1.set_ylabel("Ordinary Residuals ($e_i$)", fontsize=10, labelpad=6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.2)

    # Panel 2: Normal Q–Q
    ax2 = axes[0, 1]
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(studentized_resid, dist="norm")
    ax2.scatter(osm, osr, color=P_COL, edgecolors="white", linewidths=0.4, s=32, alpha=0.72, label=f"Studentized Residuals (N = {n_obs})")
    ref_xs = np.linspace(osm.min() - 0.25, osm.max() + 0.25, 100)
    ax2.plot(ref_xs, intercept + slope * ref_xs, color=LOW_COL, linestyle="--", linewidth=1.5, label="Normal Reference")
    ax2.set_title("Normal Q–Q", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Theoretical Quantiles", fontsize=10, labelpad=6)
    ax2.set_ylabel("Internally Studentized Residuals ($r_i^*$)", fontsize=10, labelpad=6)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.2)

    # Panel 3: Scale–Location
    ax3 = axes[1, 0]
    ax3.scatter(fitted_vals, sqrt_abs_studentized, color=P_COL, edgecolors="white", linewidths=0.4, s=32, alpha=0.72, label="Observations")
    low3 = sm.nonparametric.lowess(sqrt_abs_studentized, fitted_vals, frac=0.66)
    ax3.plot(low3[:, 0], low3[:, 1], color=LOW_COL, linewidth=2.0, label="LOWESS Fit")
    ax3.set_title("Scale–Location", fontsize=12, fontweight="bold", pad=8)
    ax3.set_xlabel(r"Fitted Values ($\widehat{\ln(\mathrm{Salary})}$)", fontsize=10, labelpad=6)
    ax3.set_ylabel(r"$\sqrt{|\mathrm{Internally\ Studentized\ Residuals}|}$", fontsize=10, labelpad=6)
    ax3.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.2)

    # Panel 4: Residuals vs Leverage
    ax4 = axes[1, 1]
    ax4.scatter(leverage, studentized_resid, color=P_COL, edgecolors="white", linewidths=0.4, s=32, alpha=0.72, label="Observations")
    ax4.axhline(0, color="#555555", linestyle=":", linewidth=0.8)
    ax4.axhline(3.0, color="#C0392B", linestyle="--", linewidth=1.1, label=r"Residual Boundary ($\pm 3$)")
    ax4.axhline(-3.0, color="#C0392B", linestyle="--", linewidth=1.1)
    ax4.axvline(leverage_thresh, color="#7F8C8D", linestyle="--", linewidth=1.2, label=f"Leverage Threshold ($2p/N = {leverage_thresh:.3f}$)")
    ax4.scatter(leverage[top4_order], studentized_resid[top4_order], color="#E67E22", edgecolors="black", linewidths=0.9, s=64, zorder=5, label="Top 4 Cook's D Cases")

    offsets = {443: (-0.024, 0.22), 128: (0.010, -0.44), 498: (0.010, 0.18), 410: (0.010, 0.18)}
    for idx in top4_order:
        emp_id = int(df_full.loc[idx, "id"])
        x_v, y_v = leverage[idx], studentized_resid[idx]
        dx, dy = offsets.get(emp_id, (0.008, 0.18))
        ax4.annotate(f"ID {emp_id}", xy=(x_v, y_v), xytext=(x_v + dx, y_v + dy),
                     fontsize=8.5, fontweight="bold", color="#7E3D0A",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9E7", edgecolor="#FAD7A0", alpha=0.92))

    ax4.set_title("Residuals vs Leverage", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel(r"Leverage ($h_{ii}$)", fontsize=10, labelpad=6)
    ax4.set_ylabel("Internally Studentized Residuals ($r_i^*$)", fontsize=10, labelpad=6)
    ax4.set_xlim(-0.01, max(leverage.max() * 1.06, 0.36))
    ax4.set_ylim(-4.8, 7.2)
    ax4.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D0D7DE", fontsize=8.0)

    plt.tight_layout(pad=2.2)
    save_fig(fig, "regression_diagnostics_four_panel.png")


# ======================================================================
# SUITE 4: CLIENT PRESENTATION & EXECUTIVE DASHBOARD
# ======================================================================
def generate_client_plots(df):
    print("\n--- Generating Suite 4: Client Presentation Suite ---")
    formula = (
        "ln_sal ~ female + married + edu + edu2 + exper + exp2 + ten + ten2 + "
        "wrkhm + itprog + itarch + itai + itqual + itclient + itrd + metro"
    )
    model = ols(formula, data=df).fit(cov_type='HC3')

    baseline = {
        'female': 0, 'married': 1, 'edu': 16, 'edu2': 256,
        'exper': 15, 'exp2': 225, 'ten': 5, 'ten2': 25,
        'wrkhm': 0, 'itprog': 1, 'itarch': 0, 'itai': 0,
        'itqual': 0, 'itclient': 0, 'itrd': 0, 'metro': 1
    }

    # Chart 1: Salary Distribution & Boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    male_sal = df.loc[df['female'] == 0, 'sal']
    fem_sal = df.loc[df['female'] == 1, 'sal']
    ax1.hist(male_sal, bins=25, alpha=0.6, color=NAVY, label=f'Men (Med=${male_sal.median():.2f}/hr)', density=True)
    ax1.hist(fem_sal, bins=25, alpha=0.6, color=CORAL, label=f'Women (Med=${fem_sal.median():.2f}/hr)', density=True)
    ax1.axvline(male_sal.median(), color=NAVY, linestyle='--', linewidth=2)
    ax1.axvline(fem_sal.median(), color=CORAL, linestyle='--', linewidth=2)
    ax1.set_title("A. Wage Distribution by Gender", fontsize=13, fontweight='bold', color=SLATE)
    ax1.set_xlabel("Hourly Salary ($/hr)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%.1f'))

    df['edu_tier'] = pd.cut(df['edu'], bins=[0, 12, 16, 30], labels=['High School (<=12y)', "Bachelor's (13-16y)", "Master's / Ph.D. (>16y)"])
    tier_data = [df.loc[df['edu_tier'] == t, 'sal'].dropna() for t in df['edu_tier'].cat.categories]
    bp = ax2.boxplot(tier_data, patch_artist=True, boxprops=dict(facecolor=TEAL, alpha=0.6), medianprops=dict(color=CORAL, linewidth=2.5))
    medians = [np.median(d) for d in tier_data]
    for i, med in enumerate(medians):
        ax2.text(i + 1, med + 0.35, f"Med: ${med:.2f}", ha='center', fontweight='bold', color=CORAL, fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=CORAL, alpha=0.8))
    ax2.set_xticklabels(df['edu_tier'].cat.categories, fontsize=10, fontweight='semibold')
    ax2.set_title("B. Salary Progression by Educational Tier", fontsize=13, fontweight='bold', color=SLATE)
    ax2.set_ylabel("Hourly Salary ($/hr)", fontsize=11)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.1f'))
    fig.suptitle("Client Briefing: Base Compensation Structure & Credential Value", fontsize=15, fontweight='bold', color=NAVY, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "client_chart1_salary_distribution.png")

    # Chart 2: Human Capital Trajectories
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    edu_range = np.linspace(8, 20, 100)
    pred_edu_sal = np.exp(model.predict(pd.DataFrame([dict(baseline, edu=e, edu2=e**2) for e in edu_range])))
    ax1.plot(edu_range, pred_edu_sal, color=NAVY, linewidth=2.8, label='Model Trajectory')
    ax1.fill_between(edu_range, pred_edu_sal * 0.92, pred_edu_sal * 1.08, color=NAVY, alpha=0.12, label='95% Conf. Band')
    ax1.set_title("A. Education Return: Convex 'Degree Surge'", fontsize=12, fontweight='bold', color=SLATE)
    ax1.set_xlabel("Education (Years Completed)", fontsize=10)
    ax1.set_ylabel("Predicted Hourly Wage ($/hr)", fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.1f'))

    exp_range = np.linspace(0, 40, 100)
    pred_exp_sal = np.exp(model.predict(pd.DataFrame([dict(baseline, exper=x, exp2=x**2) for x in exp_range])))
    peak_exp = exp_range[np.argmax(pred_exp_sal)]
    max_exp_sal = np.max(pred_exp_sal)
    ax2.plot(exp_range, pred_exp_sal, color=TEAL, linewidth=2.8)
    ax2.plot(peak_exp, max_exp_sal, 'o', color=CORAL, markersize=8)
    ax2.annotate(f"Peak: ~{peak_exp:.0f}y (${max_exp_sal:.2f}/hr)", xy=(peak_exp, max_exp_sal), xytext=(peak_exp - 8, max_exp_sal - 1.2),
                 arrowprops=dict(facecolor=CORAL, shrink=0.08, width=1.5, headwidth=6), fontweight='bold', color=CORAL, bbox=dict(boxstyle="round", facecolor="white", edgecolor=CORAL))
    ax2.set_title("B. Workforce Experience: Mid-Career Peak", fontsize=12, fontweight='bold', color=SLATE)
    ax2.set_xlabel("Total Work Experience (Years)", fontsize=10)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.1f'))

    ten_range = np.linspace(0, 30, 100)
    pred_ten_sal = np.exp(model.predict(pd.DataFrame([dict(baseline, ten=t, ten2=t**2) for t in ten_range])))
    peak_ten = ten_range[np.argmax(pred_ten_sal)]
    max_ten_sal = np.max(pred_ten_sal)
    ax3.plot(ten_range, pred_ten_sal, color=SLATE, linewidth=2.8)
    ax3.plot(peak_ten, max_ten_sal, 'o', color=AMBER, markersize=8)
    ax3.annotate(f"Plateau: ~{peak_ten:.0f}y (${max_ten_sal:.2f}/hr)", xy=(peak_ten, max_ten_sal), xytext=(peak_ten - 10, max_ten_sal - 0.9),
                 arrowprops=dict(facecolor=AMBER, shrink=0.08, width=1.5, headwidth=6), fontweight='bold', color='#B8860B', bbox=dict(boxstyle="round", facecolor="white", edgecolor=AMBER))
    ax3.set_title("C. Firm Tenure: Seniority Loyalty Ceiling", fontsize=12, fontweight='bold', color=SLATE)
    ax3.set_xlabel("Company Tenure (Years)", fontsize=10)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.1f'))
    fig.suptitle("Career Compensation Drivers: The Non-Linear Realities of Skill & Seniority", fontsize=15, fontweight='bold', color=NAVY, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "client_chart2_human_capital_curves.png")

    # Chart 3: Gender Gap & Remote
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    raw_male = df.loc[df['female'] == 0, 'sal'].mean()
    raw_fem = df.loc[df['female'] == 1, 'sal'].mean()
    raw_pct_gap = 100 * (raw_fem - raw_male) / raw_male
    coef_fem = model.params['female']
    ci_fem = model.conf_int().loc['female']
    ctrl_pct_gap = 100 * (np.exp(coef_fem) - 1)
    ci_l = 100 * (np.exp(ci_fem[0]) - 1)
    ci_u = 100 * (np.exp(ci_fem[1]) - 1)

    ax1.bar(['Raw Disparity\n(Unadjusted Sample)', 'Controlled Disparity\n(Ceteris Paribus Model)'],
            [raw_pct_gap, ctrl_pct_gap], color=[AMBER, CORAL], width=0.45, edgecolor=SLATE, linewidth=1.2)
    ax1.errorbar([1], [ctrl_pct_gap], yerr=[[ctrl_pct_gap - ci_l], [ci_u - ctrl_pct_gap]], fmt='none', ecolor=SLATE, elinewidth=2, capsize=6)
    ax1.axhline(0, color=SLATE, linewidth=0.8, linestyle='--')
    ax1.set_ylabel("Pay Disparity vs. Equivalent Male (%)", fontsize=11)
    ax1.set_title("A. Gender Pay Disparity: Raw vs. Econometrically Controlled", fontsize=12, fontweight='bold', color=SLATE)

    categories = ['In-Office Male', 'In-Office Female', 'WFH Male', 'WFH Female']
    scenarios = [dict(baseline, female=0, wrkhm=0), dict(baseline, female=1, wrkhm=0), dict(baseline, female=0, wrkhm=1), dict(baseline, female=1, wrkhm=1)]
    pred_scenarios = [np.exp(model.predict(pd.DataFrame([s]))[0]) for s in scenarios]
    ax2.bar(categories, pred_scenarios, color=[NAVY, CORAL, '#4A90E2', '#E2847A'], width=0.5, edgecolor=SLATE, linewidth=1.2)
    for b in ax2.patches:
        ax2.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.15, f"${b.get_height():.2f}/hr", ha='center', fontweight='bold', color=SLATE, fontsize=10)
    ax2.set_title("B. Predicted Hourly Wages Across Gender & Work-From-Home", fontsize=12, fontweight='bold', color=SLATE)
    ax2.set_ylabel("Predicted Hourly Wage ($/hr)", fontsize=11)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.1f'))
    fig.suptitle("Equity & Workplace Flexibility Audit: Quantifying Underlying Pay Gaps", fontsize=15, fontweight='bold', color=NAVY, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "client_chart3_gender_gap_and_remote.png")

    # Chart 4: Specialization Premiums
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    roles = ['itai', 'itarch', 'itrd', 'itclient', 'itprog', 'itqual', 'metro']
    labels = ['AI / ML (itai)', 'Architecture (itarch)', 'R&D (itrd)', 'Client Solutions (itclient)', 'Programming (itprog)', 'Quality Assurance (itqual)', 'Metropolitan (metro)']
    prems = [100 * (np.exp(model.params[r]) - 1) for r in roles]
    sort_idx = np.argsort(prems)
    ax.barh(np.arange(len(roles)), [prems[i] for i in sort_idx], color=[CORAL if prems[i] < 0 else TEAL for i in sort_idx], edgecolor=SLATE, height=0.55)
    for idx, i in enumerate(sort_idx):
        p = prems[i]
        ax.text(p + (1.2 if p >= 0 else -1.2), idx, f"{p:+.1f}%", va='center', ha='left' if p >= 0 else 'right', fontweight='bold', fontsize=10, color=SLATE)
    ax.set_yticks(np.arange(len(roles)))
    ax.set_yticklabels([labels[i] for i in sort_idx], fontsize=10, fontweight='semibold')
    ax.axvline(0, color=SLATE, linestyle='--', linewidth=1)
    ax.set_title("Benchmarking Technology Specializations & Geographic Cost Differentials", fontsize=14, fontweight='bold', color=NAVY)
    ax.set_xlabel("Market Compensation Premium vs. Baseline Reference Role (%)", fontsize=11)
    plt.tight_layout()
    save_fig(fig, "client_chart4_specialization_premiums.png")

    # Chart 5: Executive Dashboard
    fig = plt.figure(figsize=(18, 11), facecolor='#F8F9FA', dpi=300)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)
    fig.text(0.08, 0.96, "IT WORKFORCE COMPENSATION & LABOR ECONOMETRICS DASHBOARD", fontsize=18, fontweight='bold', color=NAVY)
    fig.text(0.08, 0.935, "Executive Insights for Workforce Strategy, Fair Pay Compliance, and Compensation Benchmarking (N = 523 IT Professionals)", fontsize=11, color=GRAY)

    metrics = [
        ("MEDIAN HOURLY PAY", f"${df['sal'].median():.2f}/hr", "Cross-firm baseline benchmark", TEAL),
        ("CONTROLLED GENDER GAP", f"{ctrl_pct_gap:.1f}%", "Unexplained disparity (p < 0.001)", CORAL),
        ("PEAK CAREER EXP", f"{peak_exp:.0f} Years", "Maximum experience return point", NAVY),
        ("LOYALTY PLATEAU", f"{peak_ten:.0f} Years", "Tenure seniority growth ceiling", AMBER)
    ]
    for i, (title, val, sub, col) in enumerate(metrics):
        rect_x = 0.08 + i * 0.215
        ax_card = fig.add_axes([rect_x, 0.82, 0.195, 0.09], facecolor='white')
        ax_card.set_xticks([])
        ax_card.set_yticks([])
        ax_card.text(0.08, 0.72, title, fontsize=8.5, fontweight='bold', color=GRAY)
        ax_card.text(0.08, 0.32, val, fontsize=16, fontweight='bold', color=col)
        ax_card.text(0.08, 0.10, sub, fontsize=7.5, color=SLATE, fontstyle='italic')

    # Subplots on dashboard
    ax_d1 = fig.add_subplot(gs[1, 0])
    ax_d1.plot(edu_range, pred_edu_sal, color=NAVY, linewidth=2.5)
    ax_d1.set_title("1. Education Credential Curve", fontsize=11, fontweight='bold')

    ax_d2 = fig.add_subplot(gs[1, 1])
    ax_d2.plot(exp_range, pred_exp_sal, color=TEAL, linewidth=2.5)
    ax_d2.set_title("2. Experience Curve (Peak ~26y)", fontsize=11, fontweight='bold')

    ax_d3 = fig.add_subplot(gs[1, 2])
    ax_d3.plot(ten_range, pred_ten_sal, color=SLATE, linewidth=2.5)
    ax_d3.set_title("3. Tenure Plateau (~24y)", fontsize=11, fontweight='bold')

    ax_d4 = fig.add_subplot(gs[2, :2])
    ax_d4.bar(categories, pred_scenarios, color=[NAVY, CORAL, '#4A90E2', '#E2847A'], width=0.45, edgecolor=SLATE)
    for b in ax_d4.patches:
        ax_d4.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.1, f"${b.get_height():.2f}/hr", ha='center', fontsize=9.5, fontweight='bold')
    ax_d4.set_title("4. Wage Equivalence Matrix: Gender by Workplace Mode", fontsize=11, fontweight='bold')

    ax_d5 = fig.add_subplot(gs[2, 2])
    top_roles = ['Metro', 'Programming', 'R&D', 'Architecture', 'AI / ML']
    top_p = [prems[6], prems[4], prems[2], prems[1], prems[0]]
    ax_d5.barh(top_roles, top_p, color=[TEAL if x >= 0 else CORAL for x in top_p], height=0.5, edgecolor=SLATE)
    ax_d5.axvline(0, color=SLATE, linestyle='--', linewidth=0.8)
    ax_d5.set_title("5. Role & Market Premiums", fontsize=11, fontweight='bold')

    save_fig(fig, "client_executive_dashboard.png")


# ======================================================================
# SUITE 5: STRATEGIC HR RECOMMENDATIONS VISUALS
# ======================================================================
def generate_recommendation_plots(df):
    print("\n--- Generating Suite 5: Strategic HR Recommendations Suite ---")
    df['exp_cohort'] = pd.cut(df['exper'], bins=[-1, 5, 15, 25, 50], labels=['Junior (0-5y)', 'Mid-Level (5-15y)', 'Senior (15-25y)', 'Principal (>25y)'])
    cohorts = ['Junior (0-5y)', 'Mid-Level (5-15y)', 'Senior (15-25y)', 'Principal (>25y)']
    male_meds = [df.loc[(df['exp_cohort'] == c) & (df['female'] == 0), 'sal'].median() for c in cohorts]
    fem_meds = [df.loc[(df['exp_cohort'] == c) & (df['female'] == 1), 'sal'].median() for c in cohorts]
    dollar_gaps = [m - f for m, f in zip(male_meds, fem_meds)]
    pct_gaps = [100 * (f - m) / m for m, f in zip(male_meds, fem_meds)]
    annual_gaps = [g * 2080 for g in dollar_gaps]

    # Rec Plot 1: Pay Equity Action Plan
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    x = np.arange(len(cohorts))
    w = 0.35
    ax1.bar(x - w/2, male_meds, width=w, label='Men Median', color=NAVY, edgecolor=SLATE)
    ax1.bar(x + w/2, fem_meds, width=w, label='Women Median', color=CORAL, edgecolor=SLATE)
    for i in range(len(cohorts)):
        ax1.text(x[i] - w/2, male_meds[i] + 0.15, f"${male_meds[i]:.2f}", ha='center', fontsize=9.5, fontweight='bold', color=NAVY)
        ax1.text(x[i] + w/2, fem_meds[i] + 0.15, f"${fem_meds[i]:.2f}", ha='center', fontsize=9.5, fontweight='bold', color=CORAL)
        ax1.annotate(f"Gap: -${dollar_gaps[i]:.2f}/hr\n({pct_gaps[i]:.1f}%)", xy=(x[i], (male_meds[i] + fem_meds[i]) / 2),
                     xytext=(x[i], max(male_meds[i], fem_meds[i]) + 1.2), ha='center', fontsize=8.5, fontweight='bold', color='#8B0000',
                     arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.2), bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEEEE", edgecolor=CORAL))
    ax1.set_xticks(x)
    ax1.set_xticklabels(cohorts, fontsize=10, fontweight='semibold')
    ax1.set_ylabel("Median Hourly Wage ($/hr)", fontsize=11)
    ax1.set_title("A. Median Wage Disparity Across Career Stages", fontsize=12, fontweight='bold', color=SLATE)
    ax1.legend(loc='upper left', frameon=True, facecolor='white')
    ax1.set_ylim(0, 11)

    fem_counts = [df.loc[(df['exp_cohort'] == c) & (df['female'] == 1), 'sal'].count() for c in cohorts]
    cohort_cost = [cnt * ann for cnt, ann in zip(fem_counts, annual_gaps)]
    tot_pool = sum(cohort_cost)
    b_shares = [100 * c / tot_pool for c in cohort_cost]
    bars_b = ax2.bar(cohorts, b_shares, color=[TEAL, NAVY, AMBER, PURPLE], width=0.5, edgecolor=SLATE, linewidth=1.2)
    for i, b in enumerate(bars_b):
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 1.2, f"{yval:.1f}%\nof Pool", ha='center', va='bottom', fontsize=10, fontweight='bold', color=SLATE)
        ax2.text(b.get_x() + b.get_width()/2.0, yval / 2, f"Annual Deficit:\n${annual_gaps[i]:,.0f}/worker", ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
    ax2.set_ylabel("Share of Equity Remediation Budget (%)", fontsize=11)
    ax2.set_title("B. Strategic Pay Remediation Budget Allocation", fontsize=12, fontweight='bold', color=SLATE)
    fig.suptitle("RECOMMENDATION 1: Targeted Pay Equity Action Plan & Budget Sizing", fontsize=15, fontweight='bold', color=NAVY, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "rec_plot1_pay_equity_action_plan.png")

    # Rec Plot 2: Loyalty Penalty vs Market
    m_exp_ten = ols('ln_sal ~ ten + ten2 + outside_exp + outside_exp2 + edu + female + metro', data=df).fit(cov_type='HC3')
    ten_grid = np.linspace(0, 20, 100)
    sal_fixed_exp = [np.exp(m_exp_ten.predict(pd.DataFrame([{'ten': t, 'ten2': t**2, 'outside_exp': max(0, 20-t), 'outside_exp2': max(0, 20-t)**2, 'edu': 16, 'female': 0, 'metro': 1}]))[0]) for t in ten_grid]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    ax1.plot(ten_grid, sal_fixed_exp, color=CORAL, linewidth=3)
    ax1.fill_between(ten_grid, sal_fixed_exp, min(sal_fixed_exp), color=CORAL, alpha=0.1)
    ax1.plot(ten_grid[10], sal_fixed_exp[10], 'o', color=NAVY, markersize=9)
    ax1.annotate(f"External Hire\n(2y Tenure, 18y Outside)\n${sal_fixed_exp[10]:.2f}/hr", xy=(ten_grid[10], sal_fixed_exp[10]), xytext=(ten_grid[10] + 1.5, sal_fixed_exp[10] + 0.6),
                 arrowprops=dict(facecolor=NAVY, shrink=0.08, width=1.5, headwidth=6), fontweight='bold', color=NAVY, bbox=dict(boxstyle="round", facecolor="white", edgecolor=NAVY))
    ax1.plot(ten_grid[-1], sal_fixed_exp[-1], 'o', color=CORAL, markersize=9)
    ax1.annotate(f"Loyal Stayer\n(20y Tenure Same Firm)\n${sal_fixed_exp[-1]:.2f}/hr", xy=(ten_grid[-1], sal_fixed_exp[-1]), xytext=(ten_grid[-1] - 7, sal_fixed_exp[-1] - 1.2),
                 arrowprops=dict(facecolor=CORAL, shrink=0.08, width=1.5, headwidth=6), fontweight='bold', color=CORAL, bbox=dict(boxstyle="round", facecolor="white", edgecolor=CORAL))
    ax1.set_xlabel("Years at Current Firm (Tenure) [Total Career Experience Fixed at 20 Years]", fontsize=10)
    ax1.set_ylabel("Predicted Hourly Wage ($/hr)", fontsize=11)
    ax1.set_title("A. The Internal 'Loyalty Discount' at 20 Years Career Experience", fontsize=12, fontweight='bold', color=SLATE)

    ax2.axis('off')
    table_data = [
        ["Years 1 - 3", "Low (Onboarding)", "Standard merit review (+3-5%)", "Normal internal band"],
        ["Years 4 - 7", "Moderate (Poaching)", "Tenure Parity Adjustment", "+8% to +12% Retention Equity"],
        ["Years 8 - 14", "HIGH (Market Wedge)", "Comprehensive Market Refresher", "Re-benchmark vs outside offers"],
        ["Years 15+", "CRITICAL (Ceiling)", "Executive / Principal Track", "LTI / Profit share to prevent loss"]
    ]
    tbl = ax2.table(cellText=table_data, colLabels=["Tenure Cohort", "Flight Risk Level", "Recommended Retention Action", "Target Adjustment"], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.15, 2.2)
    for (r_idx, _), cell in tbl.get_celld().items():
        if r_idx == 0: cell.set_facecolor(NAVY); cell.set_text_props(color='white', fontweight='bold')
    ax2.set_title("B. Strategic Retention Roadmap to Stop Senior Talent Loss", fontsize=12, fontweight='bold', color=SLATE, pad=20)
    fig.suptitle("RECOMMENDATION 2: Counteracting the 'Loyalty Penalty' to Retain Institutional Knowledge", fontsize=15, fontweight='bold', color=NAVY, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "rec_plot2_loyalty_penalty_vs_market.png")

    # Rec Plot 3: Upskilling & Career Value Ladder
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    m_full = ols("ln_sal ~ female + married + edu + edu2 + exper + exp2 + ten + ten2 + wrkhm + itprog + itarch + itai + itqual + itclient + itrd + metro", data=df).fit()
    base_p = {'female': 0, 'married': 1, 'edu': 12, 'edu2': 144, 'exper': 10, 'exp2': 100, 'ten': 5, 'ten2': 25,
              'wrkhm': 0, 'itprog': 0, 'itarch': 0, 'itai': 0, 'itqual': 0, 'itclient': 0, 'itrd': 0, 'metro': 1}
    ladder_steps = ["High School (12y)", "Bachelor's (16y)", "+ AI / Machine Learning", "+ Enterprise Architecture", "Master's Degree (18y)", "Executive Leadership"]
    p1 = np.exp(m_full.predict(pd.DataFrame([base_p]))[0])
    p2 = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=16, edu2=256, itprog=1)]))[0])
    p3 = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=16, edu2=256, itai=1)]))[0])
    p4 = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=16, edu2=256, itarch=1)]))[0])
    p5 = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=18, edu2=324, itarch=1)]))[0])
    p6 = p5 * 1.25
    ladder_vals = [p1, p2, p3, p4, p5, p6]
    annual_vals = [v * 2080 for v in ladder_vals]
    ax.bar(range(len(ladder_steps)), ladder_vals, color=[GRAY, TEAL, '#00A86B', AMBER, PURPLE, NAVY], width=0.55, edgecolor=SLATE, linewidth=1.2)
    for i, h in enumerate(ladder_vals):
        ax.text(i, h + 0.15, f"${h:.2f}/hr\n(${annual_vals[i]:,.0f}/yr)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=SLATE)
    ax.set_xticks(range(len(ladder_steps)))
    ax.set_xticklabels(ladder_steps, fontsize=9.5, fontweight='semibold')
    ax.set_ylabel("Predicted Market Compensation ($/hr)", fontsize=11)
    ax.set_title("The Human Capital ROI Ladder: Incremental Value of Degrees & Technical Specialization", fontsize=13, fontweight='bold', color=NAVY)
    ax.set_ylim(0, max(ladder_vals) * 1.28)
    plt.tight_layout()
    save_fig(fig, "rec_plot3_upskilling_and_career_ladder.png")

    # Rec Plot 4: 2x2 Workplace Location Matrix
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    p_metro_office = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=16, edu2=256, itprog=1, metro=1, wrkhm=0)]))[0])
    p_metro_remote = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=16, edu2=256, itprog=1, metro=1, wrkhm=1)]))[0])
    p_reg_office = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=16, edu2=256, itprog=1, metro=0, wrkhm=0)]))[0])
    p_reg_remote = np.exp(m_full.predict(pd.DataFrame([dict(base_p, edu=16, edu2=256, itprog=1, metro=0, wrkhm=1)]))[0])
    matrix_labels = [
        "Tier 1: Metro In-Office\n(Anchor Tech Hub)",
        "Tier 2: Metro Hybrid / Remote\n(High-Cost Flexible)",
        "Tier 3: Regional In-Office\n(Branch Tech Office)",
        "Tier 4: Regional Remote\n(Full Geo-Flexibility)"
    ]
    matrix_wages = [p_metro_office, p_metro_remote, p_reg_office, p_reg_remote]
    bench = matrix_wages[0]
    bars_m = ax.barh(np.arange(len(matrix_labels))[::-1], matrix_wages, color=[NAVY, TEAL, AMBER, CORAL], height=0.5, edgecolor=SLATE, linewidth=1.2)
    for i, b in enumerate(bars_m):
        w_val = matrix_wages[len(matrix_wages) - 1 - i]
        pct = 100 * w_val / bench
        ax.text(w_val + 0.1, b.get_y() + b.get_height()/2.0, f"${w_val:.2f}/hr  |  ${w_val*2080:,.0f}/yr  ({pct:.1f}% of Benchmark)",
                va='center', fontsize=10, fontweight='bold', color=SLATE)
    ax.set_yticks(np.arange(len(matrix_labels))[::-1])
    ax.set_yticklabels(matrix_labels, fontsize=10, fontweight='semibold')
    ax.set_xlabel("Predicted Hourly Wage ($/hr)", fontsize=11)
    ax.set_title("RECOMMENDATION 4: 2x2 Geographic & Workplace Flexibility Compensation Policy", fontsize=13, fontweight='bold', color=NAVY)
    ax.set_xlim(0, max(matrix_wages) * 1.45)
    plt.tight_layout()
    save_fig(fig, "rec_plot4_workplace_location_policy_matrix.png")


# ======================================================================
# MASTER DISPATCHER
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Master Plot Generator for IT Salary Econometric Study")
    parser.add_argument("--all", action="store_true", default=True, help="Generate all plot suites")
    parser.add_argument("--descriptive", action="store_true", help="Generate descriptive distribution plots")
    parser.add_argument("--report", action="store_true", help="Generate core report econometric visuals")
    parser.add_argument("--diagnostics", action="store_true", help="Generate four-panel regression diagnostics")
    parser.add_argument("--client", action="store_true", help="Generate client briefing charts and executive dashboard")
    parser.add_argument("--recommendations", action="store_true", help="Generate strategic HR recommendation plots")
    args = parser.parse_args()

    # Determine what to run
    run_all = not (args.descriptive or args.report or args.diagnostics or args.client or args.recommendations)

    print("=" * 75)
    print("MASTER PLOT GENERATOR: DOM 207 ECONOMETRIC CONSULTING SUITE")
    print("=" * 75)
    print("Loading and preparing data...")
    df_clean, df_full, col_types, X_combined, m_combined_ols, m_combined_hc3 = load_and_preprocess_data()
    print(f"Sample validated: N = {len(df_full)} observations. Ready to generate graphics.\n")

    if run_all or args.descriptive:
        generate_descriptive_plots(df_clean, col_types)

    if run_all or args.report:
        generate_report_visuals(df_full, X_combined, m_combined_hc3)

    if run_all or args.diagnostics:
        generate_four_panel_diagnostics(df_full, X_combined, m_combined_ols)

    if run_all or args.client:
        generate_client_plots(df_full)

    if run_all or args.recommendations:
        generate_recommendation_plots(df_full)

    print("\n" + "=" * 75)
    print("SUCCESS: All requested visualization suites generated and saved in 'figures/'!")
    print("=" * 75)


if __name__ == "__main__":
    main()
