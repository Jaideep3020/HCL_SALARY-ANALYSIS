# Data Cleaning Techniques Description

This document details the exact data cleaning and preprocessing procedures performed in [`mp_2.ipynb`](mp_2.ipynb) on the dataset (`IT_employees_full.csv`).

---

## 1. Steps Performed (Point-by-Point)

* **Schema Alignment & Metadata Mapping:**
  * Imported metadata from `data_description.csv` and stripped whitespace from column headers and variable types.
  * Corrected description labels (mapped `nonprev` to `nominal`).
  * Structured all 25 features into a formal schema mapping of `nominal`, `numerical`, or `numerical(continuous)`.

* **String Typo Cleaning & Data Type Coercion:**
  * Converted every feature column using `pd.to_numeric(errors='coerce')` to eliminate non-numeric values and standardize data types.
  * Detected and replaced rogue string typos across 7 attributes with `NaN`:
    * `sal`: removed `'acb'`
    * `edu`: removed `'SS0'`
    * `metro`: removed `'b'`
    * `south`: removed `'aaa'`
    * `west`: removed `'a'` (2 occurrences)
    * `itqual`: removed `'s'`
    * `momed`: removed `'ab'`
  * Resulted in an all-numeric dataset structure.

* **Type-Specific Missing Value Imputation:**
  * Identified and imputed a total of **30 missing values** across the dataset according to variable type:
    * **Nominal Features:** Imputed using the feature **Mode** (preserving 0/1 indicator values).
    * **Continuous Numerical Features:** Imputed using the feature **Median** (`sal`, `exper`).
    * **Discrete Numerical Features:** Imputed using the **Rounded Median** (`edu`, `momed`, `daded`, `sib`, `dep`, `ten`) to preserve whole-integer counts and years.

* **Sign Rectification (Impossible Negative Values):**
  * Audited numerical features for negative values.
  * Located **1 impossible negative value** and rectified it by applying absolute value (`.abs()`), preserving magnitude while correcting sign.

* **Logical Consistency & Domain Rule Auditing:**
  * Evaluated logical constraints across interrelated columns:
    * Checked if company tenure exceeded career experience (`ten > exper`): Flagged **16 rows**.
    * Checked for simultaneous assignment to multiple regional provinces (`norcen`, `south`, `west`): Flagged **285 rows**.
    * Checked if employees marked with `itclient == 1` (*client-only*) also held other IT group roles: Flagged **107 rows**.

* **Outlier Detection via IQR Method:**
  * Calculated the $1.5 \times \text{IQR}$ threshold ($Q_1 - 1.5\text{IQR}$ and $Q_3 + 1.5\text{IQR}$) to identify distribution anomalies:
    * Created `flag_extreme_sal`: Flagged **29 extreme salary records**.
    * Created `flag_extreme_exper`: Flagged **0 extreme experience records**.

* **Deduplication & Data Integrity Validation:**
  * Audited for duplicate employee profiles across all non-ID features (`check_df.duplicated()`).
  * Confirmed **0 duplicate rows**.
  * Verified that total missing values (`NaN`) equaled **0** and confirmed **no leftover string/type mismatches**.

* **Feature Standardization (Z-Score Scaling):**
  * Created a normalized copy of the dataset (`df_normalized`) using `sklearn.preprocessing.StandardScaler`.
  * Scaled 8 numeric variables (`exper`, `ten`, `edu`, `momed`, `daded`, `sal`, `sib`, `dep`) to zero mean and unit variance ($\mu = 0, \sigma = 1$).

* **Descriptive Statistics Compilation:**
  * Computed summary statistics tables for both data classes:
    * **Numerical Features Table:** Count, Mean, Median, Std Dev, Min, Q1, Q3, IQR, and Max.
    * **Nominal Features Table:** Count, Mode, and distribution counts/percentages of `0`s and `1`s.

---

## 2. Summary Table of Operations

| Step | Technique | Target Columns | Results / Impact |
| :--- | :--- | :--- | :--- |
| **1. Metadata Mapping** | Stripping & Mapping | All columns | Standardized column names and data type mappings. |
| **2. Coercion** | `to_numeric(errors='coerce')` | `sal`, `edu`, `metro`, `south`, `west`, `itqual`, `momed` | Removed 8 string typos; all columns converted to numeric. |
| **3. Imputation** | Mode / Median / Rounded Median | All columns with nulls | Imputed 30 missing values with zero data loss. |
| **4. Sign Fix** | Absolute Value (`.abs()`) | Numerical columns | Fixed 1 invalid negative value. |
| **5. Logical Audits** | Domain Boundary Checks | `ten` vs `exper`, Regions, `itclient` | Flagged 16 tenure, 285 regional, and 107 client role conflicts. |
| **6. Outlier Flags** | Tukey's IQR Rule | `sal`, `exper` | Added `flag_extreme_sal` (29 flagged) and `flag_extreme_exper` (0 flagged). |
| **7. Validation** | Deduplication & NaN Audit | All columns (excl. `id`) | Confirmed 0 duplicates, 0 NaNs, and 0 type mismatches. |
| **8. Scaling** | `StandardScaler` (Z-score) | 8 numerical columns | Created `df_normalized` ($\mu=0, \sigma=1$). |
| **9. Profiling** | Summary Tables | Numerical & Nominal columns | Produced 5-number summaries, IQR, and category percentages. |

---

## 3. Brief Review of What Was Performed

The cleaning pipeline systematically converted a raw dataset of 526 rows into a completely clean, strictly numeric, and fully populated dataset. All rogue text inputs were removed, 30 missing values were filled according to feature-specific statistical rules, an invalid negative entry was corrected, and the dataset was confirmed to have zero duplicates. Domain logic discrepancies and extreme salary values were detected and explicitly flagged with indicator columns. Finally, a standardized dataset (`df_normalized`) was generated via Z-score scaling alongside baseline descriptive statistics tables.
