# -*- coding: utf-8 -*-
"""
Null Baseline Model for Eye-tracking Data.
Predictor: Log Verb Frequency (scaled)
Dependent Variables:
  1. First-Pass Reading Time       (IA_FIRST_RUN_DWELL_TIME)
  2. Regression-Path Reading Time  (IA_REGRESSION_PATH_DURATION)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1. Load & Prepare Data 
print("Loading data...")
df = pd.read_csv('critical_RTdata_260319(in).csv')
df.columns = df.columns.str.strip()
df_verb = df[df['IA_LABEL'] == 'Verb'].copy()

# Coerce all needed columns to numeric
rt_cols = ['IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION']
for col in rt_cols + ['verb_log_freq']:
    df_verb[col] = pd.to_numeric(df_verb[col], errors='coerce')

# Build shared dataset (valid rows for both DVs)
df_base = df_verb[['Participant', 'item', 'verb_log_freq'] + rt_cols].dropna()
df_base = df_base[
    (df_base['IA_FIRST_RUN_DWELL_TIME'] > 0) &
    (df_base['IA_REGRESSION_PATH_DURATION'] > 0)
].copy()

# Coerce Participant and item to string for categorical random effects
df_base['Participant'] = df_base['Participant'].astype(str)
df_base['item'] = df_base['item'].astype(str)
df_base['group'] = 1


# Log-transform both DVs
df_base['log_fp'] = np.log(df_base['IA_FIRST_RUN_DWELL_TIME'])
df_base['log_rp'] = np.log(df_base['IA_REGRESSION_PATH_DURATION'])

# Z-score the predictor
df_base['freq_z'] = (
    (df_base['verb_log_freq'] - df_base['verb_log_freq'].mean())
    / df_base['verb_log_freq'].std()
)

print(f"Dataset: {len(df_base)} observations, "
      f"{df_base['Participant'].nunique()} participants\n")

# Helper: pseudo-R² 
def pseudo_r2(result, data):
    fitted   = result.predict(data)
    sigma2_f = np.var(fitted)
    sigma2_g = np.sum(result.vcomp)
    sigma2_e = result.scale
    denom    = sigma2_f + sigma2_g + sigma2_e
    return sigma2_f / denom, (sigma2_f + sigma2_g) / denom

# 2. Fit models for both DVs
dvs = [
    ('log_fp', 'First-Pass Reading Time',       'IA_FIRST_RUN_DWELL_TIME'),
    ('log_rp', 'Regression-Path Reading Time',  'IA_REGRESSION_PATH_DURATION'),
]

ols_models = {}
lme_models = {}

for dv_col, dv_label, _ in dvs:
    formula = f"{dv_col} ~ freq_z"

    print(f"Fitting OLS  null model : {dv_label}...")
    ols_models[dv_label] = smf.ols(formula, df_base).fit()

    print(f"Fitting LME  null model : {dv_label}...")
    vc_formula = {
        'Participant': '0 + C(Participant)',
        'item': '0 + C(item)'
    }
    lme_models[dv_label] = smf.mixedlm(
        formula, df_base, groups=df_base['group'], vc_formula=vc_formula
    ).fit(reml=True)

# 3. Print results
SEP = "=" * 80

print(f"\n{SEP}")
print("NULL BASELINE MODEL RESULTS")
print("Predictor: Log Verb Frequency (z-scored)")
print("DVs: First-Pass RT | Regression-Path RT")
print(SEP)

for _, dv_label, _ in dvs:
    print(f"\n{'=' * 80}")
    print(f"DV: {dv_label}")
    print(f"{'=' * 80}")

    print("\n--- STANDARD OLS ---")
    print(ols_models[dv_label].summary())

    print("\n--- MIXED-EFFECTS LME (by-Participant random intercept) ---")
    print(lme_models[dv_label].summary())

    r2m, r2c = pseudo_r2(lme_models[dv_label], df_base)
    print(f"  LME Marginal  R2: {r2m:.4f}")
    print(f"  LME Conditional R2: {r2c:.4f}")
    print(f"  OLS R2: {ols_models[dv_label].rsquared:.4f}")

# 4. Summary table
print(f"\n{SEP}")
print("SUMMARY: Null Baseline Model Performance")
print(SEP)
print(f"{'DV':<30} {'OLS R2':>8} {'LME Marg R2':>13} {'LME Cond R2':>13} "
      f"{'freq_z b':>10} {'freq_z p':>10}")
print("-" * 90)

for _, dv_label, _ in dvs:
    ols  = ols_models[dv_label]
    lme  = lme_models[dv_label]
    r2m, r2c = pseudo_r2(lme, df_base)
    b    = lme.params['freq_z']
    p    = lme.pvalues['freq_z']
    sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{dv_label:<30} {ols.rsquared:8.4f} {r2m:13.4f} {r2c:13.4f} "
          f"{b:10.3f} {p:10.3f} {sig}")

# 5. Save results
output_path = 'model_results_null_frequency.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("NULL BASELINE MODEL (FREQ ONLY)\n")
    f.write("Predictor: Log Verb Frequency (z-scored)\n")
    f.write("DVs: First-Pass RT | Regression-Path RT\n")
    f.write("=" * 80 + "\n\n")

    for _, dv_label, _ in dvs:
        ols = ols_models[dv_label]
        lme = lme_models[dv_label]
        r2m, r2c = pseudo_r2(lme, df_base)

        f.write(f"\n{'=' * 80}\n")
        f.write(f"DV: {dv_label}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write("STANDARD OLS:\n")
        f.write(ols.summary().as_text())
        f.write(f"\n\nMIXED-EFFECTS LME:\n")
        f.write(lme.summary().as_text())
        f.write(f"\n\nLME R-squared:\n")
        f.write(f"  Marginal  R2: {r2m:.4f}\n")
        f.write(f"  Conditional R2: {r2c:.4f}\n")
        f.write(f"  OLS R2: {ols.rsquared:.4f}\n")

print(f"\nResults saved to '{output_path}'")
