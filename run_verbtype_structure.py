# -*- coding: utf-8 -*-
"""
Run Model: Surprisal + VerbType x Structure + Frequency
========================================================
Fits the additive model:
  Formula: log_RT ~ freq_z + Surprisal_z + C(VerbType) * C(structure)

"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1. Load and Prepare Dataset
print("Loading data...")
df = pd.read_csv('critical_RTdata_260319(in).csv')
df.columns = df.columns.str.strip()

df_verb = df[df['IA_LABEL'] == 'Verb'].copy()
df_verb = df_verb.rename(columns={'verb': 'VerbType'})

# Coerce columns to numeric
for col in ['Surprisal', 'verb_log_freq', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION']:
    df_verb[col] = pd.to_numeric(df_verb[col], errors='coerce')

df_verb['structure'] = df_verb['structure'].str.strip()
df_verb['VerbType']  = pd.Categorical(df_verb['VerbType'],
                                       categories=['CP', 'DP', 'unacc'])
df_verb['structure'] = pd.Categorical(df_verb['structure'],
                                       categories=['move', 'nomove'])

cols_needed = ['Participant', 'item', 'VerbType', 'structure',
               'Surprisal', 'verb_log_freq',
               'IA_FIRST_RUN_DWELL_TIME',
               'IA_REGRESSION_PATH_DURATION']

df_base = df_verb[cols_needed].dropna()
df_base = df_base[
    (df_base['IA_FIRST_RUN_DWELL_TIME'] > 0) &
    (df_base['IA_REGRESSION_PATH_DURATION'] > 0)
].copy()

# Coerce Participant and item to string for categorical random effects
df_base['Participant'] = df_base['Participant'].astype(str)
df_base['item'] = df_base['item'].astype(str)
df_base['group'] = 1

# Log-transform DVs
df_base['log_fp'] = np.log(df_base['IA_FIRST_RUN_DWELL_TIME'])
df_base['log_rp'] = np.log(df_base['IA_REGRESSION_PATH_DURATION'])

# Z-score predictors
df_base['Surprisal_z'] = (
    (df_base['Surprisal'] - df_base['Surprisal'].mean())
    / df_base['Surprisal'].std()
)
df_base['freq_z'] = (
    (df_base['verb_log_freq'] - df_base['verb_log_freq'].mean())
    / df_base['verb_log_freq'].std()
)

print(f"Dataset prepared: {len(df_base)} observations, "
      f"{df_base['Participant'].nunique()} participants\n")


# 2. Fit Model using REML for Coefficients & p-values
formula_model = "{dv} ~ freq_z + Surprisal_z + C(VerbType, Treatment('CP')) * C(structure, Treatment('move'))"

print("========================================================================")
print("FITTING VERBTYPE X STRUCTURE MODEL (REML)")
print("========================================================================")

reml_results = {}
r2_results = {}
for dv_col, dv_label in [('log_fp', 'First-Pass RT'), ('log_rp', 'Regression-Path RT')]:
    formula = formula_model.format(dv=dv_col)
    print(f"\nFitting REML for {dv_label}...")
    vc_formula = {
        'Participant': '0 + C(Participant)',
        'item': '0 + C(item)'
    }
    lme = smf.mixedlm(formula, df_base, groups=df_base['group'], vc_formula=vc_formula)
    reml_results[dv_label] = lme.fit(reml=True)
    
    # Print summary
    print(reml_results[dv_label].summary())
    
    # Calculate R-squared
    # Marginal R2 = var(fixed) / (var(fixed) + var(random) + var(residual))
    fixed_pred = reml_results[dv_label].predict(df_base)
    
    var_fixed = np.var(fixed_pred)
    var_random = np.sum(reml_results[dv_label].vcomp)
    var_resid = reml_results[dv_label].scale
    
    marginal_r2 = var_fixed / (var_fixed + var_random + var_resid)
    conditional_r2 = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
    
    r2_results[dv_label] = (marginal_r2, conditional_r2)
    
    print(f"  Marginal R2 (fixed only): {marginal_r2:.4f}")
    print(f"  Conditional R2 (fixed + random): {conditional_r2:.4f}")


# 3. Save results to text file
output_path = 'results_verbtype_structure.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("RESULTS: MODEL (SURPRISAL + VERBTYPE X STRUCTURE + FREQUENCY)\n")
    f.write("=" * 80 + "\n\n")
    
    for dv_label in ['First-Pass RT', 'Regression-Path RT']:
        f.write(f"=== REML Summary for {dv_label} ===\n")
        f.write(reml_results[dv_label].summary().as_text())
        r2m, r2c = r2_results[dv_label]
        f.write(f"\n  Marginal R2 (fixed effects only): {r2m:.4f}\n")
        f.write(f"  Conditional R2 (fixed + random):    {r2c:.4f}\n\n")

print(f"\nResults successfully saved to '{output_path}'")
