# -*- coding: utf-8 -*-
"""
AIC/BIC Comparison of Five LME Models (Null Model as Base)
===========================================================
Compares:
  1. Null Model                         – Verb Frequency only
  2. Surprisal Only Model (with Freq)   – Surprisal_z + freq_z
  3. VerbType * Condition Model (+ Freq) – freq_z + VerbType * Structure
  4. Surprisal + Condition Model (+ Freq) – freq_z + Surprisal_z + VerbType * Structure (Model D)
  5. Surprisal * Condition Model (+ Freq) – freq_z + Surprisal_z * VerbType * Structure (Model C)

All models are fit with ML (not REML) on the identical dataset so that
AIC and BIC are directly comparable.
DVs: First-Pass RT | Regression-Path RT
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1. Load & prepare dataset
print("Loading data...")
df = pd.read_csv('critical_RTdata_260319(in).csv')
df.columns = df.columns.str.strip()

df_verb = df[df['IA_LABEL'] == 'Verb'].copy()
df_verb = df_verb.rename(columns={'verb': 'VerbType'})

# Coerce columns to numeric
cols_to_coerce = ['Surprisal', 'verb_log_freq', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION']
for col in cols_to_coerce:
    df_verb[col] = pd.to_numeric(df_verb[col], errors='coerce')

df_verb['structure'] = df_verb['structure'].str.strip()
df_verb['VerbType']  = pd.Categorical(df_verb['VerbType'], categories=['CP', 'DP', 'unacc'])
df_verb['structure'] = pd.Categorical(df_verb['structure'], categories=['move', 'nomove'])

cols_needed = ['Participant', 'item', 'VerbType', 'structure',
               'Surprisal', 'verb_log_freq',
               'IA_FIRST_RUN_DWELL_TIME',
               'IA_REGRESSION_PATH_DURATION']

# Build shared dataset (rows present in both DVs with valid non-zero values)
df_base = df_verb[cols_needed].dropna()
df_base = df_base[
    (df_base['IA_FIRST_RUN_DWELL_TIME'] > 0) &
    (df_base['IA_REGRESSION_PATH_DURATION'] > 0)
].copy()

# Coerce grouping columns to categorical string and add dummy group for crossed effects
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

print(f"Shared dataset prepared: {len(df_base)} observations, "
      f"{df_base['Participant'].nunique()} participants\n")

# 2. Define the five models
formula_Null = "{dv} ~ freq_z"

formula_Surp = "{dv} ~ Surprisal_z + freq_z"

formula_Cond = ("{dv} ~ freq_z + C(VerbType, Treatment('CP'))"
                " * C(structure, Treatment('move'))")

formula_Add  = ("{dv} ~ freq_z + Surprisal_z + C(VerbType, Treatment('CP'))"
                " * C(structure, Treatment('move'))")

formula_Int  = ("{dv} ~ freq_z + Surprisal_z * C(VerbType, Treatment('CP'))"
                " * C(structure, Treatment('move'))")

model_specs = [
    ('1. Null (Freq Only)',           formula_Null),
    ('2. Surprisal Only (+ Freq)',    formula_Surp),
    ('3. VerbType * Structure (+ Freq)', formula_Cond),
    ('4. Surprisal + Condition (+ Freq)', formula_Add),
    ('5. Surprisal * Condition (+ Freq)', formula_Int),
]

# 3. Fit all models with ML
models = {}
print("Fitting models with ML estimation...\n")

vc_formula = {
    'Participant': '0 + C(Participant)',
    'item': '0 + C(item)'
}

for dv_col, dv_label in [('log_fp', 'First-Pass RT'),
                          ('log_rp', 'Regression-Path RT')]:
    for label, formula_tmpl in model_specs:
        formula = formula_tmpl.format(dv=dv_col)
        key = f"{label} | {dv_label}"
        print(f"  Fitting: {key}...")

        lme = smf.mixedlm(formula, df_base, groups=df_base['group'], vc_formula=vc_formula)
        # Start parameters from REML to ensure convergence stability
        res_reml = lme.fit(reml=True)
        start_params = res_reml.params

        try:
            result = lme.fit(reml=False, method='cg',
                             start_params=start_params)
            if not np.isfinite(result.llf):
                raise ValueError("Non-finite log-likelihood")
        except Exception:
            result = lme.fit(reml=False, method='nm',
                             start_params=start_params)

        models[key] = result

print("\nAll models fitted.\n")

# 4. Extract stats
def model_stats(result, label, dv_label, data):
    k   = result.params.shape[0]
    ll  = result.llf
    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(result.nobs) * k
    
    # Calculate R-squared
    fitted = result.predict(data)
    var_fixed = np.var(fitted)
    var_random = np.sum(result.vcomp)
    var_resid = result.scale
    denom = var_fixed + var_random + var_resid
    r2_m = var_fixed / denom
    r2_c = (var_fixed + var_random) / denom
    
    return {
        'DV'       : dv_label,
        'Model'    : label,
        'N params' : k,
        'Log-Lik'  : round(ll,  2),
        'AIC'      : round(aic, 2),
        'BIC'      : round(bic, 2),
        'Marginal R2': round(r2_m, 4),
        'Conditional R2': round(r2_c, 4),
    }

rows = []
for dv_label in ['First-Pass RT', 'Regression-Path RT']:
    for label, _ in model_specs:
        key = f"{label} | {dv_label}"
        rows.append(model_stats(models[key], label, dv_label, df_base))

results_df = pd.DataFrame(rows)

# 5. Delta-AIC relative to the Null model (Verb Frequency)
for dv in results_df['DV'].unique():
    mask = results_df['DV'] == dv
    null_aic = results_df.loc[mask & (results_df['Model'] == '1. Null (Freq Only)'), 'AIC'].values[0]
    results_df.loc[mask, 'Delta-AIC (vs Null)'] = (
        results_df.loc[mask, 'AIC'] - null_aic
    ).round(2)

    # Compute absolute best model delta-AIC and AIC weights within each DV
    min_aic = results_df.loc[mask, 'AIC'].min()
    results_df.loc[mask, 'Delta-AIC (vs Best)'] = (
        results_df.loc[mask, 'AIC'] - min_aic
    ).round(2)

    deltas = results_df.loc[mask, 'Delta-AIC (vs Best)'].values
    rel    = np.exp(-0.5 * deltas)
    weights = rel / rel.sum()
    results_df.loc[mask, 'AIC Weight'] = weights.round(4)

# 6. Print and save results
header = "FIVE-MODEL AIC/BIC COMPARISON (Null Model as Base)"
print("=" * 110)
print(header)
print("=" * 110)
print("\nAll models fit with ML on identical datasets (crossed by-Participant and by-item intercepts).")
print("Delta-AIC (vs Null) < 0 indicates improved model fit compared to Frequency baseline.\n")

for dv in ['First-Pass RT', 'Regression-Path RT']:
    sub = results_df[results_df['DV'] == dv][
        ['Model', 'N params', 'Log-Lik', 'AIC', 'BIC', 'Marginal R2', 'Conditional R2', 'Delta-AIC (vs Null)', 'Delta-AIC (vs Best)', 'AIC Weight']
    ].reset_index(drop=True)

    print(f"\n--- {dv} ---")
    print(sub.to_string(index=False))

output_path = 'five_model_aic_comparison.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"{header}\n")
    f.write("=" * 110 + "\n\n")
    f.write("All models fit with ML (not REML) on identical datasets.\n")
    f.write(f"N observations: {len(df_base)}\n")
    f.write(f"N participants: {df_base['Participant'].nunique()}\n\n")
    
    for dv in ['First-Pass RT', 'Regression-Path RT']:
        f.write(f"--- {dv} ---\n")
        sub = results_df[results_df['DV'] == dv][
            ['Model', 'N params', 'Log-Lik', 'AIC', 'BIC', 'Marginal R2', 'Conditional R2', 'Delta-AIC (vs Null)', 'Delta-AIC (vs Best)', 'AIC Weight']
        ].reset_index(drop=True)
        f.write(sub.to_string(index=False))
        f.write("\n\n")
        
    f.write("Interpretation guide:\n")
    f.write("  Delta-AIC (vs Null) < 0 : Model fits better than the Frequency Baseline.\n")
    f.write("  Delta-AIC (vs Null) > 0 : Model fits worse than the Frequency Baseline.\n")
    f.write("  Rule of thumb: Difference > 2 is meaningful; > 7 is strong evidence.\n")

print(f"\nResults successfully saved to '{output_path}'")
