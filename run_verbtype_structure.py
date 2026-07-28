# -*- coding: utf-8 -*-
"""
LME Model: VerbType x Structure + Verb Frequency
=================================================
Formula: log(RT) ~ freq_z + C(VerbType) * C(structure)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1. Load and Prepare
print("Loading data...")
df = pd.read_csv('critical_RTdata_260319(in).csv')
df.columns = df.columns.str.strip()

df_verb = df[df['IA_LABEL'] == 'Verb'].copy()
df_verb = df_verb.rename(columns={'verb': 'VerbType'})

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

# Coerce grouping columns to categorical string
df_base['Participant'] = df_base['Participant'].astype(str)
df_base['item'] = df_base['item'].astype(str)
df_base['group'] = 1

df_base['log_fp'] = np.log(df_base['IA_FIRST_RUN_DWELL_TIME'])
df_base['log_rp'] = np.log(df_base['IA_REGRESSION_PATH_DURATION'])

df_base['freq_z'] = (
    (df_base['verb_log_freq'] - df_base['verb_log_freq'].mean())
    / df_base['verb_log_freq'].std()
)

print(f"Dataset: {len(df_base)} observations, "
      f"{df_base['Participant'].nunique()} participants\n")

# 2. Model VerbType x Structure + Frequency
formula_Bplus = "{dv} ~ freq_z + C(VerbType, Treatment('CP')) * C(structure, Treatment('move'))"

output_lines = []
output_lines.append("LME MODEL: VerbType x Structure + Verb Frequency")
output_lines.append("=" * 72)
output_lines.append("Formula: log(RT) ~ freq_z + VerbType * Structure")
output_lines.append("Reference levels: VerbType = CP, Structure = move")
output_lines.append("")

print("=" * 72)
print("MODEL: VerbType x Structure + Verb Frequency (REML)")
print("=" * 72)

reml_results = {}
for dv_col, dv_label in [('log_fp', 'First-Pass Reading Time'),
                          ('log_rp', 'Regression-Path Reading Time')]:
    formula = formula_Bplus.format(dv=dv_col)
    print(f"\n{'='*72}")
    print(f"DV: {dv_label}")
    print(f"{'='*72}")

    vc_formula = {
        'Participant': '0 + C(Participant)',
        'item': '0 + C(item)'
    }
    lme = smf.mixedlm(formula, df_base, groups=df_base['group'], vc_formula=vc_formula)
    result = lme.fit(reml=True)
    reml_results[dv_label] = result
    print(result.summary())

    # R-squared
    fixed_pred = result.predict(df_base)
    var_fixed = np.var(fixed_pred)
    var_random = np.sum(result.vcomp)
    var_resid = result.scale
    marginal_r2 = var_fixed / (var_fixed + var_random + var_resid)
    conditional_r2 = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
    print(f"\n  Marginal R2 (fixed only): {marginal_r2:.4f}")
    print(f"  Conditional R2 (fixed + random): {conditional_r2:.4f}")

    output_lines.append(f"{'='*72}")
    output_lines.append(f"DV: {dv_label}")
    output_lines.append(f"{'='*72}")
    output_lines.append(result.summary().as_text())
    output_lines.append(f"\n  Marginal R2 (fixed only): {marginal_r2:.4f}")
    output_lines.append(f"  Conditional R2 (fixed + random): {conditional_r2:.4f}")
    output_lines.append("")

output_path = 'results_verbtype_structure.txt'
with open(output_path, 'w', encoding='utf-8') as fout:
    fout.write('\n'.join(output_lines))

print(f"\nResults successfully saved to '{output_path}'")
