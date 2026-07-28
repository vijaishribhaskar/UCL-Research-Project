# -*- coding: utf-8 -*-
"""
Linear Mixed Effects (LME) Models: Surprisal → Reading Times
=============================================================
Independent Variable (IV):  Surprisal
Dependent Variables (DVs):
  1. First-Pass Reading Time  (IA_FIRST_RUN_DWELL_TIME)
  2. Regression-Path Reading Time (IA_REGRESSION_PATH_DURATION)

Random effect: by-participant intercept (Participant)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1. Load & filter
print("Loading data...")
df = pd.read_csv('critical_RTdata_260319(in).csv')
df.columns = df.columns.str.strip()

# Keep only the Verb region
df_verb = df[df['IA_LABEL'] == 'Verb'].copy()

cols = ['Participant', 'item',
        'Surprisal', 'verb_log_freq',
        'IA_FIRST_RUN_DWELL_TIME',
        'IA_REGRESSION_PATH_DURATION']

df_verb = df_verb[cols].copy()

# Coerce to numeric (entries like 'NA' become NaN)
for col in cols[1:]:
    df_verb[col] = pd.to_numeric(df_verb[col], errors='coerce')

# 2. First-Pass model dataset
df_fp = df_verb[['Participant', 'item', 'Surprisal', 'verb_log_freq', 'IA_FIRST_RUN_DWELL_TIME']].dropna()
df_fp = df_fp[df_fp['IA_FIRST_RUN_DWELL_TIME'] > 0].copy()
df_fp['log_fp']         = np.log(df_fp['IA_FIRST_RUN_DWELL_TIME'])
df_fp['Surprisal_z']    = (df_fp['Surprisal'] - df_fp['Surprisal'].mean()) / df_fp['Surprisal'].std()
df_fp['freq_z']         = (df_fp['verb_log_freq'] - df_fp['verb_log_freq'].mean()) / df_fp['verb_log_freq'].std()
df_fp['Participant']    = df_fp['Participant'].astype(str)
df_fp['item']           = df_fp['item'].astype(str)
df_fp['group']          = 1
print(f"First-Pass model dataset: {len(df_fp)} observations")

# 3. Regression-Path model dataset
df_rp = df_verb[['Participant', 'item', 'Surprisal', 'verb_log_freq', 'IA_REGRESSION_PATH_DURATION']].dropna()
df_rp = df_rp[df_rp['IA_REGRESSION_PATH_DURATION'] > 0].copy()
df_rp['log_rp']         = np.log(df_rp['IA_REGRESSION_PATH_DURATION'])
df_rp['Surprisal_z']    = (df_rp['Surprisal'] - df_rp['Surprisal'].mean()) / df_rp['Surprisal'].std()
df_rp['freq_z']         = (df_rp['verb_log_freq'] - df_rp['verb_log_freq'].mean()) / df_rp['verb_log_freq'].std()
df_rp['Participant']    = df_rp['Participant'].astype(str)
df_rp['item']           = df_rp['item'].astype(str)
df_rp['group']          = 1
print(f"Regression-Path model dataset: {len(df_rp)} observations")

# Helper: R²-like statistics
def pseudo_r2(result, data, dv_col):
    """Compute marginal and conditional pseudo-R² """
    fitted     = result.predict(data)
    sigma2_f   = np.var(fitted)
    sigma2_g   = np.sum(result.vcomp)
    sigma2_e   = result.scale        
    denom      = sigma2_f + sigma2_g + sigma2_e
    r2_m       = sigma2_f / denom
    r2_c       = (sigma2_f + sigma2_g) / denom
    return r2_m, r2_c

# 4. Model 1 – First-Pass Reading Time 
print("\nFitting Model 1: Surprisal + Freq -> First-Pass Reading Time...")
formula_fp = "log_fp ~ Surprisal_z + freq_z"
vc_formula = {
    'Participant': '0 + C(Participant)',
    'item': '0 + C(item)'
}
lme_fp     = smf.mixedlm(formula_fp, df_fp, groups=df_fp['group'], vc_formula=vc_formula)
res_fp     = lme_fp.fit(reml=True)

r2m_fp, r2c_fp = pseudo_r2(res_fp, df_fp, 'log_fp')

header_fp = (
    "\n" + "=" * 72 + "\n"
    "MODEL 1 – First-Pass Reading Time  (DV: log IA_FIRST_RUN_DWELL_TIME)\n"
    "IVs: Surprisal (z-scored) + Verb Frequency (z-scored)   |   Random effects: by-Participant and by-item intercepts\n"
    + "=" * 72
)
print(header_fp)
print(res_fp.summary())
print(f"  Marginal  R²  (fixed effects only): {r2m_fp:.4f}")
print(f"  Conditional R² (fixed + random):    {r2c_fp:.4f}")

# 5. Model 2 – Regression-Path Reading Time
print("\nFitting Model 2: Surprisal + Freq -> Regression-Path Reading Time...")
formula_rp = "log_rp ~ Surprisal_z + freq_z"
lme_rp     = smf.mixedlm(formula_rp, df_rp, groups=df_rp['group'], vc_formula=vc_formula)
res_rp     = lme_rp.fit(reml=True)

r2m_rp, r2c_rp = pseudo_r2(res_rp, df_rp, 'log_rp')

header_rp = (
    "\n" + "=" * 72 + "\n"
    "MODEL 2 – Regression-Path Reading Time  (DV: log IA_REGRESSION_PATH_DURATION)\n"
    "IVs: Surprisal (z-scored) + Verb Frequency (z-scored)   |   Random effects: by-Participant and by-item intercepts\n"
    + "=" * 72
)
print(header_rp)
print(res_rp.summary())
print(f"  Marginal  R²  (fixed effects only): {r2m_rp:.4f}")
print(f"  Conditional R² (fixed + random):    {r2c_rp:.4f}")

# 6. Save results
output_path = 'lme_surprisal_results.txt'
with open(output_path, 'w', encoding='utf-8') as f:

    f.write("LINEAR MIXED EFFECTS MODELS – SURPRISAL AND READING TIMES\n")
    f.write("=" * 72 + "\n\n")
    f.write("IVs: Surprisal (z-scored) + Verb Frequency (z-scored)\n")
    f.write("DVs: First-Pass Reading Time | Regression-Path Reading Time\n")
    f.write("Random effects: by-Participant and by-item random intercepts\n")
    f.write("Region: Verb interest area\n")
    f.write("Transformation: log(RT) applied to both DVs\n\n")

    f.write(header_fp + "\n")
    f.write(res_fp.summary().as_text())
    f.write(f"\n  Marginal  R²  (fixed effects only): {r2m_fp:.4f}")
    f.write(f"\n  Conditional R² (fixed + random):    {r2c_fp:.4f}\n\n")

    f.write(header_rp + "\n")
    f.write(res_rp.summary().as_text())
    f.write(f"\n  Marginal  R²  (fixed effects only): {r2m_rp:.4f}")
    f.write(f"\n  Conditional R² (fixed + random):    {r2c_rp:.4f}\n")

print(f"\nResults saved to '{output_path}'")
