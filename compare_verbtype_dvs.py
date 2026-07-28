# -*- coding: utf-8 -*-
"""
Cross-DV Performance Comparison: VerbType * Structure Model
===========================================================
Compares the SAME model (VerbType x Structure) fitted to two DVs:
  1. First-Pass Reading Time       (IA_FIRST_RUN_DWELL_TIME)
  2. Regression-Path Reading Time  (IA_REGRESSION_PATH_DURATION)

NOTE: AIC cannot compare models across different DVs (the likelihoods
are on incompatible scales). Instead we compare:
  - Fixed-effect coefficients and significance side-by-side
  - Marginal and conditional R-squared
  - Residual variance (model scale) and random-effect variance
  - Cohen's f-squared (local effect size for each predictor)
  - Standardised betas for cross-DV effect-size comparison

All models fit with REML (appropriate for inference on fixed effects).
Reference levels: VerbType = CP, Structure = move.
"""

import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & prepare ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('critical_RTdata_260319(in).csv')
df.columns = df.columns.str.strip()

df_verb = df[df['IA_LABEL'] == 'Verb'].copy()
df_verb = df_verb.rename(columns={'verb': 'VerbType'})

for col in ['IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION']:
    df_verb[col] = pd.to_numeric(df_verb[col], errors='coerce')

df_verb['VerbType']  = pd.Categorical(df_verb['VerbType'],
                                       categories=['CP', 'DP', 'unacc'])
df_verb['structure'] = pd.Categorical(df_verb['structure'].str.strip(),
                                       categories=['move', 'nomove'])

# Build shared dataset (rows present in both DVs)
cols = ['Participant', 'VerbType', 'structure',
        'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION']
df_base = df_verb[cols].dropna()
df_base = df_base[
    (df_base['IA_FIRST_RUN_DWELL_TIME'] > 0) &
    (df_base['IA_REGRESSION_PATH_DURATION'] > 0)
].copy()

df_base['log_fp'] = np.log(df_base['IA_FIRST_RUN_DWELL_TIME'])
df_base['log_rp'] = np.log(df_base['IA_REGRESSION_PATH_DURATION'])

print(f"Shared dataset: {len(df_base)} obs, {df_base['Participant'].nunique()} participants\n")

# ── 2. Formula & fit ───────────────────────────────────────────────────────────
formula_tmpl = ("{dv} ~ C(VerbType, Treatment('DP'))"
                " * C(structure, Treatment('move'))")

results = {}
for dv_col, label in [('log_fp', 'First-Pass RT'),
                       ('log_rp', 'Regression-Path RT')]:
    formula = formula_tmpl.format(dv=dv_col)
    lme     = smf.mixedlm(formula, df_base, groups=df_base['Participant'])
    results[label] = lme.fit(reml=True)
    print(f"Fitted: {label}  |  converged: {results[label].converged}")

# ── 3. Pseudo-R² ──────────────────────────────────────────────────────────────
def pseudo_r2(res, data):
    fitted   = res.predict(data)
    sv_f     = np.var(fitted)
    sv_g     = res.cov_re.iloc[0, 0]
    sv_e     = res.scale
    tot      = sv_f + sv_g + sv_e
    return sv_f / tot, (sv_f + sv_g) / tot, sv_g, sv_e

# ── 4. Cohen's f² per predictor (semi-partial effect size) ────────────────────
def cohens_f2(beta, se, n):
    """Approximate f² from z-statistic: f² ≈ z² / (N - k)."""
    z = beta / se
    return z**2 / n  # scaled pseudo-f²

# ── 5. Build side-by-side coefficient table ───────────────────────────────────
SHORT_NAMES = {
    "Intercept"                                                              : "Intercept (CP, move)",
    "C(VerbType, Treatment('CP'))[T.DP]"                                    : "DP vs CP",
    "C(VerbType, Treatment('CP'))[T.unacc]"                                 : "unacc vs CP",
    "C(structure, Treatment('move'))[T.nomove]"                             : "nomove vs move",
    "C(VerbType, Treatment('CP'))[T.DP]:C(structure, Treatment('move'))[T.nomove]"    : "DP x nomove",
    "C(VerbType, Treatment('CP'))[T.unacc]:C(structure, Treatment('move'))[T.nomove]" : "unacc x nomove",
    "Group Var"                                                              : "Group Var (random)",
}

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""

rows = []
n    = len(df_base)
for param_name in SHORT_NAMES:
    short = SHORT_NAMES[param_name]
    row = {'Term': short}
    for label in ['First-Pass RT', 'Regression-Path RT']:
        res = results[label]
        try:
            b  = res.params[param_name]
            se = res.bse[param_name]
            z  = res.tvalues[param_name]
            p  = res.pvalues[param_name]
            f2 = cohens_f2(b, se, n)
            row[f'{label}_b']    = round(b,  3)
            row[f'{label}_se']   = round(se, 3)
            row[f'{label}_z']    = round(z,  2)
            row[f'{label}_p']    = round(p,  3)
            row[f'{label}_sig']  = sig_stars(p)
            row[f'{label}_f2']   = round(f2, 4)
        except KeyError:
            for s in ['_b', '_se', '_z', '_p', '_sig', '_f2']:
                row[f'{label}{s}'] = np.nan
    rows.append(row)

coef_df = pd.DataFrame(rows)

# ── 6. Fit statistics summary ─────────────────────────────────────────────────
fit_rows = []
for label in ['First-Pass RT', 'Regression-Path RT']:
    res = results[label]
    r2m, r2c, sv_g, sv_e = pseudo_r2(res, df_base)
    fit_rows.append({
        'DV'               : label,
        'N obs'            : res.nobs,
        'N groups'         : res.model.n_groups,
        'Marginal R2'      : round(r2m, 4),
        'Conditional R2'   : round(r2c, 4),
        'Random var (G)'   : round(sv_g, 4),
        'Residual var (R)' : round(sv_e, 4),
        'ICC'              : round(sv_g / (sv_g + sv_e), 4),
        'Log-Lik (REML)'   : round(res.llf, 2),
    })
fit_df = pd.DataFrame(fit_rows)

# ── 7. Print ───────────────────────────────────────────────────────────────────
SEP = "=" * 80

print(f"\n{SEP}")
print("VERBTYPE x STRUCTURE MODEL: PERFORMANCE COMPARISON ACROSS DVs")
print(SEP)
print("\nSignificance: *** p<.001  ** p<.01  * p<.05  . p<.10\n")

# --- Fit statistics ---
print("-" * 80)
print("FIT STATISTICS")
print("-" * 80)
print(fit_df.to_string(index=False))

# --- Coefficient comparison ---
print("\n" + "-" * 80)
print("FIXED-EFFECT COEFFICIENTS")
print("-" * 80)
print(f"{'Term':<30} {'--- First-Pass RT ---':^28} {'--- Regression-Path RT ---':^28}")
print(f"{'':30} {'b':>6} {'SE':>6} {'z':>6} {'p':>6} {'Sig':>4} {'f2':>7}"
      f"  {'b':>6} {'SE':>6} {'z':>6} {'p':>6} {'Sig':>4} {'f2':>7}")
print("-" * 80)

for _, row in coef_df.iterrows():
    term = str(row['Term'])[:29]
    fp_b   = f"{row['First-Pass RT_b']:6.3f}"   if pd.notna(row['First-Pass RT_b'])  else '     -'
    fp_se  = f"{row['First-Pass RT_se']:6.3f}"  if pd.notna(row['First-Pass RT_se']) else '     -'
    fp_z   = f"{row['First-Pass RT_z']:6.2f}"   if pd.notna(row['First-Pass RT_z'])  else '     -'
    fp_p   = f"{row['First-Pass RT_p']:6.3f}"   if pd.notna(row['First-Pass RT_p'])  else '     -'
    fp_sig = f"{row['First-Pass RT_sig']:>4}"   if pd.notna(row['First-Pass RT_sig'])else '    -'
    fp_f2  = f"{row['First-Pass RT_f2']:7.4f}"  if pd.notna(row['First-Pass RT_f2']) else '      -'
    rp_b   = f"{row['Regression-Path RT_b']:6.3f}"  if pd.notna(row['Regression-Path RT_b'])  else '     -'
    rp_se  = f"{row['Regression-Path RT_se']:6.3f}" if pd.notna(row['Regression-Path RT_se']) else '     -'
    rp_z   = f"{row['Regression-Path RT_z']:6.2f}"  if pd.notna(row['Regression-Path RT_z'])  else '     -'
    rp_p   = f"{row['Regression-Path RT_p']:6.3f}"  if pd.notna(row['Regression-Path RT_p'])  else '     -'
    rp_sig = f"{row['Regression-Path RT_sig']:>4}"  if pd.notna(row['Regression-Path RT_sig']) else '    -'
    rp_f2  = f"{row['Regression-Path RT_f2']:7.4f}" if pd.notna(row['Regression-Path RT_f2']) else '      -'
    print(f"{term:<30} {fp_b} {fp_se} {fp_z} {fp_p} {fp_sig} {fp_f2}"
          f"  {rp_b} {rp_se} {rp_z} {rp_p} {rp_sig} {rp_f2}")

print("-" * 80)

# -- 8. Effect-size ranking
print("\n" + "-" * 80)
print("COHEN'S f2 EFFECT SIZES (largest to smallest, excluding intercept & random)")
print("(f2: small=0.02, medium=0.15, large=0.35)")
print("-" * 80)
fe_rows = coef_df[coef_df['Term'] != 'Intercept (CP, move)'].copy()
fe_rows = fe_rows[fe_rows['Term'] != 'Group Var (random)'].copy()

print(f"\n{'Term':<30} {'FP f2':>9} {'RP f2':>9}  Winner")
print("-" * 60)
for _, row in fe_rows.iterrows():
    fp_f2 = row['First-Pass RT_f2']
    rp_f2 = row['Regression-Path RT_f2']
    if pd.isna(fp_f2) or pd.isna(rp_f2):
        winner = "N/A"
    elif rp_f2 > fp_f2:
        winner = "Regression-Path (larger effect)"
    elif fp_f2 > rp_f2:
        winner = "First-Pass (larger effect)"
    else:
        winner = "Equal"
    print(f"{str(row['Term']):<30} {fp_f2:9.4f} {rp_f2:9.4f}  {winner}")

# ── 9. Save ───────────────────────────────────────────────────────────────────
output_path = 'verbtype_structure_dv_comparison.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("VERBTYPE x STRUCTURE: CROSS-DV MODEL COMPARISON\n")
    f.write("=" * 80 + "\n\n")
    f.write("NOTE: AIC cannot compare models with different DVs.\n")
    f.write("This report uses R2, residual variance, and Cohen's f2 instead.\n\n")
    f.write("FIT STATISTICS\n" + "-" * 40 + "\n")
    f.write(fit_df.to_string(index=False))
    f.write("\n\nCOEFFICIENT TABLE\n" + "-" * 40 + "\n")
    for _, row in coef_df.iterrows():
        f.write(f"\n{row['Term']}\n")
        f.write(f"  First-Pass:       b={row['First-Pass RT_b']:.3f}, "
                f"SE={row['First-Pass RT_se']:.3f}, "
                f"z={row['First-Pass RT_z']:.2f}, "
                f"p={row['First-Pass RT_p']:.3f} {row['First-Pass RT_sig']}\n")
        f.write(f"  Regression-Path:  b={row['Regression-Path RT_b']:.3f}, "
                f"SE={row['Regression-Path RT_se']:.3f}, "
                f"z={row['Regression-Path RT_z']:.2f}, "
                f"p={row['Regression-Path RT_p']:.3f} {row['Regression-Path RT_sig']}\n")

print(f"\nResults saved to '{output_path}'")
