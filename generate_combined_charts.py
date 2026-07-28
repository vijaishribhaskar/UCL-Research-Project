# -*- coding: utf-8 -*-
"""
- Left: Chart 1 (Surprisal)
- Middle: Chart 2 (First-Pass Reading Time)
- Right: Chart 3 (Regression-Path Reading Time)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PALETTE = {
    'move':   '#4C72B0',   # blue
    'nomove': '#DD8452',   # orange
}

VERB_ORDER      = ['CP', 'DP', 'unacc']
VERB_LABELS     = {'CP': 'CP verb', 'DP': 'DP verb', 'unacc': 'Unaccusative'}
STRUCTURE_ORDER = ['move', 'nomove']

BAR_WIDTH   = 0.32
GROUP_GAP   = 0.14
GROUP_WIDTH = len(STRUCTURE_ORDER) * BAR_WIDTH + GROUP_GAP
X_POS       = np.arange(len(VERB_ORDER)) * GROUP_WIDTH

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
})

def add_bars(ax, grp, verb_col, struct_col, mean_col, sem_col):
    for i, struct in enumerate(STRUCTURE_ORDER):
        sub = grp[grp[struct_col] == struct].set_index(verb_col)
        means = [sub.loc[v, mean_col] if v in sub.index else 0 for v in VERB_ORDER]
        sems  = [sub.loc[v, sem_col]  if v in sub.index else 0 for v in VERB_ORDER]
        offset = (i - (len(STRUCTURE_ORDER) - 1) / 2) * BAR_WIDTH
        ax.bar(
            X_POS + offset, means, BAR_WIDTH,
            label=struct.capitalize(),
            color=PALETTE[struct],
            edgecolor='white', linewidth=0.6,
            yerr=sems, capsize=4, error_kw={'linewidth': 1.2},
            zorder=3,
        )
    ax.set_xticks(X_POS)
    ax.set_xticklabels([VERB_LABELS[v] for v in VERB_ORDER])
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Structure', loc='upper left')

# Data Loading
print("Loading data for combined visualization...")
df_surp = pd.read_csv('surprisals_Exp3(in).csv')
df_surp.columns = df_surp.columns.str.strip()

df_et = pd.read_csv('critical_RTdata_260319(in).csv')
df_et.columns = df_et.columns.str.strip()
df_verb_region = df_et[df_et['IA_LABEL'] == 'Verb'].copy()

ET_MEASURES = {
    'IA_FIRST_RUN_DWELL_TIME':      'First-Pass Reading Time (ms)',
    'IA_REGRESSION_PATH_DURATION':  'Regression-Path Duration (ms)',
}

# Plotting logic
fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(1, 3, wspace=0.3)

# CHART 1 (Surprisal)
ax1 = fig.add_subplot(gs[0, 0])
grp1 = (
    df_surp
    .groupby(['Verb', 'Structure'])['Surprisal']
    .agg(['mean', 'sem'])
    .reset_index()
)
grp1.columns = ['Verb', 'Structure', 'mean', 'sem']
grp1 = grp1[grp1['Verb'].isin(VERB_ORDER)]

add_bars(ax1, grp1, 'Verb', 'Structure', 'mean', 'sem')
ax1.get_legend().remove()
ax1.set_xlabel('Verb Type')
ax1.set_ylabel('Mean Surprisal (bits)')
ax1.set_title('Mean Surprisal', pad=15)
ax1.text(0.0, 1.05, 'a', transform=ax1.transAxes, fontsize=16, fontweight='normal', va='bottom', ha='left')

# CHART 2 & 3 (Eye-tracking)
axes_et = [
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2])
]

for letter, ax, (measure, label) in zip(['b', 'c'], axes_et, ET_MEASURES.items()):
    grp2 = (
        df_verb_region
        .groupby(['verb', 'structure'])[measure]
        .agg(['mean', 'sem'])
        .reset_index()
    )
    grp2.columns = ['Verb', 'Structure', 'mean', 'sem']
    add_bars(ax, grp2, 'Verb', 'Structure', 'mean', 'sem')
    ax.set_xlabel('Verb Type')
    ax.set_ylabel(label)
    ax.set_title(label, fontsize=11, style='italic')
    ax.get_legend().remove()
    ax.text(0.0, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight='normal', va='bottom', ha='left')

# Add a single legend for the eye-tracking part on the top right plot
axes_et[0].legend(title='Structure', loc='upper left', bbox_to_anchor=(1, 1))

fig.suptitle(
    'Comparison of Linguistic Processing: Surprisal vs. Eye-Tracking Measures',
    fontsize=16, y=0.98, fontweight='normal'
)

fig.subplots_adjust(top=0.82)
plt.savefig('combined_chart_surprisal_et.png', dpi=150, bbox_inches='tight')
print("  Saved -> combined_chart_surprisal_et.png")
plt.close(fig)

print("\nCombined visualization generated successfully.")
