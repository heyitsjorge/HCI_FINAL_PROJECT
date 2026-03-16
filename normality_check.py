"""
DND Study - Normality Check for Paired T-Tests
================================================
Shapiro-Wilk tests on paired differences (DND - Normal) per participant.
Q-Q plots for visual confirmation.

Language: Python 3
Packages: pandas, numpy, scipy, matplotlib, openpyxl

Install dependencies:
    pip install pandas numpy scipy matplotlib openpyxl
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
# matplotlib.use('Agg')  # Remove this line if running in an IDE with display
import matplotlib.pyplot as plt


# =============================================================================
# DATA LOADING (same setup as DND_Stats.py)
# =============================================================================

df = pd.read_excel('Checkins_sheet.xlsx')
df = df[df['Participant ID'] != 'P10'].copy()
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
df.columns = df.columns.str.replace('O', 'o')

col_map = {
    'I felt stressed today.': 'Stressed',
    'I felt overwhelmed today.': 'overwhelmed',
    'I felt in control of my responsibilities.': 'Incontrol',
    'I was able to stay focused on my tasks.': 'Focused',
    'My phone distracted me today.': 'Distracted',
    'It was easy to maintain my attention.': 'Attention',
    'I felt productive today.': 'Productive',
    'I accomplished what I planned to do.': 'Accomplished',
    'My phone interruptions reduced my productivity.': 'PhoneInterruptions',
    'I slept well.': 'SleepQuality',
    'Notifications disturbed my sleep.': 'SleepDisturbed',
}
df.rename(columns=col_map, inplace=True)
likert_vars = list(col_map.values())

df['Date'] = df['Date this entry applies to']
df['Week'] = df['Date'].apply(lambda x: 1 if x.month == 2 else 2)

week1_dnd = ['P01', 'P02', 'P04', 'P06', 'P08', 'P11', 'P14']

def assign_condition(row):
    if row['Participant ID'] in week1_dnd:
        return 'DND' if row['Week'] == 1 else 'Normal'
    else:
        return 'Normal' if row['Week'] == 1 else 'DND'

df['condition'] = df.apply(assign_condition, axis=1)


# =============================================================================
# SHAPIRO-WILK ON PAIRED DIFFERENCES
# =============================================================================

print("=" * 70)
print("SHAPIRO-WILK NORMALITY TESTS")
print("Applied to paired DIFFERENCES (DND - Normal) per participant")
print("This is what matters for paired t-tests")
print("=" * 70)

results = []
for var in likert_vars:
    sub = df[['Participant ID', 'condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'condition'])[var].mean().unstack('condition').dropna()
    diff = means['DND'] - means['Normal']

    w_stat, p_val = stats.shapiro(diff)
    normal = "YES" if p_val > 0.05 else "NO"
    skew = diff.skew()
    kurt = diff.kurtosis()

    results.append({'Variable': var, 'W': w_stat, 'p': p_val,
                    'Normal': normal, 'Skewness': skew, 'Kurtosis': kurt})

    sig = "*" if p_val < .05 else ""
    print(f"\n{var}:")
    print(f"  W = {w_stat:.4f}, p = {p_val:.4f} {sig}")
    print(f"  Skewness = {skew:.3f}, Kurtosis = {kurt:.3f}")
    print(f"  Normality assumption: {normal}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

res_df = pd.DataFrame(results)
violations = res_df[res_df['Normal'] == 'NO']
passed = res_df[res_df['Normal'] == 'YES']

print(f"\nVariables where normality HOLDS (p > .05): {len(passed)}")
for _, r in passed.iterrows():
    print(f"  - {r['Variable']}: p={r['p']:.4f}")

print(f"\nVariables where normality is VIOLATED (p < .05): {len(violations)}")
for _, r in violations.iterrows():
    print(f"  - {r['Variable']}: W={r['W']:.4f}, p={r['p']:.4f}, Skew={r['Skewness']:.3f}")


# =============================================================================
# Q-Q PLOTS
# =============================================================================

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, var in enumerate(likert_vars):
    sub = df[['Participant ID', 'condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'condition'])[var].mean().unstack('condition').dropna()
    diff = means['DND'] - means['Normal']

    ax = axes[i]
    stats.probplot(diff, dist="norm", plot=ax)
    shapiro_p = stats.shapiro(diff)[1]
    status = "PASS" if shapiro_p > 0.05 else "FAIL"
    ax.set_title(f'{var}\n(Shapiro p={shapiro_p:.3f}, {status})', fontsize=9, fontweight='bold')
    ax.get_lines()[0].set_markerfacecolor('#8B6DB5')
    ax.get_lines()[0].set_markersize(6)
    ax.get_lines()[1].set_color('#E8A838')

axes[11].set_visible(False)

plt.suptitle('Q-Q Plots of Paired Differences (DND - Normal)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_qq_normality.png', dpi=200, bbox_inches='tight')
plt.close()
print("\n→ Saved fig_qq_normality.png")