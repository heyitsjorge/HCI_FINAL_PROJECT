# DND Statitcal Analysis

"""
DND Study Statistical Analysis
Crossover design: DND vs Normal notification settings
10-day study (5 days per condition) with N=13 college students
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
# matplotlib.use('Agg')  # Remove this line if running in an IDE with display
import matplotlib.pyplot as plt


# 1. DATA LOADING & CLEANING
df = pd.read_excel('Checkins_sheet.xlsx')

# Cleaning Some White Spaces in Column Names
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Exclude P10 (insufficient survey completion)
df = df[df['Participant ID'] != 'P10'].copy()

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Rename columns for convenience
col_map = {
    'I felt stressed today.': 'Stressed',
    'I felt overwhelmed today.': 'Overwhelmed',
    'I felt in control of my responsibilities.': 'InControl',
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
print(" \n Columns after renaming:")
print(df.columns.tolist())
likert_vars = list(col_map.values())


# 2. CONDITION ASSIGNMENT (Crossover Design)

# Week 1 = Feb 23-27, Week 2 = Mar 2-6
df['Date'] = df['Date this entry applies to']
df['Week'] = df['Date'].apply(lambda x: 1 if x.month == 2 else 2)

# Participants who had DND in Week 1 (switch to Normal in Week 2)
week1_dnd = ['P01', 'P02', 'P04', 'P06', 'P08', 'P11', 'P14']

def assign_condition(row):
    if row['Participant ID'] in week1_dnd:
        return 'DND' if row['Week'] == 1 else 'Normal'
    else:
        return 'Normal' if row['Week'] == 1 else 'DND'

df['Condition'] = df.apply(assign_condition, axis=1)

print(f"Total observations: {len(df)}")
print(f"Participants (N={df['Participant ID'].nunique()}): {sorted(df['Participant ID'].unique())}")
print(f"Condition counts:\n{df['Condition'].value_counts()}\n")


# 3. DESCRIPTIVE STATISTICS BY CONDITION
print("=" * 70)
print("DESCRIPTIVE STATISTICS BY CONDITION")
print("=" * 70)

for var in likert_vars:
    dnd = df[df['Condition'] == 'DND'][var].dropna()
    norm = df[df['Condition'] == 'Normal'][var].dropna()
    print(f"\n{var}:")
    print(f"  DND:    M={dnd.mean():.2f}, SD={dnd.std():.2f}, n={len(dnd)}")
    print(f"  Normal: M={norm.mean():.2f}, SD={norm.std():.2f}, n={len(norm)}")


# 4. PAIRED T-TESTS (Primary Analysis)
#    Standard crossover approach: average each participant's scores per
#    condition, then run paired comparisons.

print("\n" + "=" * 70)
print("PAIRED T-TESTS (Within-Subject: DND vs Normal)")
print("=" * 70)

ttest_results = []

for var in likert_vars:
    sub = df[['Participant ID', 'Condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[var].mean().unstack('Condition')
    means = means.dropna()  # Keep only participants with both conditions

    dnd_means = means['DND']
    norm_means = means['Normal']
    diff = dnd_means - norm_means

    t_stat, p_val = stats.ttest_rel(dnd_means, norm_means)
    cohens_d = diff.mean() / diff.std()

    ttest_results.append({
        'Variable': var,
        'DND_M': dnd_means.mean(),
        'DND_SD': dnd_means.std(),
        'Normal_M': norm_means.mean(),
        'Normal_SD': norm_means.std(),
        'Diff': diff.mean(),
        't': t_stat,
        'df': len(means) - 1,
        'p': p_val,
        'Cohens_d': cohens_d,
        'n_pairs': len(means)
    })

    sig = "***" if p_val < .001 else "**" if p_val < .01 else "*" if p_val < .05 else "†" if p_val < .10 else ""
    print(f"\n{var}:")
    print(f"  DND:    M={dnd_means.mean():.2f} (SD={dnd_means.std():.2f})")
    print(f"  Normal: M={norm_means.mean():.2f} (SD={norm_means.std():.2f})")
    print(f"  Diff={diff.mean():.2f}, t({len(means)-1})={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f} {sig}")

ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv('paired_ttest_results.csv', index=False)
print("\n→ Results saved to paired_ttest_results.csv")


# 5. WILCOXON SIGNED-RANK TESTS (Non-Parametric Robustness Check)

print("\n" + "=" * 70)
print("WILCOXON SIGNED-RANK TESTS (Non-parametric robustness check)")
print("=" * 70)

for var in likert_vars:
    sub = df[['Participant ID', 'Condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[var].mean().unstack('Condition').dropna()
    diff = means['DND'] - means['Normal']

    nonzero = diff[diff != 0]
    if len(nonzero) >= 5:
        w_stat, w_p = stats.wilcoxon(nonzero)
        sig = "*" if w_p < .05 else "†" if w_p < .10 else ""
        print(f"{var}: W={w_stat:.1f}, p={w_p:.4f} {sig} (n_nonzero={len(nonzero)})")
    else:
        print(f"{var}: Too few non-zero differences ({len(nonzero)})")


# 6. PERIOD / CARRYOVER EFFECT TESTS

print("\n" + "=" * 70)
print("PERIOD EFFECTS CHECK (Week 1 vs Week 2)")
print("=" * 70)

for var in likert_vars:
    sub = df[['Week', var]].dropna()
    w1 = sub[sub['Week'] == 1][var]
    w2 = sub[sub['Week'] == 2][var]
    t_stat, p_val = stats.ttest_ind(w1, w2)
    print(f"{var}: Week1 M={w1.mean():.2f}, Week2 M={w2.mean():.2f}, t={t_stat:.3f}, p={p_val:.4f}")


# 7. COMPOSITE SCORES
#    Reverse-code negatively-worded items (6 - score), then average within domain

print("\n" + "=" * 70)
print("COMPOSITE SCORE ANALYSES")
print("=" * 70)

# Stress: Stressed + Overwhelmed + reverse(InControl) → higher = more stressed
df['Stress_Composite'] = (df['Stressed'] + df['Overwhelmed'] + (6 - df['InControl'])) / 3

# Focus: Focused + reverse(Distracted) + Attention + reverse(PhoneInterruptions) → higher = better focus
df['Focus_Composite'] = (df['Focused'] + (6 - df['Distracted']) + df['Attention'] + (6 - df['PhoneInterruptions'])) / 4

# Productivity: Productive + Accomplished → higher = more productive
df['Productivity_Composite'] = (df['Productive'] + df['Accomplished']) / 2

# Sleep: SleepQuality + reverse(SleepDisturbed) → higher = better sleep
df['Sleep_Composite'] = (df['SleepQuality'] + (6 - df['SleepDisturbed'])) / 2

composites = ['Stress_Composite', 'Focus_Composite', 'Productivity_Composite', 'Sleep_Composite']
composite_labels = ['Stress (higher=more stressed)', 'Focus (higher=better focus)',
                    'Productivity (higher=more productive)', 'Sleep (higher=better sleep)']

for comp, label in zip(composites, composite_labels):
    sub = df[['Participant ID', 'Condition', comp]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[comp].mean().unstack('Condition').dropna()
    dnd_m = means['DND']
    norm_m = means['Normal']
    diff = dnd_m - norm_m
    t_stat, p_val = stats.ttest_rel(dnd_m, norm_m)
    cohens_d = diff.mean() / diff.std()
    sig = "*" if p_val < .05 else "†" if p_val < .10 else ""
    print(f"\n{label}:")
    print(f"  DND:    M={dnd_m.mean():.2f} (SD={dnd_m.std():.2f})")
    print(f"  Normal: M={norm_m.mean():.2f} (SD={norm_m.std():.2f})")
    print(f"  t({len(means)-1})={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f} {sig}")


# 8. EFFECT SIZE SUMMARY (All variables ranked by |d|)

print("\n" + "=" * 70)
print("EFFECT SIZE SUMMARY (ranked by |Cohen's d|)")
print("=" * 70)

all_vars = likert_vars + composites
effect_results = []

for var in all_vars:
    sub = df[['Participant ID', 'Condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[var].mean().unstack('Condition').dropna()
    diff = means['DND'] - means['Normal']
    t_stat, p_val = stats.ttest_rel(means['DND'], means['Normal'])
    cohens_d = diff.mean() / diff.std()
    effect_results.append({'Variable': var, 'd': cohens_d, 'p': p_val,
                           'DND_M': means['DND'].mean(), 'Normal_M': means['Normal'].mean()})

effect_df = pd.DataFrame(effect_results).sort_values('d', key=abs, ascending=False)
for _, r in effect_df.iterrows():
    sig = "***" if r['p'] < .001 else "**" if r['p'] < .01 else "*" if r['p'] < .05 else "†" if r['p'] < .10 else ""
    direction = "DND higher" if r['d'] > 0 else "Normal higher"
    print(f"  d={r['d']:+.3f}  p={r['p']:.4f} {sig:3s}  {r['Variable']:<25s}  ({direction})")

effect_df.to_csv('effect_sizes_summary.csv', index=False)
print("\n→ Effect sizes saved to effect_sizes_summary.csv")


# 9. COMPLIANCE CHECK

print("\n" + "=" * 70)
print("COMPLIANCE")
print("=" * 70)

compliance_col = 'Did you keep your assigned notification setting for most of today (>90%)?'
print(df[compliance_col].value_counts())
partial = df[df[compliance_col] == 'Partially']
print(f"\nPartial compliance entries:")
print(partial[['Participant ID', 'Date', 'Condition']].to_string(index=False))



# 10. FIGURES
# Updated with publication-style brackets, data tables, and purple/orange palette

DND_COLOR = '#8B6DB5'    # darker pastel purple
NORM_COLOR = '#E8A838'   # warm golden orange


def add_bracket(ax, x_pos, y, p_val, width=0.32):
    """Draw a significance bracket between a pair of bars."""
    if p_val >= 0.10:
        return
    label = '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else '†'
    x1 = x_pos - width/2
    x2 = x_pos + width/2
    bracket_y = y + 0.08
    text_y = bracket_y + 0.02
    ax.plot([x1, x1, x2, x2], [bracket_y - 0.05, bracket_y, bracket_y, bracket_y - 0.05],
            color='black', linewidth=1.2)
    ax.text((x1 + x2) / 2, text_y, label, ha='center', va='bottom', fontsize=13, fontweight='bold')


# --- Figure 1: All items comparison with brackets and data table ---
fig, ax = plt.subplots(figsize=(15, 8))

var_labels = ['Stressed', 'Overwhelmed', 'In Control', 'Focused', 'Distracted',
              'Attention', 'Productive', 'Accomplished', 'Phone\nInterruptions',
              'Sleep\nQuality', 'Sleep\nDisturbed']

dnd_means = [df[df['Condition'] == 'DND'][v].mean() for v in likert_vars]
norm_means = [df[df['Condition'] == 'Normal'][v].mean() for v in likert_vars]
dnd_sems = [df[df['Condition'] == 'DND'][v].sem() for v in likert_vars]
norm_sems = [df[df['Condition'] == 'Normal'][v].sem() for v in likert_vars]

x = np.arange(len(likert_vars))
width = 0.32

# Compute p-values for brackets
p_values = []
for var in likert_vars:
    sub = df[['Participant ID', 'Condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[var].mean().unstack('Condition').dropna()
    t, p = stats.ttest_rel(means['DND'], means['Normal'])
    p_values.append(p)

ax.bar(x - width/2, dnd_means, width, yerr=dnd_sems, color=DND_COLOR,
       capsize=4, alpha=0.92, edgecolor='white', linewidth=0.5,
       error_kw={'linewidth': 1.2, 'capthick': 1.2})
ax.bar(x + width/2, norm_means, width, yerr=norm_sems, color=NORM_COLOR,
       capsize=4, alpha=0.92, edgecolor='white', linewidth=0.5,
       error_kw={'linewidth': 1.2, 'capthick': 1.2})

ax.set_ylabel('Mean Rating (1–5)', fontsize=13, fontweight='bold')
ax.set_title('Daily Check-In Ratings: DND vs Normal Condition', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(var_labels, fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 5.8)
ax.set_xlim(-0.6, len(likert_vars) - 0.4)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.tick_params(axis='y', labelsize=10)
ax.yaxis.set_ticks(np.arange(0, 6, 1))
ax.yaxis.grid(True, alpha=0.15, linewidth=0.5)

# Significance brackets
for i in range(len(likert_vars)):
    max_h = max(dnd_means[i] + dnd_sems[i], norm_means[i] + norm_sems[i])
    add_bracket(ax, x[i], max_h, p_values[i])

# Legend
import matplotlib.patches as mpatches
legend_elements = [mpatches.Patch(facecolor=DND_COLOR, alpha=0.92, label='DND'),
                   mpatches.Patch(facecolor=NORM_COLOR, alpha=0.92, label='Normal')]
ax.legend(handles=legend_elements, fontsize=12, loc='upper right', framealpha=0.9,
          edgecolor='gray', fancybox=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

# Data table below chart
table_data = [
    [f'{m:.2f}' for m in dnd_means],
    [f'{s:.3f}' for s in dnd_sems],
    [f'{m:.2f}' for m in norm_means],
    [f'{s:.3f}' for s in norm_sems],
]
row_labels = ['DND  M', 'SE', 'Normal  M', 'SE']

table = ax.table(cellText=table_data, rowLabels=row_labels,
                 colLabels=None, cellLoc='center', rowLoc='center',
                 bbox=[0.0, -0.38, 1.0, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(8.5)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)
    if col == -1:
        cell.set_fontsize(8.5)
        cell.set_text_props(fontweight='bold')
        if row in [0, 1]:
            cell.set_facecolor('#E8DFF5')
        else:
            cell.set_facecolor('#FDF0D5')
    else:
        if row in [0, 1]:
            cell.set_facecolor('#F5F0FA')
        else:
            cell.set_facecolor('#FFF8EC')

plt.savefig('fig1_conditions_comparison.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig1_conditions_comparison.png")


# --- Figure 2: Composite scores with brackets and data table ---
comp_labels_short = ['Stress', 'Focus', 'Productivity', 'Sleep']
comp_subtitles = ['(higher = more stressed)', '(higher = better focus)',
                  '(higher = more productive)', '(higher = better sleep)']

comp_dnd_m = []
comp_norm_m = []
comp_dnd_se = []
comp_norm_se = []
comp_p = []

for comp in composites:
    sub = df[['Participant ID', 'Condition', comp]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[comp].mean().unstack('Condition').dropna()
    t, p = stats.ttest_rel(means['DND'], means['Normal'])
    comp_dnd_m.append(means['DND'].mean())
    comp_norm_m.append(means['Normal'].mean())
    comp_dnd_se.append(means['DND'].sem())
    comp_norm_se.append(means['Normal'].sem())
    comp_p.append(p)

fig, ax = plt.subplots(figsize=(10, 7))
x = np.arange(len(composites))
width = 0.32

ax.bar(x - width/2, comp_dnd_m, width, yerr=comp_dnd_se, color=DND_COLOR,
       capsize=5, alpha=0.92, edgecolor='white', linewidth=0.5,
       error_kw={'linewidth': 1.2, 'capthick': 1.2})
ax.bar(x + width/2, comp_norm_m, width, yerr=comp_norm_se, color=NORM_COLOR,
       capsize=5, alpha=0.92, edgecolor='white', linewidth=0.5,
       error_kw={'linewidth': 1.2, 'capthick': 1.2})

ax.set_ylabel('Mean Composite Score (1–5)', fontsize=13, fontweight='bold')
ax.set_title('Composite Scores: DND vs Normal Condition', fontsize=15, fontweight='bold', pad=20)
label_strs = [f'{l}\n{s}' for l, s in zip(comp_labels_short, comp_subtitles)]
ax.set_xticks(x)
ax.set_xticklabels(label_strs, fontsize=10, fontweight='bold')
ax.set_ylim(0, 5.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.yaxis.set_ticks(np.arange(0, 6, 1))
ax.yaxis.grid(True, alpha=0.15, linewidth=0.5)

for i in range(len(composites)):
    max_h = max(comp_dnd_m[i] + comp_dnd_se[i], comp_norm_m[i] + comp_norm_se[i])
    add_bracket(ax, x[i], max_h, comp_p[i])

legend_elements = [mpatches.Patch(facecolor=DND_COLOR, alpha=0.92, label='DND'),
                   mpatches.Patch(facecolor=NORM_COLOR, alpha=0.92, label='Normal')]
ax.legend(handles=legend_elements, fontsize=12, loc='upper right', framealpha=0.9,
          edgecolor='gray', fancybox=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

table_data2 = [
    [f'{m:.2f}' for m in comp_dnd_m],
    [f'{s:.3f}' for s in comp_dnd_se],
    [f'{m:.2f}' for m in comp_norm_m],
    [f'{s:.3f}' for s in comp_norm_se],
]
row_labels2 = ['DND  M', 'SE', 'Normal  M', 'SE']

table2 = ax.table(cellText=table_data2, rowLabels=row_labels2,
                  colLabels=None, cellLoc='center', rowLoc='center',
                  bbox=[0.0, -0.32, 1.0, 0.20])
table2.auto_set_font_size(False)
table2.set_fontsize(10)

for (row, col), cell in table2.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)
    if col == -1:
        cell.set_fontsize(10)
        cell.set_text_props(fontweight='bold')
        if row in [0, 1]:
            cell.set_facecolor('#E8DFF5')
        else:
            cell.set_facecolor('#FDF0D5')
    else:
        if row in [0, 1]:
            cell.set_facecolor('#F5F0FA')
        else:
            cell.set_facecolor('#FFF8EC')

plt.savefig('fig2_composites.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig2_composites.png")


# --- Figure 3: Individual spaghetti plots (updated colors) ---
from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
key_vars = ['PhoneInterruptions', 'Distracted', 'SleepQuality', 'Accomplished']
key_labels = ['Phone Interruptions\nReduced Productivity', 'Phone Distracted Me',
              'Sleep Quality', 'Accomplished What I Planned']

for idx, (var, label) in enumerate(zip(key_vars, key_labels)):
    ax = axes[idx // 2][idx % 2]
    sub = df[['Participant ID', 'Condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[var].mean().unstack('Condition').dropna()
    t, p = stats.ttest_rel(means['DND'], means['Normal'])

    for pid in means.index:
        dnd_val = means.loc[pid, 'DND']
        norm_val = means.loc[pid, 'Normal']
        color = '#B89DD6' if dnd_val < norm_val else '#F0C87A'
        ax.plot(['DND', 'Normal'], [dnd_val, norm_val],
                'o-', color=color, alpha=0.5, markersize=6, linewidth=1.5)

    ax.plot(0, means['DND'].mean(), 's', color=DND_COLOR, markersize=14,
            zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(1, means['Normal'].mean(), 's', color=NORM_COLOR, markersize=14,
            zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot([0, 1], [means['DND'].mean(), means['Normal'].mean()],
            '-', color='black', linewidth=2.5, zorder=4)

    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .10 else 'n.s.'
    ax.set_title(f'{label}  (p = {p:.3f}, {sig})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Rating (1–5)', fontsize=10)
    ax.set_ylim(0.5, 5.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.yaxis.grid(True, alpha=0.15)

    legend_els = [Line2D([0], [0], marker='s', color='w', markerfacecolor=DND_COLOR, markersize=10, label='DND Mean'),
                  Line2D([0], [0], marker='s', color='w', markerfacecolor=NORM_COLOR, markersize=10, label='Normal Mean'),
                  Line2D([0], [0], color='gray', alpha=0.5, linewidth=1.5, label='Individual')]
    ax.legend(handles=legend_els, fontsize=8, loc='upper right')

plt.suptitle('Individual Participant Trajectories: DND vs Normal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_individual.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig3_individual.png")


# --- Figure 4: Effect size forest plot (updated colors) ---
fig, ax = plt.subplots(figsize=(10, 7.5))

all_labels = ['Stressed', 'Overwhelmed', 'In Control', 'Focused', 'Distracted',
              'Attention', 'Productive', 'Accomplished', 'Phone Interruptions',
              'Sleep Quality', 'Sleep Disturbed',
              'STRESS (composite)', 'FOCUS (composite)', 'PRODUCTIVITY (composite)', 'SLEEP (composite)']

ds = []
ps = []
for var in all_vars:
    sub = df[['Participant ID', 'Condition', var]].dropna()
    means = sub.groupby(['Participant ID', 'Condition'])[var].mean().unstack('Condition').dropna()
    diff = means['DND'] - means['Normal']
    d = diff.mean() / diff.std()
    t, p = stats.ttest_rel(means['DND'], means['Normal'])
    ds.append(d)
    ps.append(p)

y_pos = np.arange(len(all_vars))
colors = ['#7B4FA0' if p < .05 else NORM_COLOR if p < .10 else '#C0C0C0' for p in ps]

ax.barh(y_pos, ds, color=colors, alpha=0.88, height=0.6, edgecolor='white', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(all_labels, fontsize=10)
ax.set_xlabel("Cohen's d  (positive = higher under DND)", fontsize=12, fontweight='bold')
ax.set_title("Effect Sizes: DND vs Normal Condition", fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.xaxis.grid(True, alpha=0.15)

for i, (d, p) in enumerate(zip(ds, ps)):
    offset = 0.03 if d >= 0 else -0.03
    ha = 'left' if d >= 0 else 'right'
    sig = '*' if p < .05 else '†' if p < .10 else ''
    ax.text(d + offset, i, f'{d:+.2f} {sig}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Separator between individual items and composites
ax.axhline(y=10.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

legend_elements = [mpatches.Patch(facecolor='#7B4FA0', alpha=0.88, label='p < .05'),
                   mpatches.Patch(facecolor=NORM_COLOR, alpha=0.88, label='p < .10'),
                   mpatches.Patch(facecolor='#C0C0C0', alpha=0.88, label='n.s.')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.savefig('fig4_effectsizes.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig4_effectsizes.png")

print("\n✓ Analysis complete. Output files:")
print("  - paired_ttest_results.csv")
print("  - effect_sizes_summary.csv")
print("  - fig1_conditions_comparison.png")
print("  - fig2_composites.png")
print("  - fig3_individual.png")
print("  - fig4_effectsizes.png")