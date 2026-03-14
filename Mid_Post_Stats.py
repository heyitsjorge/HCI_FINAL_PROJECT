"""
DND Study - Mid & Post Survey Statistical Analysis
====================================================
Paired comparisons of DND vs Normal from mid-study and post-study surveys.
Two-factor analysis: Condition (DND/Normal) x Survey Period (Mid/Post).

Language: Python 3
Packages: pandas, numpy, scipy, matplotlib, openpyxl

Install dependencies:
    pip install pandas numpy scipy matplotlib openpyxl
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D



# 1. DATA LOADING & CLEANING

mid = pd.read_excel('midStudyStatsSheet.xlsx')
post = pd.read_excel('postStudyStatsSheet.xlsx')

mid['Survey'] = 'Mid'
post['Survey'] = 'Post'

df = pd.concat([mid, post], ignore_index=True)
df = df[df['Participant ID'] != 'P10'].copy()
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Assign condition from survey response
cond_col = 'Were you on DND or no-DND this week?'
df['Condition'] = df[cond_col].map({'Do Not Disturb': 'DND', 'Not on Do Not Disturb': 'Normal'})

# Encode Likert text to numeric (1-5)
likert_map = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly agree': 5
}

likert_text_cols = [
    'I was able to focus better than usual during this period.',
    'I felt more productive than usual.',
    'My phone interrupted me frequently during this period.',
    'I worried about missing important messages.',
    'The notification setting made it easier to manage my time.',
    'The notification setting was disruptive to my normal routine'
]

for col in likert_text_cols:
    df[col] = df[col].map(likert_map)

# Rename to short names
short_names = {
    'My overall stress level was...': 'Stress',
    'My feeling of being overwhelmed was...': 'Overwhelmed',
    'I was able to focus better than usual during this period.': 'FocusBetter',
    'I felt more productive than usual.': 'MoreProductive',
    'My phone interrupted me frequently during this period.': 'PhoneInterrupted',
    'I worried about missing important messages.': 'MissedMessages',
    'The notification setting made it easier to manage my time.': 'EasierTimeManage',
    'The notification setting was disruptive to my normal routine': 'DisruptiveRoutine',
}
df.rename(columns=short_names, inplace=True)

analysis_vars = list(short_names.values())

var_labels = ['Stress', 'Overwhelmed', 'Focus\nBetter', 'More\nProductive',
              'Phone\nInterrupted', 'Missed\nMessages', 'Easier Time\nManagement',
              'Disruptive\nRoutine']

print(f"Total rows: {len(df)}")
print(f"Participants (N={df['Participant ID'].nunique()}): {sorted(df['Participant ID'].unique())}")
print(f"\nCondition x Survey:\n{pd.crosstab(df['Condition'], df['Survey'])}\n")


# =============================================================================
# 2. PAIRED T-TESTS: DND vs Normal
#    Each participant filled out one survey under DND and one under Normal.
#    This is the primary within-subjects comparison.
# =============================================================================

print("=" * 70)
print("PAIRED T-TESTS: DND vs Normal (within-subjects)")
print("=" * 70)

ttest_results = []

for var in analysis_vars:
    dnd_scores = []
    norm_scores = []
    for pid in sorted(df['Participant ID'].unique()):
        pid_data = df[df['Participant ID'] == pid]
        dnd_val = pid_data[pid_data['Condition'] == 'DND'][var].dropna()
        norm_val = pid_data[pid_data['Condition'] == 'Normal'][var].dropna()
        if len(dnd_val) > 0 and len(norm_val) > 0:
            dnd_scores.append(dnd_val.values[0])
            norm_scores.append(norm_val.values[0])

    dnd_arr = np.array(dnd_scores, dtype=float)
    norm_arr = np.array(norm_scores, dtype=float)
    diff = dnd_arr - norm_arr

    t_stat, p_val = stats.ttest_rel(dnd_arr, norm_arr)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    nonzero = diff[diff != 0]
    w_stat, w_p = (np.nan, np.nan)
    if len(nonzero) >= 5:
        w_stat, w_p = stats.wilcoxon(nonzero)

    ttest_results.append({
        'Variable': var, 'DND_M': dnd_arr.mean(), 'DND_SD': dnd_arr.std(),
        'Normal_M': norm_arr.mean(), 'Normal_SD': norm_arr.std(),
        'Diff': diff.mean(), 't': t_stat, 'df': len(dnd_arr) - 1,
        'p': p_val, 'Cohens_d': cohens_d, 'W': w_stat, 'W_p': w_p,
        'n': len(dnd_arr)
    })

    sig = "***" if p_val < .001 else "**" if p_val < .01 else "*" if p_val < .05 else "†" if p_val < .10 else ""
    print(f"\n{var}:")
    print(f"  DND:    M={dnd_arr.mean():.2f} (SD={dnd_arr.std():.2f})")
    print(f"  Normal: M={norm_arr.mean():.2f} (SD={norm_arr.std():.2f})")
    print(f"  t({len(dnd_arr)-1})={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f} {sig}")
    if not np.isnan(w_p):
        w_sig = "*" if w_p < .05 else "†" if w_p < .10 else ""
        print(f"  Wilcoxon: W={w_stat:.1f}, p={w_p:.4f} {w_sig}")

ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv('mid_post_ttest_results.csv', index=False)
print("\n→ Results saved to mid_post_ttest_results.csv")


# =============================================================================
# 3. TWO-FACTOR ANALYSIS: Condition (DND/Normal) x Period (Mid/Post)
# =============================================================================

print("\n" + "=" * 70)
print("TWO-FACTOR ANALYSIS: Condition x Survey Period (2x2)")
print("=" * 70)

twofactor_results = []

for var in analysis_vars:
    dnd_mid = df[(df['Condition'] == 'DND') & (df['Survey'] == 'Mid')][var].dropna()
    dnd_post = df[(df['Condition'] == 'DND') & (df['Survey'] == 'Post')][var].dropna()
    norm_mid = df[(df['Condition'] == 'Normal') & (df['Survey'] == 'Mid')][var].dropna()
    norm_post = df[(df['Condition'] == 'Normal') & (df['Survey'] == 'Post')][var].dropna()

    # Main effect of Condition
    all_dnd = np.concatenate([dnd_mid.values, dnd_post.values])
    all_norm = np.concatenate([norm_mid.values, norm_post.values])
    t_cond, p_cond = stats.ttest_ind(all_dnd, all_norm)

    # Main effect of Period
    all_mid = np.concatenate([dnd_mid.values, norm_mid.values])
    all_post = np.concatenate([dnd_post.values, norm_post.values])
    t_per, p_per = stats.ttest_ind(all_mid, all_post)

    # Interaction estimate
    dnd_effect_mid = dnd_mid.mean() - norm_mid.mean()
    dnd_effect_post = dnd_post.mean() - norm_post.mean()
    interaction = dnd_effect_mid - dnd_effect_post

    twofactor_results.append({
        'Variable': var,
        'DND_Mid_M': dnd_mid.mean(), 'DND_Post_M': dnd_post.mean(),
        'Norm_Mid_M': norm_mid.mean(), 'Norm_Post_M': norm_post.mean(),
        't_Condition': t_cond, 'p_Condition': p_cond,
        't_Period': t_per, 'p_Period': p_per,
        'Interaction': interaction
    })

    c_sig = "***" if p_cond < .001 else "**" if p_cond < .01 else "*" if p_cond < .05 else "†" if p_cond < .10 else ""
    p_sig = "***" if p_per < .001 else "**" if p_per < .01 else "*" if p_per < .05 else "†" if p_per < .10 else ""
    print(f"\n{var}:")
    print(f"  DND-Mid: {dnd_mid.mean():.2f}  |  DND-Post: {dnd_post.mean():.2f}")
    print(f"  Norm-Mid: {norm_mid.mean():.2f} |  Norm-Post: {norm_post.mean():.2f}")
    print(f"  Main effect Condition: t={t_cond:.3f}, p={p_cond:.4f} {c_sig}")
    print(f"  Main effect Period:    t={t_per:.3f}, p={p_per:.4f} {p_sig}")
    print(f"  Interaction (DND effect mid - post): {interaction:+.2f}")

twofactor_df = pd.DataFrame(twofactor_results)
twofactor_df.to_csv('mid_post_twofactor_results.csv', index=False)
print("\n→ Results saved to mid_post_twofactor_results.csv")


# =============================================================================
# 4. NOTIFICATION PREFERENCE ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("NOTIFICATION PREFERENCE ANALYSIS")
print("=" * 70)

pref_col = 'Based on your experience so far, which notification setting would you prefer for your daily life?'

print("\nMid-Study Preferences:")
print(df[df['Survey'] == 'Mid'][pref_col].value_counts().to_string())
print("\nPost-Study Preferences:")
print(df[df['Survey'] == 'Post'][pref_col].value_counts().to_string())

print("\nPreference by Condition (Mid):")
print(pd.crosstab(df[df['Survey'] == 'Mid']['Condition'], df[df['Survey'] == 'Mid'][pref_col]).to_string())
print("\nPreference by Condition (Post):")
print(pd.crosstab(df[df['Survey'] == 'Post']['Condition'], df[df['Survey'] == 'Post'][pref_col]).to_string())

print("\nPreference Shifts (Mid → Post per participant):")
for pid in sorted(df['Participant ID'].unique()):
    mid_pref = df[(df['Participant ID'] == pid) & (df['Survey'] == 'Mid')][pref_col].values
    post_pref = df[(df['Participant ID'] == pid) & (df['Survey'] == 'Post')][pref_col].values
    if len(mid_pref) > 0 and len(post_pref) > 0:
        changed = " ← CHANGED" if mid_pref[0] != post_pref[0] else ""
        print(f"  {pid}: {mid_pref[0]:20s} → {post_pref[0]:20s}{changed}")


# =============================================================================
# 5. FIGURES
# =============================================================================

DND_COLOR = '#8B6DB5'
NORM_COLOR = '#E8A838'


def add_bracket(ax, x_pos, y, p_val, width=0.32):
    if p_val >= 0.10:
        return
    label = '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else '†'
    x1 = x_pos - width / 2
    x2 = x_pos + width / 2
    bracket_y = y + 0.08
    text_y = bracket_y + 0.02
    ax.plot([x1, x1, x2, x2], [bracket_y - 0.05, bracket_y, bracket_y, bracket_y - 0.05],
            color='black', linewidth=1.2)
    ax.text((x1 + x2) / 2, text_y, label, ha='center', va='bottom', fontsize=13, fontweight='bold')


# --- Figure 5: DND vs Normal paired comparison (survey items) ---
fig, ax = plt.subplots(figsize=(15, 8))

dnd_means = [ttest_results[i]['DND_M'] for i in range(len(analysis_vars))]
norm_means = [ttest_results[i]['Normal_M'] for i in range(len(analysis_vars))]
dnd_sds = [ttest_results[i]['DND_SD'] for i in range(len(analysis_vars))]
norm_sds = [ttest_results[i]['Normal_SD'] for i in range(len(analysis_vars))]
n = ttest_results[0]['n']
dnd_sems = [sd / np.sqrt(n) for sd in dnd_sds]
norm_sems = [sd / np.sqrt(n) for sd in norm_sds]
p_vals = [ttest_results[i]['p'] for i in range(len(analysis_vars))]

x = np.arange(len(analysis_vars))
width = 0.32

ax.bar(x - width / 2, dnd_means, width, yerr=dnd_sems, color=DND_COLOR,
       capsize=4, alpha=0.92, edgecolor='white', linewidth=0.5,
       error_kw={'linewidth': 1.2, 'capthick': 1.2})
ax.bar(x + width / 2, norm_means, width, yerr=norm_sems, color=NORM_COLOR,
       capsize=4, alpha=0.92, edgecolor='white', linewidth=0.5,
       error_kw={'linewidth': 1.2, 'capthick': 1.2})

ax.set_ylabel('Mean Rating (1–5)', fontsize=13, fontweight='bold')
ax.set_title('Mid & Post Survey Ratings: DND vs Normal Condition', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(var_labels, fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 5.8)
ax.set_xlim(-0.6, len(analysis_vars) - 0.4)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.tick_params(axis='y', labelsize=10)
ax.yaxis.set_ticks(np.arange(0, 6, 1))
ax.yaxis.grid(True, alpha=0.15, linewidth=0.5)

for i in range(len(analysis_vars)):
    max_h = max(dnd_means[i] + dnd_sems[i], norm_means[i] + norm_sems[i])
    add_bracket(ax, x[i], max_h, p_vals[i])

legend_elements = [mpatches.Patch(facecolor=DND_COLOR, alpha=0.92, label='DND'),
                   mpatches.Patch(facecolor=NORM_COLOR, alpha=0.92, label='Normal')]
ax.legend(handles=legend_elements, fontsize=12, loc='upper right', framealpha=0.9,
          edgecolor='gray', fancybox=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

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
        cell.set_facecolor('#E8DFF5' if row in [0, 1] else '#FDF0D5')
    else:
        cell.set_facecolor('#F5F0FA' if row in [0, 1] else '#FFF8EC')

plt.savefig('fig5_survey_dnd_vs_normal.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig5_survey_dnd_vs_normal.png")


# --- Figure 6: Two-factor interaction plot (Condition x Period) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, var in enumerate(analysis_vars):
    ax = axes[i]
    r = twofactor_results[i]

    # Plot lines: DND across periods, Normal across periods
    ax.plot(['Mid', 'Post'], [r['DND_Mid_M'], r['DND_Post_M']],
            'o-', color=DND_COLOR, markersize=10, linewidth=2.5,
            markeredgecolor='white', markeredgewidth=1.5, label='DND', zorder=5)
    ax.plot(['Mid', 'Post'], [r['Norm_Mid_M'], r['Norm_Post_M']],
            's-', color=NORM_COLOR, markersize=10, linewidth=2.5,
            markeredgecolor='white', markeredgewidth=1.5, label='Normal', zorder=5)

    c_sig = '***' if r['p_Condition'] < .001 else '**' if r['p_Condition'] < .01 else '*' if r['p_Condition'] < .05 else '†' if r['p_Condition'] < .10 else 'n.s.'
    ax.set_title(f'{var_labels[i].replace(chr(10), " ")}\nCond: p={r["p_Condition"]:.3f} ({c_sig})',
                 fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.5, 5.5)
    ax.set_ylabel('Mean (1–5)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15)

    if i == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Two-Factor Interaction: Condition × Survey Period', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_twofactor_interaction.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig6_twofactor_interaction.png")


# --- Figure 7: Effect sizes forest plot (survey data) ---
fig, ax = plt.subplots(figsize=(10, 6))

ds = [r['Cohens_d'] for r in ttest_results]
ps = [r['p'] for r in ttest_results]

y_pos = np.arange(len(analysis_vars))
colors = ['#7B4FA0' if p < .05 else NORM_COLOR if p < .10 else '#C0C0C0' for p in ps]

ax.barh(y_pos, ds, color=colors, alpha=0.88, height=0.6, edgecolor='white', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels([l.replace('\n', ' ') for l in var_labels], fontsize=10)
ax.set_xlabel("Cohen's d  (positive = higher under DND)", fontsize=12, fontweight='bold')
ax.set_title("Effect Sizes: Mid & Post Surveys (DND vs Normal)", fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.xaxis.grid(True, alpha=0.15)

for i, (d, p) in enumerate(zip(ds, ps)):
    offset = 0.05 if d >= 0 else -0.05
    ha = 'left' if d >= 0 else 'right'
    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .10 else ''
    ax.text(d + offset, i, f'{d:+.2f} {sig}', va='center', ha=ha, fontsize=9.5, fontweight='bold')

legend_elements = [mpatches.Patch(facecolor='#7B4FA0', alpha=0.88, label='p < .05'),
                   mpatches.Patch(facecolor=NORM_COLOR, alpha=0.88, label='p < .10'),
                   mpatches.Patch(facecolor='#C0C0C0', alpha=0.88, label='n.s.')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.savefig('fig7_survey_effectsizes.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig7_survey_effectsizes.png")


# --- Figure 8: Preference shift alluvial/bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mid preferences
pref_col = 'Based on your experience so far, which notification setting would you prefer for your daily life?'
pref_order = ['Prefer DND ON', 'No preference', 'Prefer DND OFF']
pref_colors = ['#8B6DB5', '#C0C0C0', '#E8A838']

for idx, (survey, title) in enumerate([('Mid', 'Mid-Study Preferences'), ('Post', 'Post-Study Preferences')]):
    ax = axes[idx]
    counts = df[df['Survey'] == survey][pref_col].value_counts()
    vals = [counts.get(p, 0) for p in pref_order]

    bars = ax.bar(range(len(pref_order)), vals, color=pref_colors, alpha=0.9,
                  edgecolor='white', linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(pref_order)))
    ax.set_xticklabels(['Prefer\nDND ON', 'No\nPreference', 'Prefer\nDND OFF'],
                       fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Participants', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks(range(0, 9))

    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    str(val), ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Notification Preference: Mid vs Post Study', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig8_preferences.png', dpi=250, bbox_inches='tight')
plt.close()
print("→ Saved fig8_preferences.png")


print("\n✓ Mid/Post analysis complete. Output files:")
print("  - mid_post_ttest_results.csv")
print("  - mid_post_twofactor_results.csv")
print("  - fig5_survey_dnd_vs_normal.png")
print("  - fig6_twofactor_interaction.png")
print("  - fig7_survey_effectsizes.png")
print("  - fig8_preferences.png")