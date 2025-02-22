"""
We manually annotated outliers on the plot in the paper, this just plots the data.

We use jittering to make the scatter plot not overlap too much, so your dots might not be at the same y-value but they
should have the same x-value.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Load ICLR and ACL results
iclr_results = pd.read_csv('https://docs.google.com/spreadsheets/d/' +
                           '1HkQ-nbE7A44Xfu34JqkvoOEUHWXkCuESqsS4HulBUM8' +
                           '/export?gid=1528686706&format=csv')

acl_results = pd.read_csv('https://docs.google.com/spreadsheets/d/' +
                          '1HkQ-nbE7A44Xfu34JqkvoOEUHWXkCuESqsS4HulBUM8' +
                          '/export?gid=1633737867&format=csv')

# Concatenate dataframes and keep necessary columns
results = pd.concat([iclr_results, acl_results], axis=0)
results = results[["Paper link", "Category", "Dataset", "Direct perf.", "CoT perf.", "Delta"]]

# Load categories and map datasets to categories
# categories = pd.read_csv("https://docs.google.com/spreadsheets/d/1_ifN9FhoAdzp6-EtUKtSugtGshZWEBRaNvGvjPFknpk/export?gid=387769381&format=csv")
categories = pd.read_csv("https://docs.google.com/spreadsheets/d/1HkQ-nbE7A44Xfu34JqkvoOEUHWXkCuESqsS4HulBUM8/export?gid=387769381&format=csv")
dataset_to_category = categories.set_index('Dataset')['Category'].to_dict()
results['Category'] = results['Dataset'].map(dataset_to_category).str.split('(').str[0].str.strip()

# Filter out NaN rows
results = results.dropna()
results = results[['Paper link', 'Category', 'Delta', 'CoT perf.', 'Direct perf.']]

# Average Delta over (paper link, category) pairs and sort by Delta
# these are the "blue dots"
individual_results = results.groupby(["Paper link", "Category"]).mean().reset_index()


REPORT_METRICS_FOR_INDIVIDAL_EXPS = False

if REPORT_METRICS_FOR_INDIVIDAL_EXPS:
    print("REPORTING ON ALL EXPERIMENTS")
    # get the top 3 categories on average and print their name plus values in order
    top_categories = results.groupby('Category')['Delta'].mean().sort_values(ascending=False).head(3)
    print("Top 3 categories")
    print(top_categories)

    # for the "math, physics, algos" print out the mean CoT perf. and Direct perf. values
    math_physics_algos = results[results['Category'].isin(top_categories.index)]
    print("Cot Perf vs Direct perf for math, physics, algos")
    print(math_physics_algos[['CoT perf.', 'Direct perf.']].mean())

    # Average the CoT. perf for every category not in the top 3 all together and print it out. Also do this for Direct Perf.
    other_categories = results[~results['Category'].isin(top_categories.index)]
    print("Cot Perf vs Direct perf for other categories")
    print(other_categories[['CoT perf.', 'Direct perf.']].mean())
else:
    print("REPORTING ON AGGREGATED METRICS")
    # get the top 3 categories on average and print their name plus values in order
    top_categories = individual_results.groupby('Category')['Delta'].mean().sort_values(ascending=False).head(3)
    print("Top 3 categories")
    print(top_categories)

    # for the "math, physics, algos" print out the mean CoT perf. and Direct perf. values
    math_physics_algos = individual_results[individual_results['Category'].isin(top_categories.index)]
    print("Cot Perf vs Direct perf for math, physics, algos")
    print(math_physics_algos[['CoT perf.', 'Direct perf.']].mean())

    # Average the CoT. perf for every category not in the top 3 all together and print it out. Also do this for Direct Perf.
    other_categories = individual_results[~individual_results['Category'].isin(top_categories.index)]
    print("Cot Perf vs Direct perf for other categories")
    print(other_categories[['CoT perf.', 'Direct perf.']].mean())

# Plot settings
orig_size = (32,24)
new_size = (42, 30)
fig, ax = plt.subplots(figsize=new_size)
xlim = (-100, 100)
textpos = -49

# Prepare data for plotting
df = individual_results[['Category', 'Delta']].dropna()
all_df = results[['Category', 'Delta']].dropna()

# Exclude 'various' category
df = df[df['Category'] != 'various']
all_df = all_df[all_df['Category'] != 'various']

# Order categories by the median of Delta
#category_means = df.groupby('Category')['Delta'].mean().sort_values(ascending=True).index
category_means = df.groupby('Category')['Delta'].median().sort_values(ascending=True).index

df['Category'] = pd.Categorical(df['Category'], categories=category_means, ordered=True)
all_df['Category'] = pd.Categorical(all_df['Category'], categories=category_means, ordered=True)

small_dot_color = 'dimgrey'#'gray'
small_dot_alpha = .6 #.5
violin_saturation = .4#.5
big_dot_size = 17 #16

# Plot violin and strip plot
np.random.seed(123)
sns.violinplot(x='Delta', y='Category', data=all_df, color='lightgray', bw=.9, saturation=violin_saturation, inner='stick', split=True, linewidth=0, ax=ax)
np.random.seed(123)
g1 = sns.stripplot(x='Delta', y='Category', data=all_df, jitter=0.3, color=small_dot_color, size=8, dodge=True, alpha=small_dot_alpha, ax=ax, label="Individual experiments")  # Muted background dots
np.random.seed(123)
g2 = sns.stripplot(x='Delta', y='Category', data=df, jitter=0.2, color="#1f77b4", size=big_dot_size, dodge=True, ax=ax, edgecolor="white", linewidth=0.7, label="Aggregated paper results")  # Highlighted aggregated dots

# Add vertical lines for baseline (0) and significance threshold (5)
ax.axvline(0, color='black', linestyle='-', linewidth=4)
# get median of all deltas in all_df
median = df['Delta'].mean()
ax.axvline(median, color='red', linestyle='--', linewidth=8, label=f'Mean CoT Improvement ({median:.2f}%)')

# Customize plot appearance
xlabel_fontsize = 40#40

ax.set(xlim=xlim)
ax.set_title("CoT Performance Improvement Across Tasks Aggregated by Paper and Category", fontsize=50, pad=20)#, font='Arial')
ax.set_xlabel('Delta (%) – Improvement from Chain-of-Thought', fontsize=xlabel_fontsize, font='Arial')

# Rotate x-tick labels
for tick in ax.get_xticklabels():
    tick.set_fontsize(35) #30

# Remove y-axis labels but keep custom category labels
ax.set(yticks=[], ylabel="")
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add custom category labels to the left of the plot
for i, category in enumerate(df['Category'].cat.categories):
    ax.text(textpos, i, category, ha='right', va='center', fontsize=40, color="black", fontweight='bold', font='Arial')

# Add legend with unique labels
legend_fontsize = 40#30
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Ensure each label is unique
ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=legend_fontsize, frameon=False)

# Save plot in high quality for NeurIPS submission
plt.tight_layout()
dpi = 600#600
plt.savefig('final_plot_neurips_ready.png', dpi=dpi, bbox_inches='tight')


######################################################
# See what top outliers are:

results2 = pd.concat([iclr_results, acl_results], axis=0)
results2 = results2[["Paper link", "Category", "Dataset", "Direct perf.", "CoT perf.", "Delta"]]

# Load categories and map datasets to categories
categories = pd.read_csv("https://docs.google.com/spreadsheets/d/1HkQ-nbE7A44Xfu34JqkvoOEUHWXkCuESqsS4HulBUM8/export?gid=387769381&format=csv")
dataset_to_category = categories.set_index('Dataset')['Category'].to_dict()
results2['Category'] = results2['Dataset'].map(dataset_to_category).str.split('(').str[0].str.strip()

# Filter out NaN rows
results2 = results2.dropna()
avg_delta_df = results2.groupby(['Category', 'Paper link']).agg({
    'Delta': 'mean',               # Calculate the mean of 'Delta'
    'Dataset': lambda x: list(x.unique())  # Collect unique 'Datasets'
}).reset_index()

# Display the resulting dataframe
#print(avg_delta_df)

df_sorted = avg_delta_df.sort_values(by='Delta', ascending=False)

# non-math for now
df_filtered = df_sorted[~df_sorted['Category'].isin(['math', 'logical reasoning', 'symbolic & algorithmic'])]

print(tabulate(df_sorted.head(20)))
print("TOP NON-MATH, SYMBOLIC or LOGICAL:")
print(tabulate(df_filtered.head(40)))
#plt.show()
