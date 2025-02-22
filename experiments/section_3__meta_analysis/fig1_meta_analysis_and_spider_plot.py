import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from math import pi
from scipy.stats import ttest_rel
import warnings

warnings.filterwarnings('ignore')

# Define your mapping
mapping = {
    'MuSR Team Allocations': 'math',
    'MuSiQue': 'multi-hop QA',
    'BigGen Bench': 'other',
    'CommonsenseQA': 'commonsense',
    'AGIEval LSAT RC': 'entailment',
    'AGIEval LSAT LR': 'entailment',
    'PiQA': 'commonsense',
    'Arc Easy': 'entailment',
    'GPQA': 'commonsense',
    'SiQA': 'commonsense',
    'Arc Challenge': 'entailment',
    'AGIEval LSAT AR': 'entailment',
    'Winogrande': 'commonsense',
    'MMLU': 'other',
    'Folio': 'logical reasoning',
    'StrategyQA': 'commonsense',
    'MuSR Object Placements': 'spatial reasoning',
    'MuSR Murder Mysteries': 'logical reasoning',
    'ContextHub Abductive L2': 'logical reasoning',
    'ContextHub Deductive L2': 'logical reasoning',
    'MMLU Pro': 'math',
    'ContextHub Deductive L1': 'logical reasoning',
    'BigBench-Hard': 'other',
    'ContextHub Abductive L1': 'logical reasoning',
    'GSM8k-Hard': 'math',
    'MATH': 'math',
    'GSM8k': 'math',
}

# Define models and datasets
models = [
    ('meta-llama__Llama-2-7b-chat-hf', 'L2 7b', 'Meta-Llama 2 7b'),
    ('mistralai__Mistral-7B-Instruct-v0.3', 'M 7b', 'Mistral 7b'),
    ('meta-llama__Meta-Llama-3.1-8B-Instruct', 'L3.1 8b', 'Meta-Llama 3.1 8b'),
    ('meta-llama__Meta-Llama-3.1-70B-Instruct', 'L 3.1 70b', 'Meta-Llama 3.1 70b'),
    ('google__gemma-2-9b-it', 'G2 9b', 'Gemma 2 9b'),
    ('microsoft__Phi-3-small-8k-instruct', 'P3-small', 'Phi-3 Small 8k'),
    ('Qwen__Qwen2-7B-Instruct', 'Q 7b', 'Qwen 2 7b'),
    ('Qwen__Qwen2-72B-Instruct', 'Q 72b', 'Qwen 2 72b'),
    ('gpt-4o-mini-2024-07-18', 'GPT-4o mini', 'GPT-4o Mini'),
    ('gpt-4o-2024-08-06', 'GPT4o', 'Gpt-4o'),
    ('claude-3-haiku-20240307', 'Claude-3 Haiku', 'Claude-3 Haiku'),
    ('claude-3-5-sonnet-20240620', 'Claude-3.5 Sonnet', 'Claude-3.5 Sonnet'),
    ('google__gemini-1.5-flash-001', 'Gemini 1.5 Flash', 'Gemini 1.5 Flash'),
    ('google__gemini-1.5-pro-001', 'Gemini 1.5 Pro', 'Gemini 1.5 Pro'),
]

datasets = [
    ('csqa', 'CommonsenseQA'),
    ('stratqa', 'StrategyQA'),
    ('siqa', 'SiQA'),
    ('piqa', 'PiQA'),
    ('winogrande', 'Winogrande'),
    ('arc_easy', 'Arc Easy'),
    ('arc_challenge', 'Arc Challenge'),
    ('agieval_lsat_lr', 'AGIEval LSAT LR'),
    ('agieval_lsat_ar', 'AGIEval LSAT AR'),
    ('agieval_lsat_rc', 'AGIEval LSAT RC'),
    ('contexthub_deductive_level1', 'ContextHub Deductive L1'),
    ('contexthub_deductive_level2', 'ContextHub Deductive L2'),
    ('contexthub_abductive_level1', 'ContextHub Abductive L1'),
    ('contexthub_abductive_level2', 'ContextHub Abductive L2'),
    ('mm_musr', 'MuSR Murder Mysteries'),
    ('ta_musr', 'MuSR Team Allocations'),
    ('op_musr', 'MuSR Object Placements'),
    ('mmlu', 'MMLU'),
    ('mmlu_pro', 'MMLU Pro'),
    ('gpqa', 'GPQA'),
    ('math', 'MATH'),
    ('gsm8k', 'GSM8k'),
    ('biggen_bench', 'BigGen Bench'),
    ('gsm8k_hard', 'GSM8k-Hard'),
    ('musique_all', 'MuSiQue'),
    ('folio', 'Folio'),
    ('bbh', 'BigBench-Hard')
]

# Define groups
groups = {
    'Commonsense': ['CommonsenseQA', 'StrategyQA', 'SiQA', 'PiQA', 'Winogrande'],
    'Knowledge': ['Arc Easy', 'Arc Challenge', 'MMLU', 'MMLU Pro'],
    'Mathematical': ['GSM8k', 'GSM8k-Hard', 'MATH', 'GPQA'],
    'Symbolic': ['BigBench-Hard', 'Folio', 'ContextHub Deductive L1', 'ContextHub Deductive L2', 'ContextHub Abductive L1', 'ContextHub Abductive L2'],
    'Soft Reasoning': ['BigGen Bench', 'MuSR Object Placements', 'MuSR Murder Mysteries', 'MuSR Team Allocations', 'MuSiQue', 'AGIEval LSAT LR', 'AGIEval LSAT RC', 'AGIEval LSAT AR']
}

# Map each dataset to its group
dataset_to_group = {}
for group_name, dataset_list in groups.items():
    for dataset_name in dataset_list:
        dataset_to_group[dataset_name] = group_name

# Functions from the first script (including necessary modifications)

def violin_plot(df, vertical=True, set_y_label=False, swarm_or_strip='swarm', xlim=(-60, 60), textpos=-49, ax=None):
    """Plot the violin and scatter plot."""
    df = df[['Category', 'Delta']]
    df = df.dropna()  # Drop rows with missing values
    df = df[df['Category'] != 'various']  # Exclude 'various' category
    category_means = df.groupby('Category')['Delta'].mean()
    # Sort category means by my list of categories
    sorted_list_of_categories = [
        'text classification',
        'meta-linguistic',
        'commonsense reasoning',
        'encyclopedic knowledge',
        'multi-hop QA',
        'generation',
        'entailment',
        'context-aware QA',
        'other',
        'spatial & temporal reasoning',
        'logical reasoning',
        'math',
        'symbolic & algorithmic'
    ]
    # Get the categories actually present in the data, in the desired order
    categories_in_data = df['Category'].unique()
    categories_ordered = [cat for cat in sorted_list_of_categories if cat in categories_in_data]

    # Now set the category order
    df['Category'] = pd.Categorical(df['Category'], categories=categories_ordered, ordered=True)

    # df['Category'] = pd.Categorical(df['Category'], categories=category_means, ordered=True)
    palette = sns.color_palette("flare", as_cmap=True)(np.linspace(0.3, 1, len(category_means)))
    palette = [list(i) for i in palette]

    if vertical:
        g = sns.swarmplot(x='Category', y='Delta', data=df, palette=palette, size=4, ax=ax)
        sns.violinplot(x='Category', y='Delta', data=df, color='lightblue', bw_method=.9, saturation=.5, inner='stick', split=True, linewidth=0, ax=ax)
    else:
        g = sns.swarmplot(x='Delta', y='Category', data=df, palette=palette, size=4, ax=ax)
        sns.violinplot(x='Delta', y='Category', data=df, color='lightblue', bw_method=.9, saturation=.5, inner='stick', split=True, linewidth=0, ax=ax)
        ax.set(xlim=xlim)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(df['Delta'].mean(), color='red', linestyle='--', linewidth=2, label='Mean Improvement ({:.1f})'.format(df['Delta'].mean()), zorder=3)

    ax.set_title("Meta-analysis of CoT improvements", fontsize=16)
    ax.set(yticks=[], ylabel="")
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, category in enumerate(df['Category'].cat.categories):
        x = textpos
        ax.text(x, i, category, ha='right', va='center', fontsize=12, color=palette[i], fontweight='bold', font='Arial')

    if vertical:
        ax.set_xlabel('Category', fontsize=15)
        ax.set_ylabel('Delta', fontsize=15)
        ax.set_xticklabels(rotation=-40, ha='left', fontsize=15)
    else:
        ax.set_xlabel('Improvement of CoT over direct answering', fontsize=15)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(15)

def load_accuracy_data(file_name='accuracy_data_zs.json'):
    with open(file_name, 'r') as f:
        accuracy_data = json.load(f)

    # Convert accuracy data into a long-form DataFrame
    data_list = []
    for ds in datasets:
        dataset = ds[0]
        datasetname = ds[1]
        if datasetname in accuracy_data:
            for model, label, model_name in models:
                if label in accuracy_data[datasetname]:
                    entry = accuracy_data[datasetname][label]
                    is_sig = False
                    if entry['has_zs_direct'] and entry['has_zs_cot']:
                        sig_results = entry['sig_test_results']
                        is_sig = sig_results['p_value'] <= 0.05 / len(models)
                        cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']
                        data_list.append({
                            'Dataset': datasetname,
                            'Category': mapping.get(datasetname, 'other'),
                            'Model': model_name,
                            'Accuracy': entry['zero_shot_direct'],
                            'Type': 'Zero Shot Direct',
                            'Delta': cot_delta * 100,
                            'Significant': is_sig
                        })
    df_accuracy = pd.DataFrame(data_list)
    return df_accuracy, accuracy_data

def compute_group_accuracies(df):
    """
    Compute average accuracies per group and type across datasets and models.
    Returns a DataFrame with columns ['Group', 'Type', 'Accuracy']
    """
    group_accuracy = df.groupby(['Group', 'Type'])['Accuracy'].mean().reset_index()
    return group_accuracy

def compute_group_significance(df):
    """
    Compute significance of the difference between Zero-shot Direct and Zero-shot CoT accuracies per group.
    Returns a dictionary mapping group names to significance (True/False) and p-values.
    """
    group_significance = {}
    alpha = 0.05  # Significance level

    group_names = df['Group'].unique()
    for group in group_names:
        df_group = df[df['Group'] == group]
        # Pivot the DataFrame to have 'Accuracy' per 'Type'
        df_pivot = df_group.pivot_table(values='Accuracy', index=['Dataset', 'Model'], columns='Type').dropna()

        if df_pivot.empty:
            continue

        # Perform paired t-test
        t_stat, p_value = ttest_rel(df_pivot['Zero-shot CoT'], df_pivot['Zero-shot direct answer'])

        significant = p_value <= alpha

        group_significance[group] = {'significant': significant, 'p_value': p_value}

    return group_significance

def create_accuracy_dataframe(accuracy_data, models, datasets, dataset_to_group):
    """
    Create a DataFrame from the accuracy data.
    """
    data_list = []

    for dataset_identifier, dataset_name in datasets:
        if dataset_name in accuracy_data:
            group_name = dataset_to_group.get(dataset_name, 'Other')
            for model_identifier, label, model_name in models:
                if label in accuracy_data[dataset_name]:
                    entry = accuracy_data[dataset_name][label]
                    if entry.get('has_zs_direct') and entry.get('has_zs_cot'):
                        sig_results = entry['sig_test_results']
                        is_sig = sig_results['significant']

                        actual_cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']

                        data_list.append({
                            'Dataset': dataset_name,
                            'Group': group_name,
                            'Model': model_name,
                            'Accuracy': entry['zero_shot_direct'],
                            'Type': 'Zero-shot direct answer',
                            'CoT_Delta': actual_cot_delta,
                            'Significant': is_sig
                        })
                        data_list.append({
                            'Dataset': dataset_name,
                            'Group': group_name,
                            'Model': model_name,
                            'Accuracy': entry['zero_shot_cot'],
                            'Type': 'Zero-shot CoT',
                            'CoT_Delta': actual_cot_delta,
                            'Significant': is_sig
                        })
                    else:
                        print(f"Skipping model '{model_name}' for dataset '{dataset_name}' because it has incomplete data")
    df = pd.DataFrame(data_list)
    return df

def plot_radar_chart_custom_labels(group_accuracy, group_significance,
                                   output_file='neuralips_zs_radar_custom_labels.png', ax=None):
    """
    Plot radar chart of average accuracies per group and annotate significance.
    Custom label positions and rotations.
    """
    # Prepare data for radar plot
    categories = ['Commonsense', 'Soft Reasoning', 'Mathematical', 'Symbolic', 'Knowledge']
    N = len(categories)

    # Compute angles for radar plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig = None
    if ax is None:
        # Initialize radar plot with adjusted figure size
        fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))

    # Values for Zero-shot Direct Answer
    values_direct = group_accuracy[group_accuracy['Type'] == 'Zero-shot direct answer'].set_index('Group').loc[categories]['Accuracy'].tolist()
    values_direct += values_direct[:1]

    # Values for Zero-shot CoT
    values_cot = group_accuracy[group_accuracy['Type'] == 'Zero-shot CoT'].set_index('Group').loc[categories]['Accuracy'].tolist()
    values_cot += values_cot[:1]

    # Significance per group
    significance = [group_significance.get(cat, {}).get('significant', False) for cat in categories]
    significance += significance[:1]

    # Plot Direct Answer results
    ax.plot(angles, values_direct, linewidth=2, linestyle='solid', label='Zero-shot direct answer', color='blue')
    ax.fill(angles, values_direct, alpha=0.1, color='blue')

    # Plot CoT results
    ax.plot(angles, values_cot, linewidth=2, linestyle='solid', label='Zero-shot CoT', color='orange')
    ax.fill(angles, values_cot, alpha=0.1, color='orange')

    # Annotate significance
    # for i, sig in enumerate(significance[:-1]):  # Exclude the last point which is the same as the first
        # if sig:
        #     angle = angles[i]
        #     value = max(values_direct[i], values_cot[i])
        #     ax.text(angle, value + 0.05, '*', horizontalalignment='center', size=20, color='red', weight='bold')

    # Adjust Y labels
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", size=12)
    ax.set_ylim(0, 1)

    # Add title and legend to ax
    ax.set_title('Our experiments on CoT improvements', size=15, y=1.10)
    ax.legend(loc='upper right', bbox_to_anchor=(0.3, 0.03), fontsize=10)

    # Set the xticks and xticklabels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # Adjust label positions and orientations
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle)
        if angle_deg >= 90 and angle_deg <= 270:
            label.set_horizontalalignment('right')
            label.set_rotation(angle_deg + 180)
        else:
            label.set_horizontalalignment('left')
            label.set_rotation(angle_deg)
        label.set_rotation_mode('anchor')

    # Increase the padding between the labels and the plot
    ax.tick_params(axis='x', which='major', pad=0)

    if fig is not None:
        # Save and show the plot
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.show()

# Main script
if __name__ == '__main__':
    # Load data from literature
    iclr_results = pd.read_csv('https://docs.google.com/spreadsheets/d/' +
                    '1HkQ-nbE7A44Xfu34JqkvoOEUHWXkCuESqsS4HulBUM8' +
                    '/export?gid=1528686706&format=csv',
                    )
    acl_results = pd.read_csv('https://docs.google.com/spreadsheets/d/' +
                        '1HkQ-nbE7A44Xfu34JqkvoOEUHWXkCuESqsS4HulBUM8' +
                        '/export?gid=1633737867&format=csv',
                       )
    df1 = pd.concat([iclr_results, acl_results], ignore_index=True, sort=False)

    # Read in categories from other sheet:
    categories_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1HkQ-nbE7A44Xfu34JqkvoOEUHWXkCuESqsS4HulBUM8/export?gid=387769381&format=csv")
    duplicates = categories_df.groupby('Dataset')['Category'].nunique().reset_index()
    duplicates_with_multiple_categories = duplicates[duplicates['Category'] > 1]
    if len(duplicates_with_multiple_categories) != 0:
        raise ValueError("More than one category for a given dataset!")

    dataset_to_category = categories_df.set_index('Dataset')['Category'].to_dict()
    df1['Category'] = df1['Dataset'].map(dataset_to_category).str.split('(').str[0].str.strip()

    # Load accuracy data
    df2, accuracy_data = load_accuracy_data()
    df_accuracy = create_accuracy_dataframe(accuracy_data, models, datasets, dataset_to_group)

    # Compute group accuracies and significances
    group_accuracy = compute_group_accuracies(df_accuracy)
    group_significance = compute_group_significance(df_accuracy)

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')

    # Plot violin plot on ax1
    violin_plot(df1, vertical=False, ax=ax1)

    # Plot radar chart on ax2
    plot_radar_chart_custom_labels(group_accuracy, group_significance, ax=ax2)



    # Add super title
    # fig.suptitle('Average Improvement from CoT by Task Category', fontsize=30)

    # Adjust layout
    plt.tight_layout()

    plt.show()
