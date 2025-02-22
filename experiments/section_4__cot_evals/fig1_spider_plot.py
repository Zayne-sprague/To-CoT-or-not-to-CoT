"""
Downloads all the models evals and then plots a spider plot.
"""
import os
import json
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
import warnings

warnings.filterwarnings('ignore')

# Set your HuggingFace API token (make sure to set the environment variable 'HUGGINGFACE_TOKEN')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

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
    'Soft': ['BigGen Bench', 'MuSR Object Placements', 'MuSR Murder Mysteries', 'MuSR Team Allocations', 'MuSiQue', 'AGIEval LSAT LR',  'AGIEval LSAT RC', 'AGIEval LSAT AR']
}

# Map each dataset to its group
dataset_to_group = {}
for group_name, dataset_list in groups.items():
    for dataset_name in dataset_list:
        dataset_to_group[dataset_name] = group_name

def perform_significance_test(zs_sample, cot_sample, alpha=0.05):
    """
    Perform McNemar's test for paired nominal data.
    Returns (significant: bool, p_value: float)
    """
    from collections import Counter
    n00 = n11 = n01 = n10 = 0
    for zs, cot in zip(zs_sample, cot_sample):
        if zs == cot == 0:
            n00 +=1
        elif zs == cot ==1:
            n11 +=1
        elif zs ==1 and cot ==0:
            n10 +=1
        elif zs ==0 and cot ==1:
            n01 +=1
    # Create contingency table
    table = [[n00, n01],
             [n10, n11]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue <= alpha, result.pvalue

def load_model_dataset_data(model_identifier, dataset_identifier):
    """
    Load the dataset for the given model and dataset identifiers.
    """
    try:
        dataset_name = f"TAUR-Lab/Taur_CoT_Analysis_Project___{model_identifier.replace('/', '__')}"
        data = load_dataset(dataset_name, dataset_identifier.replace("/", "__"), use_auth_token=HUGGINGFACE_TOKEN)
        data = data["latest"]
        return data
    except Exception as e:
        print(f"Failed to load data for {dataset_identifier} in {model_identifier}: {e}")
        return None

def process_dataset_data(data, dataset_identifier):
    """
    Process the dataset, applying any necessary filtering.
    Returns the processed data.
    """
    if dataset_identifier == 'biggen_bench':
        # Apply filtering as per the original code
        to_filter = [
            ['theory_of_mind', 'faux_pas_explanation', 'maybe'],
            ['instruction_following', 'faithful_explanation', 'maybe'],
            ['reasoning', 'competition_mwp', 'no'],
            ['reasoning', 'hypothesis_proposal', 'no'],
            ['reasoning', 'high_school_mwp', 'no'],
            ['reasoning', 'first_order_logic', 'no'],
            ['reasoning', 'table_reason', 'no'],
            ['grounding', 'false_context', 'maybe'],
            ['grounding', 'role_playing', 'maybe'],
            ['refinement', 'rationale_revision', 'maybe'],
            ['refinement', 'llm_judge_absolute', 'maybe'],
            ['refinement', 'revision_with_tools', 'no'],
            ['refinement', 'llm_judge_relative', 'maybe']
        ]
        look_up_table = {
            x[0] + "_" + x[1]: x[2] for x in to_filter
        }

        filtering_data = [(x, json.loads(x['additional_information'])) for x in data]
        filtering_data = [x for x in filtering_data if x[1]['capability'] not in ['multilingual', 'tool_usage']]
        data = [x[0] for x in filtering_data if look_up_table.get(x[1]['capability'] + "_" + x[1]['task'], 'yes') not in ['no', 'maybe']]

    return data

def compute_accuracies_and_significance(data):
    """
    Compute zero-shot direct and zero-shot CoT accuracies,
    check if data is available, and perform significance testing.
    Returns a dictionary with accuracies, flags, and significance results.
    """
    total = len(data)
    if total == 0:
        return None

    zs_sample = [1 if x['zero_shot_direct_is_correct'] else 0 for x in data]
    cot_sample = [1 if x['zero_shot_cot_is_correct'] else 0 for x in data]

    zs_direct_correct = sum(zs_sample)
    zs_cot_correct = sum(cot_sample)

    custom_default_message = "[invalid] - configuration was not run"

    has_zs_direct = any([x['zero_shot_direct_model_response'] != custom_default_message for x in data])
    has_zs_cot = any([x['zero_shot_cot_model_response'] != custom_default_message for x in data])

    # Perform significance test
    significant = False
    p_value = None
    if has_zs_direct and has_zs_cot:
        significant, p_value = perform_significance_test(zs_sample, cot_sample)

    return {
        'zero_shot_direct': zs_direct_correct / total,
        'zero_shot_cot': zs_cot_correct / total,
        'has_zs_direct': has_zs_direct,
        'has_zs_cot': has_zs_cot,
        'sig_test_results': {
            'significant': bool(significant),
            'p_value': float(p_value)
        }
    }

def collect_accuracy_data(models, datasets, force_reload=False, fetch_if_empty=True, file_name='accuracy_data_zs.json'):
    """
    Collect accuracy data for the given models and datasets.
    Returns a dictionary of accuracy data.
    """
    accuracy_data = {}

    if os.path.exists(file_name) and not force_reload:
        with open(file_name, 'r') as f:
            accuracy_data = json.load(f)
    else:
        # Initialize empty accuracy_data
        accuracy_data = {}

    for model_identifier, label, model_name in models:
        for dataset_identifier, dataset_name in datasets:
            # Check if data is already present
            if (accuracy_data.get(dataset_name, {}).get(label) is not None and
                accuracy_data[dataset_name][label].get('has_zs_direct') and
                accuracy_data[dataset_name][label].get('has_zs_cot')):
                continue  # Skip if data is already present

            print(f"Processing data for model '{model_name}' on dataset '{dataset_name}'")
            data = load_model_dataset_data(model_identifier, dataset_identifier)
            if data is None:
                continue  # Skip if data could not be loaded

            data = process_dataset_data(data, dataset_identifier)

            result = compute_accuracies_and_significance(data)
            if result is None:
                continue  # Skip if data could not be processed

            if dataset_name not in accuracy_data:
                accuracy_data[dataset_name] = {}
            accuracy_data[dataset_name][label] = result

    # Save the accuracy data to file
    with open(file_name, 'w') as f:
        json.dump(accuracy_data, f)

    return accuracy_data

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

    groups = df['Group'].unique()
    for group in groups:
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


def plot_radar_chart_custom_labels(group_accuracy, group_significance,
                                   output_file='neuralips_zs_radar_custom_labels.png', ax=None):
    """
    Plot radar chart of average accuracies per group and annotate significance.
    Custom label positions and rotations.
    """
    # Prepare data for radar plot
    categories = ['Soft', 'Knowledge', 'Mathematical', 'Symbolic', 'Commonsense']
    N = len(categories)

    # Compute angles for radar plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig = None
    if ax is None:
        # Initialize radar plot with adjusted figure size
        fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))
    # else:
    #     set the ax subplot kw polar to true
        # ax.set_theta_offset(pi / 2)

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
    #     if sig:
    #         angle = angles[i]
    #         value = max(values_direct[i], values_cot[i])
    #         ax.text(angle, value + 0.05, '*', horizontalalignment='center', size=20, color='red', weight='bold')


    # Adjust Y labels
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", size=22)
    ax.set_ylim(0, 1)

    # Add title and legend to ax
    ax.set_title('CoT vs. Direct Answer Prompts on Reasoning Categories', size=32, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(0.3, 0.03), fontsize=22)

    # Set the xticks and xticklabels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=28)

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
    ax.tick_params(axis='x', which='major', pad=20)

    if fig is not None:
        # Save and show the plot
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.show()


def main(ax=None):
    # Collect accuracy data
    accuracy_data = collect_accuracy_data(models, datasets, force_reload=False, fetch_if_empty=True,
                                          file_name='accuracy_data_zs.json')

    # Create DataFrame from accuracy data
    df = create_accuracy_dataframe(accuracy_data, models, datasets, dataset_to_group)

    # Compute group accuracies
    group_accuracy = compute_group_accuracies(df)

    # Compute group significance
    group_significance = compute_group_significance(df)

    # Plot radar chart with custom labels
    plot_radar_chart_custom_labels(group_accuracy, group_significance,
                                   output_file='bump__neuralips_zs_radar_custom_labels.png', ax=ax)


if __name__ == "__main__":
    main()

