import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import math
import matplotlib.gridspec as gridspec

models = [
    ('meta-llama__Llama-2-7b-chat-hf', 'L2 7b', 'Meta-Llama 2 7b'),
    ('mistralai__Mistral-7B-Instruct-v0.3', 'M 7b', 'Mistral 7b'),
    ('meta-llama__Meta-Llama-3.1-8B-Instruct', 'L3.1 8b', 'Meta-Llama 3.1 8b'),
    ('meta-llama__Meta-Llama-3.1-70B-Instruct', 'L 3.1 70b', 'Meta-Llama 3.1 70b'),
    ('google__gemma-2-9b-it', 'G2 9b', 'Gemma 2 9b'),
    ('microsoft__Phi-3-small-8k-instruct', 'P3-small', 'Phi-3 Small 8k'),
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
    ('agieval_lsat_lr', 'AGIEval LSAT'),
    ('agieval_lsat_ar', 'AGIEval LSAT'),
    ('agieval_lsat_rc', 'AGIEval LSAT'),
    ('contexthub_deductive_level1', 'ContextHub'),
    ('contexthub_deductive_level2', 'ContextHub'),
    ('contexthub_abductive_level1', 'ContextHub'),
    ('contexthub_abductive_level2', 'ContextHub'),
    ('mm_musr', 'MuSR'),
    ('ta_musr', 'MuSR'),
    ('op_musr', 'MuSR'),
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



def sigtest(sample1, sample2, alpha, test_type):
    # Placeholder implementation using a simple t-test
    from scipy.stats import ttest_ind
    stat, p_value = ttest_ind(sample1, sample2)
    significant = p_value <= alpha
    return significant, p_value

# Mapping datasets to categories
mapping = {
    'MuSiQue': 'multi-hop QA',
    'BigGen Bench': 'other',
    'CommonsenseQA': 'commonsense',
    'PiQA': 'commonsense',
    'Arc Easy': 'entailment',
    'GPQA': 'commonsense',
    'SiQA': 'commonsense',
    'Arc Challenge': 'entailment',
    'AGIEval LSAT': 'entailment',
    'Winogrande': 'commonsense',
    'MMLU': 'other',
    'Folio': 'logical reasoning',
    'StrategyQA': 'commonsense',
    'MuSR': 'spatial reasoning',
    'ContextHub': 'logical reasoning',
    'MMLU Pro': 'math',
    'BigBench-Hard': 'other',
    'GSM8k-Hard': 'math',
    'MATH': 'math',
    'GSM8k': 'math',
}

# Loading and processing accuracy data
def load_accuracy_data(file_name='aggregated_accuracy_data_zs.json', force_reload: bool = True):

    if force_reload:
        from datasets import load_dataset
        from key_handler import KeyHandler
        from tqdm import tqdm
        import time

        all_data = {}
        # all_data = json.load(open('tmp.json', 'r'))

        for (model, label, model_name) in tqdm(models, desc="Loading models", total=len(models)):

            for (dataset, datasetname) in datasets:
                for i in range(10):
                    try:
                        dataset_loaded = load_dataset(
                            f"TAUR-Lab/Taur_CoT_Analysis_Project___" + model.replace('/', '__'),
                            dataset.replace("/", "__"),
                            token=KeyHandler.hf_key
                        )

                        data = dataset_loaded["latest"]
                        break
                    except Exception:
                        print(f"Failed to load data for {dataset} in {model}")
                        time.sleep(3)
                        continue

                if dataset == 'biggen_bench':
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
                    filtering_data = [x for x in filtering_data if
                                      x[1]['capability'] not in ['multilingual', 'tool_usage']]
                    data = [x[0] for x in filtering_data if
                            look_up_table.get(x[1]['capability'] + "_" + x[1]['task'], 'yes') not in ['no', 'maybe']]

                total = len(data)

                zs_sample = [1 if x['zero_shot_direct_is_correct'] else 0 for x in data]
                cot_sample = [1 if x['zero_shot_cot_is_correct'] else 0 for x in data]

                if datasetname not in all_data:
                    all_data[datasetname] = {}

                if label not in all_data[datasetname]:
                    all_data[datasetname][label] = {
                        'zero_shot_direct': zs_sample,
                        'zero_shot_cot': cot_sample,
                        'zero_shot_responses': [x['zero_shot_direct_model_response'] for x in data],
                        'zero_shot_cot_responses': [x['zero_shot_cot_model_response'] for x in data]
                    }
                else:
                    all_data[datasetname][label]['zero_shot_direct'].extend(zs_sample)
                    all_data[datasetname][label]['zero_shot_cot'].extend(cot_sample)
                    all_data[datasetname][label]['zero_shot_responses'].extend([x['zero_shot_direct_model_response'] for x in data])
                    all_data[datasetname][label]['zero_shot_cot_responses'].extend([x['zero_shot_cot_model_response'] for x in data])

        # Save all_data to tmp.json
        with open('tmp.json', 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False)

        # Process all_data to create accuracy_data
        accuracy_data = {}
        for datasetname in all_data:
            for label in all_data[datasetname]:
                data = all_data[datasetname][label]
                total = len(data['zero_shot_direct'])
                zs_sample = data['zero_shot_direct']
                cot_sample = data['zero_shot_cot']

                significant, p_value = sigtest(zs_sample, cot_sample, 0.05, "bernoulli")

                zs_direct_correct = sum(zs_sample)
                zs_cot_correct = sum(cot_sample)

                custom_default_message = "[invalid] - configuration was not run"

                has_zs_direct = any([response != custom_default_message for response in data['zero_shot_responses']])
                has_zs_cot = any([response != custom_default_message for response in data['zero_shot_cot_responses']])

                if datasetname not in accuracy_data:
                    accuracy_data[datasetname] = {}

                accuracy_data[datasetname][label] = {
                    'zero_shot_direct': zs_direct_correct / total,
                    'zero_shot_cot': zs_cot_correct / total,
                    'has_zs_direct': has_zs_direct,
                    'has_zs_cot': has_zs_cot,
                    'sig_test_results': {
                        'significant': bool(significant),
                        'p_value': float(p_value)
                    }
                }

        # Save accuracy_data to the specified file_name
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(accuracy_data, f, ensure_ascii=False)

    # Load accuracy_data from file_name
    with open(file_name, 'r', encoding='utf-8') as f:
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
                            'Category': mapping.get(datasetname, 'Other'),
                            'Model': model_name,
                            'Accuracy': entry['zero_shot_direct'],
                            'Type': 'Zero Shot Direct',
                            'Delta': cot_delta * 100,
                            'Significant': is_sig
                        })
    df_accuracy = pd.DataFrame(data_list)
    return df_accuracy, accuracy_data

def get_avg_acc_deltas(accuracy_data):
    # Calculate average CoT delta for each dataset
    avg_cot_delta = {}
    all_deltas = []
    for ds in datasets:
        dataset = ds[0]
        datasetname = ds[1]
        if datasetname in accuracy_data:
            total_delta = 0
            totals = []
            count = 0
            for model, label, model_name in models:
                if label in accuracy_data[datasetname]:
                    entry = accuracy_data[datasetname][label]
                    if entry['has_zs_direct'] and entry['has_zs_cot']:
                        cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']
                        total_delta += cot_delta
                        count += 1
                        totals.append(cot_delta)
                        all_deltas.append(cot_delta)
            if count > 0:
                if datasetname not in avg_cot_delta:
                    avg_cot_delta[datasetname] = []
                avg_cot_delta[datasetname].extend(totals)

    for ds in datasets:
        datasetname = ds[1]
        if datasetname in avg_cot_delta and isinstance(avg_cot_delta[datasetname], list):
            avg_cot_delta[datasetname] = sum(avg_cot_delta[datasetname]) / len(avg_cot_delta[datasetname]) * 100
    mean_delta = sum(all_deltas) / len(all_deltas)
    return avg_cot_delta, mean_delta

def get_cot_deltas_for_model(accuracy_data, model_name):

    # Find the label for the model_name
    label = None
    for m in models:
        if m[2] == model_name:
            label = m[1]
            break

    if label is None:
        print(f"Model {model_name} not found")
        return None

    cot_deltas = {}
    for ds in datasets:
        datasetname = ds[1]
        if datasetname in accuracy_data and label in accuracy_data[datasetname]:
            entry = accuracy_data[datasetname][label]
            if entry['has_zs_direct'] and entry['has_zs_cot']:
                cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']
                cot_deltas[datasetname] = cot_delta * 100  # multiply by 100 to get percentage
    return cot_deltas

def plot_cot_deltas(cot_deltas, axs, model_name, dataset_order, bar_colors, hide_x_axis=False, highlight_with_box=False):
    # Create DataFrame
    cot_delta_df = pd.DataFrame(list(cot_deltas.items()), columns=['Dataset', 'CoT_Delta'])
    # Use the specified dataset order
    cot_delta_df['Dataset'] = pd.Categorical(cot_delta_df['Dataset'], categories=dataset_order, ordered=True)
    cot_delta_df = cot_delta_df.sort_values('Dataset')

    # Plot
    axs.grid()
    # Use the same colors
    b = sns.barplot(
        y='CoT_Delta',
        x='Dataset',
        data=cot_delta_df,
        palette=bar_colors,
        zorder=10,
        ax=axs
    )

    # Plot the average accuracy (assuming mean_delta is defined)
    # If mean_delta is not defined, calculate it as the mean of 'CoT_Delta'
    mean_delta = cot_delta_df['CoT_Delta'].mean()
    sns.lineplot(
        x='Dataset',
        y=[mean_delta] * len(cot_delta_df),
        data=cot_delta_df,
        color='red',
        label='Mean CoT Improvement',
        linestyle='--',
        linewidth=2,
        zorder=11,
        ax=axs
    )

    if highlight_with_box:
        # Box this plot to make it highlighted
        axs.spines['top'].set_visible(True)
        axs.spines['right'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_linewidth(4)
        axs.spines['right'].set_linewidth(4)
        axs.spines['bottom'].set_linewidth(4)
        axs.spines['left'].set_linewidth(4)
        axs.spines['top'].set_color('black')
        axs.spines['right'].set_color('black')
        axs.spines['bottom'].set_color('black')
        axs.spines['left'].set_color('black')
        axs.spines['top'].set_zorder(10)
        axs.spines['right'].set_zorder(10)
        axs.spines['bottom'].set_zorder(10)
        axs.spines['left'].set_zorder(10)


    # Adjust the axes
    axs.set_title(f'{model_name.replace("Meta-", "")}', fontsize=20)
    axs.set_xlabel('CoT Improvement (%)', fontsize=15)
    axs.set_ylabel('')
    axs.tick_params(axis='x', labelsize=13)
    axs.tick_params(axis='y', labelsize=16)
    axs.set_axisbelow(True)
    # Remove legend
    axs.legend().remove()

    # Make y-axis grid line at 0 black
    axs.axhline(0, color='black', linewidth=1)

    # Set y-limits
    axs.set_ylim(-10, 70)

    # Set y-ticks for every 10 units
    axs.set_yticks(np.arange(-10, 80, 10))

    if hide_x_axis:
        axs.set_xlabel('')
        axs.set_xticklabels([])

    # Shade every other bar's background
    positions = axs.get_xticks()
    for i in range(len(positions)):
        if i % 2 == 0:
            axs.axvspan(positions[i] - 0.5, positions[i] + 0.5, facecolor='lightgray', alpha=0.5, zorder=0)
    tick_labels = ax.get_xticklabels()
    for tidx, tick in enumerate(tick_labels):
        text = tick.get_text()
        if text == 'Folio':
            tick.set_text('FOLIO')
        elif text == 'BigBench-Hard':
            tick.set_text('Big-Bench Hard')
        elif text == 'BigGen Bench':
            tick.set_text('BiGGen Bench')
        tick_labels[tidx] = tick
    axs.set_xticklabels(tick_labels)


if __name__ == '__main__':
    # Define groups
    groups = {
        'Commonsense': ['CommonsenseQA', 'StrategyQA', 'SiQA', 'PiQA', 'Winogrande'],
        'Knowledge': ['Arc Easy', 'Arc Challenge', 'MMLU', 'MMLU Pro', 'AGIEval LSAT'],
        'Mathematical': ['GSM8k', 'GSM8k-Hard', 'MATH', 'GPQA'],
        'Symbolic': ['BigBench-Hard', 'Folio', 'ContextHub'],
        'Soft Reasoning': ['BigGen Bench', 'MuSR', 'MuSiQue']
    }

    # Map each dataset to its group
    dataset_to_group = {}
    for group_name, dataset_list in groups.items():
        for datasetname in dataset_list:
            dataset_to_group[datasetname] = group_name

    # Initialize accuracy data
    accuracy_data = {}

    # File name to save or load data
    file_name = 'aggregated_accuracy_data_zs.json'

    # Load or generate accuracy data
    if not os.path.exists(file_name):
        pass
    else:
        with open(file_name, 'r') as f:
            accuracy_data = json.load(f)

    # Calculate average CoT deltas
    df2, acc_data = load_accuracy_data(force_reload=False if len(accuracy_data) is None else True)
    avg_accs, mean_delta = get_avg_acc_deltas(acc_data)

    # Initialize variables for tracking and grouping
    groups_list = [
        'Commonsense',
        'Knowledge',
        'Soft Reasoning',
        'Symbolic',
        'Mathematical'
    ]
    group_deltas = {}
    data_list = []

    # Process data for plotting
    for dataset, datasetname in datasets:
        if datasetname in accuracy_data:
            group_name = dataset_to_group.get(datasetname, 'Other')
            if group_name not in group_deltas:
                group_deltas[group_name] = []
            for model, label, model_name in models:
                if label in accuracy_data[datasetname]:
                    entry = accuracy_data[datasetname][label]
                    is_sig = False
                    if entry.get('has_zs_direct') and entry.get('has_zs_cot'):
                        sig_results = entry['sig_test_results']
                        is_sig = sig_results['p_value'] <= 0.05 / len(models)
                        actual_cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']
                        group_deltas[group_name].append(actual_cot_delta)
                        data_list.append({
                            'Dataset': datasetname,
                            'Group': group_name,
                            'Model': model_name,
                            'Accuracy': entry['zero_shot_direct'],
                            'Type': 'Zero-Shot Direct',
                            'CoT_Delta': actual_cot_delta * 100,
                            'Significant': is_sig
                        })
                        data_list.append({
                            'Dataset': datasetname,
                            'Group': group_name,
                            'Model': model_name,
                            'Accuracy': entry['zero_shot_cot'],
                            'Type': 'Zero-Shot CoT',
                            'CoT_Delta': actual_cot_delta * 100,
                            'Significant': is_sig
                        })

    # Compute average CoT delta per group
    avg_cot_delta = {group: np.mean(deltas) for group, deltas in group_deltas.items() if deltas}

    # Convert the data into a DataFrame
    df = pd.DataFrame(data_list)

    # Group the data by Group, Model, and Type
    grouped_df = df.groupby(['Group', 'Model', 'Type']).agg({'Accuracy': 'mean'}).reset_index()

    # Pivot the DataFrame to compute CoT delta
    pivot_df = grouped_df.pivot_table(index=['Group', 'Model'], columns='Type', values='Accuracy').reset_index()
    pivot_df['CoT_Delta'] = pivot_df['Zero-Shot CoT'] - pivot_df['Zero-Shot Direct']

    # Prepare to plot
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.10)

    # Left column GridSpec: 5 rows, 1 column
    gs_left = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0, 0], hspace=0.3)
    ax_left_list = [fig.add_subplot(gs_left[i]) for i in range(5)]

    # Right column GridSpec: 3 rows, 2 columns
    num_rows = 3
    num_cols = 2
    gs_right = gridspec.GridSpecFromSubplotSpec(num_rows, num_cols, subplot_spec=gs[0, 1], hspace=0.2, wspace=0.10)
    axes_list = [fig.add_subplot(gs_right[i, j]) for i in range(num_rows) for j in range(num_cols)]

    # Prepare the data for left plots
    group_model_cot_delta = df.groupby(['Group', 'Model']).agg({'CoT_Delta': 'mean'}).reset_index()
    group_model_cot_delta['Group'] = pd.Categorical(group_model_cot_delta['Group'], categories=groups_list, ordered=True)

    all_math = group_model_cot_delta[group_model_cot_delta['Group'] == 'Mathematical']
    # sort all_math
    all_math = all_math.sort_values('CoT_Delta')

    # Left side plots: per-category average CoT delta per model
    for axidx, (ax, group_name) in enumerate(zip(ax_left_list, groups_list)):
        group_data = group_model_cot_delta[group_model_cot_delta['Group'] == group_name]

        # Sort group_data by the order in all_math
        # group_data = group_data.sort_values('Model', key=lambda x: x.map(all_math.set_index('Model')['CoT_Delta']))
        # Sort group_data by the order in the models array
        group_data = group_data.sort_values('Model', key=lambda x: x.map({m[2]: i for i, m in enumerate(models)}))

        b = sns.barplot(
            data=group_data,
            x='Model',
            y='CoT_Delta',
            ax=ax,
            edgecolor='white',
            errorbar=None,
            palette='viridis',
        )

        # Make y-axis grid line at 0 black
        ax.axhline(0, color='black', linewidth=1)


        ax.set_title(f'{group_name}', fontsize=20)
        if True or axidx != len(ax_left_list) - 1:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('CoT Improvement (%)', fontsize=15)
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=16)
        # rotate x-axis labels and align left
        tick_labels = ax.get_xticklabels()
        for tidx, tick in enumerate(tick_labels):
            text = tick.get_text()
            if text == 'Folio':
                tick.set_text('FOLIO')
            elif text == 'BigBench-Hard':
                tick.set_text('Big-Bench Hard')
            elif text == 'BigGen Bench':
                tick.set_text('BiGGen Bench')
            elif 'Meta-' in text:
                text = text.replace('Meta-', '')
                tick.set_text(text)
            tick_labels[tidx] = tick

        ax.set_xticklabels(tick_labels, rotation=-33, ha='left')
        ax.grid(True)
        ax.set_axisbelow(True)

        # set y-lims
        ax.set_ylim(-10, 40)

        if axidx != len(ax_left_list) - 1:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        # Shade every other bar's background
        positions = ax.get_xticks()
        for i in range(len(positions)):
            if i % 2 == 0:
                ax.axvspan(positions[i] - 0.5, positions[i] + 0.5, facecolor='lightgray', alpha=0.5, zorder=0)

        # Set y-ticks for every 10 units
        ax.set_yticks(np.arange(-10, 50, 10))
        # Remove x-lines
        ax.xaxis.grid(False)

    # Prepare data for right plots
    avg_accs, mean_delta = get_avg_acc_deltas(acc_data)
    # sort avg_accs
    avg_accs = {k: v for k, v in sorted(avg_accs.items(), key=lambda item: item[1])}
    sorted_datasets = list(avg_accs.keys())
    bar_colors = sns.color_palette("flare", len(sorted_datasets))

    # Right side plots: per-model CoT delta per dataset
    model_names = ['Averaged across models', 'Qwen 2 7b', 'Meta-Llama 3.1 8b', 'Meta-Llama 3.1 70b', 'Gpt-4o', 'Claude-3.5 Sonnet']


    for axidx, (ax, model_name) in enumerate(zip(axes_list, model_names)):
        if axidx == 0:
            cot_deltas = avg_accs
        else:
            cot_deltas = get_cot_deltas_for_model(acc_data, model_name)
        if cot_deltas:
            plot_cot_deltas(cot_deltas, ax, model_name, sorted_datasets, bar_colors,hide_x_axis=True if axidx < len(model_names) - 2 else False, highlight_with_box=axidx==0)
            if True or axidx < len(model_names) - 2:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('CoT Improvement (%)', fontsize=15)

            ax.set_ylabel('')
            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=16)
            # rotate x-axis labels and align left
            ax.set_xticklabels(ax.get_xticklabels(), rotation=-33, ha='left')
        else:
            print(f"No data for model {model_name}")

    # Remove unused axes (if any)
    if len(model_names) < len(axes_list):
        for ax in axes_list[len(model_names):]:
            fig.delaxes(ax)

    # Add a super title
    fig.suptitle('CoT vs Direct Answer Prompting in Zero-Shot Setting', fontsize=30, weight='bold', y=0.99)

    # Add text for the y-axis saying "CoT Improvement"
    fig.text(0.01, 0.5, 'CoT Improvement (%)', va='center', rotation='vertical', fontsize=25, weight='bold')

    # Add text for the x-axis saying "Model" for the first plot, dataset for the 2nd
    fig.text(0.19, 0.005, 'Model', ha='center', fontsize=25, weight='bold')
    fig.text(0.525, 0.005, 'Dataset', ha='center', fontsize=25, weight='bold')
    fig.text(0.83, 0.005, 'Dataset', ha='center', fontsize=25, weight='bold')



    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.96, top=0.92, bottom=0.13)

    # Save and show the figure
    plt.savefig('combined_plots.png', dpi=300)
    plt.show()
