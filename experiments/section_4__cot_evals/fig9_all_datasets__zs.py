import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json, os, math
from datasets import load_dataset
from key_handler import KeyHandler
from experiments.sig_tests import sigtest


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

for x in models:
    print(x[0])

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

# Define clusters for coloring titles
symbolic = ['ContextHub Deductive L1', 'ContextHub Deductive L2', 'ContextHub Abductive L1', 'ContextHub Abductive L2', 'MATH']
semi_symbolic = ['AGIEval LSAT LR', 'AGIEval LSAT AR', 'AGIEval LSAT RC', 'Murder Mysteries MuSR', 'Team Allocations MuSR', 'Object Placements MuSR', 'GPQA', 'GSM8k', 'MMLU', 'MMLU Pro']
non_symbolic = ['CommonsenseQA', 'StrategyQA', 'SiQA', 'PiQA', 'Arc Easy', 'Arc Challenge']

# Initialize accuracy data
accuracy_data = {}

# File name to save or load data
file_name = 'accuracy_data_zs.json'

force_reload = False
fetch_if_empty = True



# Load or generate accuracy data
if not os.path.exists(file_name) or force_reload:
    for (model, label, model_name) in models:

        for (dataset, datasetname) in datasets:
            try:
                all_data = load_dataset(f"TAUR-Lab/Taur_CoT_Analysis_Project___" + model.replace('/', '__'),
                                        dataset.replace("/", "__"), token=KeyHandler.hf_key)

                data = all_data["latest"]
            except Exception:
                print(f"Failed to load data for {dataset} in {model}")
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
                filtering_data = [x for x in filtering_data if x[1]['capability'] not in ['multilingual', 'tool_usage']]
                data = [x[0] for x in filtering_data if look_up_table.get(x[1]['capability'] + "_" + x[1]['task'], 'yes') not in ['no', 'maybe']]



            total = len(data)

            zs_sample = [1 if x['zero_shot_direct_is_correct'] else 0 for x in data]
            cot_sample = [1 if x['zero_shot_cot_is_correct'] else 0 for x in data]

            # sig__1_to_2, pval__1_to_2 = sigtest(raw_p1_data, raw_p1_to_2_data, sig_test_alpha, sig_test_test_name)

            significant, p_value = sigtest(zs_sample, cot_sample, 0.05, "bernoulli")

            zs_direct_correct = sum([1 if x['zero_shot_direct_is_correct'] else 0 for x in data])
            zs_cot_correct = sum([1 if x['zero_shot_cot_is_correct'] else 0 for x in data])

            custom_default_message = "[invalid] - configuration was not run"

            has_zs_direct = any([x['zero_shot_direct_model_response'] != custom_default_message for x in data])
            has_zs_cot = any([x['zero_shot_cot_model_response'] != custom_default_message for x in data])

            if datasetname not in accuracy_data:
                accuracy_data[datasetname] = {}

            accuracy_data[datasetname][label] = {
                'zero_shot_direct': zs_direct_correct / total,
                'zero_shot_cot': zs_cot_correct / total,
                'has_zs_direct': has_zs_direct,
                'has_zs_cot': has_zs_cot,
                'sig_test_results': {
                    'significant': significant,
                    'p_value': p_value
                }
            }

    with open(file_name, 'w') as f:
        json.dump(accuracy_data, f)

else:
    with open(file_name, 'r') as f:
        accuracy_data = json.load(f)

    if fetch_if_empty:
        for (model, label, real_name) in models:

            for (dataset, datasetname) in datasets:
                if 'Phi' in model:
                    pass
                elif accuracy_data.get(datasetname, {}).get(label) is not None and accuracy_data[datasetname][label].get('has_zs_direct')  and accuracy_data[datasetname][label].get('has_zs_cot'):
                    continue
                print(f"Fetching data for {dataset} in {model}")
                try:
                    all_data = load_dataset(f"TAUR-Lab/Taur_CoT_Analysis_Project___" + model.replace('/', '__'),
                                            dataset.replace("/", "__"), token=KeyHandler.hf_key)

                    data = all_data["latest"]
                except Exception:
                    print(f"Failed to load data for {dataset} in {model}")
                    continue

                total = len(data)

                zs_sample = [1 if x['zero_shot_direct_is_correct'] else 0 for x in data]
                cot_sample = [1 if x['zero_shot_cot_is_correct'] else 0 for x in data]

                # sig__1_to_2, pval__1_to_2 = sigtest(raw_p1_data, raw_p1_to_2_data, sig_test_alpha, sig_test_test_name)

                significant, p_value = sigtest(zs_sample, cot_sample, 0.05, "bernoulli")

                zs_direct_correct = sum([1 if x['zero_shot_direct_is_correct'] else 0 for x in data])
                zs_cot_correct = sum([1 if x['zero_shot_cot_is_correct'] else 0 for x in data])

                custom_default_message = "[invalid] - configuration was not run"

                has_zs_direct = any([x['zero_shot_direct_model_response'] != custom_default_message for x in data])
                has_zs_cot = any([x['zero_shot_cot_model_response'] != custom_default_message for x in data])

                if datasetname not in accuracy_data:
                    accuracy_data[datasetname] = {}

                accuracy_data[datasetname][label] = {
                    'zero_shot_direct': zs_direct_correct / total,
                    'zero_shot_cot': zs_cot_correct / total,
                    'has_zs_direct': has_zs_direct,
                    'has_zs_cot': has_zs_cot,
                    'sig_test_results': {
                        'significant': significant,
                        'p_value': p_value
                    }
                }
        with open(file_name, 'w') as f:
            json.dump(accuracy_data, f)


below_stratqa = 0
sig_below_stratqa = 0
sig_datasets = []
sig_models = []

avg_cot_delta = {}
for dataset, datasetname in datasets:
    if datasetname in accuracy_data:
        total_delta = 0
        totals = []
        count = 0
        for model, label, model_name in models:
            if label in accuracy_data[datasetname]:
                entry = accuracy_data[datasetname][label]
                if entry['has_zs_direct'] and entry['has_zs_cot']:
                    cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']
                    # cot_delta = ((1 - entry['zero_shot_direct']) - (1 - entry['zero_shot_cot'])) / max(1e-10, (1 - entry['zero_shot_cot']))
                    total_delta += cot_delta
                    count += 1
                    totals.append(cot_delta)
        if count > 0:
            avg_cot_delta[datasetname] = total_delta / count

            # get the mode
            avg_cot_delta[datasetname] = list(sorted(totals))[len(totals) // 2]


# Convert the data into a long-form DataFrame
data_list = []
for dataset, datasetname in datasets:
    if datasetname in accuracy_data:
        for model, label, model_name in models:
            if label in accuracy_data[datasetname]:
                entry = accuracy_data[datasetname][label]
                is_sig = False
                if entry.get('has_zs_direct') and entry.get('has_zs_cot'):
                    sig_results = entry['sig_test_results']

                    # Bonferroni Correct
                    is_sig = sig_results['p_value'] <= 0.05 / len(models)

                    actual_cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']
                    cot_delta = avg_cot_delta[datasetname]


                    if cot_delta <= 0.05 and cot_delta >= 0.0 and actual_cot_delta > 0.0:
                        below_stratqa += 1
                        if is_sig:
                            sig_below_stratqa += 1

                            sig_datasets.append(datasetname)
                            sig_models.append(model_name)
                    data_list.append({
                        'Dataset': datasetname,
                        'Model': model_name,
                        'Accuracy': entry['zero_shot_direct'],
                        'Type': 'zero-shot direct answer',
                        'CoT_Delta': cot_delta,
                        'Significant': is_sig
                    })
                    data_list.append({
                        'Dataset': datasetname,
                        'Model': model_name,
                        'Accuracy': entry['zero_shot_cot'],
                        'Type': 'zero-shot CoT',
                        'CoT_Delta': cot_delta,
                        'Significant': is_sig
                    })
                else:
                    print("Skipping model", model_name, "for dataset", datasetname, "because it has no data")

df = pd.DataFrame(data_list)

print(f"Below 0.05: {below_stratqa}")
print(f"Sig Below 0.05: {sig_below_stratqa}")
from collections import Counter
print(Counter(sig_datasets).most_common())
print(Counter(sig_models).most_common())
# Calculate average CoT delta for each dataset
avg_cot_delta = {}
for dataset, datasetname in datasets:
    if datasetname in accuracy_data:
        total_delta = 0
        totals = []
        count = 0
        for model, label, model_name in models:
            if label in accuracy_data[datasetname]:
                entry = accuracy_data[datasetname][label]

                if entry['has_zs_direct'] and entry['has_zs_cot']:
                    cot_delta = entry['zero_shot_cot'] - entry['zero_shot_direct']
                    # cot_delta = ((1 - entry['zero_shot_direct']) - (1 - entry['zero_shot_cot'])) / max(1e-10, (1 - entry['zero_shot_cot']))
                    total_delta += cot_delta
                    count += 1
                    totals.append(cot_delta)
                else:
                    print("Skipping model", model_name, "for dataset", datasetname, "because it has no data")
            else:
                print("Skipping model", model_name, "for dataset", datasetname, "because it has no data")
        if count > 0:
            avg_cot_delta[datasetname] = total_delta / count

            # get the mode
            avg_cot_delta[datasetname] = list(sorted(totals))[len(totals) // 2]

# Sort datasets by average CoT delta
sorted_datasets = sorted(avg_cot_delta.keys(), key=lambda x: avg_cot_delta[x])

# Calculate the number of rows needed
num_plots = len(sorted_datasets)
num_cols = 3
num_rows = math.ceil(num_plots / num_cols)

# Adjust figure size based on number of rows
fig = plt.figure(figsize=(56, num_rows * 10.0))
grid = plt.GridSpec(num_rows, num_cols, hspace=.80, wspace=0.2)

# Add the individual plots for each dataset in the sorted order
for i, datasetname in enumerate(sorted_datasets):
    ax = fig.add_subplot(grid[math.floor(i / num_cols), i % num_cols])
    group_data = df[df['Dataset'] == datasetname]

    barplot = sns.barplot(
        data=group_data,
        x='Model',
        y='Accuracy',
        hue='Type',
        ax=ax,
        edgecolor='white',  # White outline for bars
        errorbar=None,  # Don't show error bars
    )

    # When DF["Significant"] is true, outline the bars in orange
    # for bar, sig in zip(barplot.patches, group_data['Significant']):
    #     if sig:
    #         bar.set_edgecolor('red')
    #         bar.set_linewidth(2)

    # Determine the color of the title based on the category
    # if datasetname in symbolic:
    #     title_color = 'blue'
    # elif datasetname in semi_symbolic:
    #     title_color = 'orange'
    # else:
    #     title_color = 'red'

    if datasetname == 'Folio':
        datasetname = 'FOLIO'
    if datasetname == 'BigBench-Hard':
        datasetname = 'Big-Bench Hard'
    if datasetname == 'BigGen Bench':
        datasetname = 'BiGGen Bench'

    ax.set_title(f'{datasetname}', fontsize=50)#, color=title_color)
    tick_labels = ax.get_xticklabels()
    for tidx, tick in enumerate(tick_labels):
        text = tick.get_text()
        if 'Meta-' in text:
            text = text.replace('Meta-', '')
            tick.set_text(text)
        tick_labels[tidx] = tick
    ax.set_xticklabels(tick_labels, rotation=-25, ha='left', fontsize=32)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_xlabel('')  # Remove the x-axis label
    ax.set_axisbelow(True)  # Ensure grid lines are behind the bars

    # Set Y-axis ticks
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='y', labelsize=44)  # Set larger Y-axis tick labels

    # Add the "Accuracy" label on the leftmost plots only
    if i % num_cols == 0:
        ax.set_ylabel("Accuracy", fontsize=50)
    else:
        ax.set_ylabel("")

    # Remove individual legends
    ax.get_legend().remove()



# Add a single title and legend for the entire figure
# fig.suptitle('Zero-shot CoT vs Direct Accuracy (sorted by CoT Delta)', fontsize=64, weight='bold', y=.99)
fig.suptitle('CoT vs direct answer prompting in zero-shot setting (sorted by CoT delta)', fontsize=64, weight='bold', y=.99)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=50, bbox_to_anchor=(0.5, .98))

plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.05)  # Adjust margins
plt.savefig('exp_0_zs.png')
plt.show()