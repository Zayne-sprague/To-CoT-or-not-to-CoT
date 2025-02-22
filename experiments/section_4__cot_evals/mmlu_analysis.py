"""
Generate the figure used for the MMLU analysis in the paper along with some other stats about how useful = is in the
context of CoT for MMLU and MMLU Pro.

"""
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datasets import load_dataset

import json
from pathlib import Path
import re

from pathlib import Path
import json

reload_data = True
filename = 'mmlu_manual_cluster_figure.json'

# Define models and datasets
models = [
    ('meta-llama__Llama-2-7b-chat-hf', 'L2 7b', 'Llama 2 7b'),
    ('mistralai__Mistral-7B-Instruct-v0.3', 'M 7b', 'Mistral 7b'),
    ('meta-llama__Meta-Llama-3.1-8B-Instruct', 'L3.1 8b', 'Llama 3.1 8b'),
    ('meta-llama__Meta-Llama-3.1-70B-Instruct', 'L 3.1 70b', 'Llama 3.1 70b'),
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

model_names = [x[1].lower() for x in models]
dataset_names = ['mmlu', 'mmlu_pro']

if reload_data or not Path(filename).exists():
    stats = []

    for (model_path, name, actual_name) in models:
        all_data = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___" + model_path.replace('/', '__'), 'mmlu')
        data = all_data['latest']
        # Extract the necessary statistics
        stats.extend([{
            'model': name,
            'dataset': 'mmlu',
            'subject': json.loads(x['additional_information']).get('src', json.loads(x['additional_information']).get('subject')),
            'question': x['question'],
            'answer': x['answer'],
            'choices': json.loads(x['choices'])['text'],
            'cot_correct': x['zero_shot_cot_is_correct'],
            'direct_correct': x['zero_shot_direct_is_correct'],
            'cot_helps': x['zero_shot_cot_is_correct'] and not x['zero_shot_direct_is_correct'],
            'cot_hurts': not x['zero_shot_cot_is_correct'] and x['zero_shot_direct_is_correct'],
            'cot_resp': x['zero_shot_cot_model_response'],
            'direct_resp': x['zero_shot_direct_model_response'],
        } for x in data ])

        all_data = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___" + model_path.replace('/', '__'), 'mmlu_pro')
        data = all_data['latest']

        stats.extend([{
            'model': name,
            'dataset': 'mmlu_pro',
            'subject': json.loads(x['additional_information']).get('src', json.loads(x['additional_information']).get('subject')),
            'question': x['question'],
            'answer': x['answer'],
            'choices': json.loads(x['choices'])['text'],
            'cot_correct': x['zero_shot_cot_is_correct'],
            'direct_correct': x['zero_shot_direct_is_correct'],
            'cot_helps': x['zero_shot_cot_is_correct'] and not x['zero_shot_direct_is_correct'],
            'cot_hurts': not x['zero_shot_cot_is_correct'] and x['zero_shot_direct_is_correct'],
            'cot_resp': x['zero_shot_cot_model_response'],
            'direct_resp': x['zero_shot_direct_model_response'],
        } for x in data ])
    with open(filename, 'w') as f:
        json.dump(stats, f)
else:
    with open(filename, 'r') as f:
        stats = json.load(f)

data = {
    'text': [f'dataset[{x["dataset"]}]_model[{x["model"]}] {x["question"]} {x["answer"]} {x["cot_resp"]}' for x in stats],
    'cot_correct': [x['cot_correct'] for x in stats],
    'direct_correct': [x['direct_correct'] for x in stats],
    'cot_helps': [x['cot_helps'] for x in stats],
    'cot_hurts': [x['cot_hurts'] for x in stats]
}

df = pd.DataFrame(data)


# Correctly label the data
def label_row(row):
    if row['cot_correct']:
        return 'cot_helps'
    elif row['cot_hurts']:
        return 'cot_hurts'
    else:
        return 'neither'


df['target'] = df.apply(label_row, axis=1)


# Function to preprocess text
def preprocess_text(text):
    # Remove non-alphanumeric characters
    # text = re.sub(r'\W+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    # text = re.sub(r'\d+', '', text)
    return text


df['text'] = df['text'].apply(preprocess_text)


# Example predefined function
def predefined_function__OLD(text):
    is_true_or_false = 'is true' in text or 'is false' in text
    num_of_numbers = sum(c.isdigit() for c in text)
    num_of_conjunctions = text.count('and') + text.count('or') + text.count(',') + text.count('but') + text.count('nor')
    has_fill_in_blank = '____' in text
    which_of_the_following = 'which of the following' in text
    equals_in_text = '=' in text
    length = len(text)

    features = [
        is_true_or_false,
        num_of_numbers,
        num_of_conjunctions,
        has_fill_in_blank,
        which_of_the_following,
        equals_in_text,
        length
    ]

    return np.array(features)

def predefined_function(text):
    num_of_numbers = sum(c.isdigit() for c in text)


    for didx, dataset in enumerate(dataset_names):
        for midx, modelname in enumerate(model_names):
            if f'dataset[{dataset}]_model[{modelname}]' in text:
                if 'he best answer is ' not in text.lower():
                    # return ((didx * len(model_names) + midx)*3)
                    return f"{dataset} {modelname} No Answer"
                if text.count('=') >= 1: #or num_of_numbers / len(text) > 0.01:
                #     return ((didx * len(model_names) + midx)*3) + 2
                    return f"{dataset} {modelname} with ="
                # return ((didx * len(model_names) + midx)*3) + 1
                return f"{dataset} {modelname} w/o ="
    return -1

# Apply the predefined function
embeddings_or_clusters = df['text'].apply(predefined_function)

# Check if the predefined function returns a vector or a cluster label
if isinstance(embeddings_or_clusters.iloc[0], np.ndarray):
    # Run K-means if the function returns vectors
    X = np.stack(embeddings_or_clusters.values)
    optimal_clusters = 10
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters

    # Plot clusters if using vectors
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('Clusters of Text Documents')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()
else:
    # Use the predefined clusters directly
    df['cluster'] = embeddings_or_clusters
    optimal_clusters = df['cluster'].nunique()

# Calculate delta of CoT_correct - direct_correct for each cluster
cluster_metrics = df.groupby('cluster').apply(
    lambda x: pd.Series({
        'cot_correct_mean': x['cot_correct'].mean(),
        'direct_correct_mean': x['direct_correct'].mean(),
        'delta_cot_direct': x['cot_correct'].mean() - x['direct_correct'].mean(),
        'N': len(x['cot_correct'])
    })
)

# Display cluster metrics
print(cluster_metrics)

# Ensure each cluster has meaningful differences
print("Checking for meaningful differences in clusters...")

for i in range(optimal_clusters):
    cluster_texts = df[df['cluster'] == i]['text'].values
    # print(f"Cluster {i}:")
    # print("Sample texts:", cluster_texts[:3])
    # print()

print('Process completed.')

# Plot for each model and dataset the cot delta with and without =
bar_width = 0.175
bar_gap = 0.15

x = []
tick_labels = []
current_x = 0
fig, ax = plt.subplots(figsize=(11, 4))

table_header = 'model,dataset,with eq,delta,% of improvement,N'.split(',')
mmlu_relative_improvement_table_rows = []
mmlu_pro_relative_improvement_table_rows = []

# Plot the bars for both datasets (mmlu_pro first, then mmlu)
for modelname in model_names:
    try:
        # mmlu_pro bars
        with_eq = \
        cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} with ='), 'delta_cot_direct'].values[
            0] * 100
    except Exception:
        print(f'No mmlu {modelname} with =')
        continue
    without_eq = \
    cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} w/o ='), 'delta_cot_direct'].values[
        0] * 100

    ax.bar(current_x, with_eq, width=bar_width, color='purple', alpha=0.5, label='MMLU With =', edgecolor='white', linewidth=3, hatch="x")
    # current_x += bar_width * 2
    ax.bar(current_x, without_eq, width=bar_width, color='purple', alpha=1.0, label='MMLU Without =', edgecolor='white', linewidth=2)
    current_x += bar_width * 2 + bar_gap

    portion_of_improvement_with_eq = ((with_eq/100) * cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} with ='), 'N'].values[0]) / (
            (with_eq + without_eq) *
            (cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} w/o ='), 'N'].values[0] + cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} with ='), 'N'].values[0])
    )
    portion_of_improvement_without_eq = ((without_eq/100) *
                                         cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} w/o ='), 'N'].values[0]) / ((with_eq + without_eq) * (cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} w/o ='), 'N'].values[0] + cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} with ='), 'N'].values[0]))

    percentage_of_delta_with_eq = portion_of_improvement_with_eq / (portion_of_improvement_without_eq + portion_of_improvement_with_eq)
    percentage_of_delta_without_eq = portion_of_improvement_without_eq / (portion_of_improvement_without_eq + portion_of_improvement_with_eq)

    # Maybe
    cluster_metrics.loc[
        cluster_metrics.index.str.contains(f'mmlu.*{modelname}', regex=True), 'delta_cot_direct'].sum() * 100

    # total_delta = cluster_metrics[cluster_metrics.str.contains(f'{modelname}')]
    print(f'{modelname} MMLU with =: {with_eq:.3f} ({percentage_of_delta_with_eq:.3f})')
    print(f'{modelname} MMLU w/o =: {without_eq:.3f} ({percentage_of_delta_without_eq:.3f})')

    # mmlu_relative_improvement_table_rows.append([modelname, 'MMLU', 'True', round(with_eq, 0), round(percentage_of_delta_with_eq*100, 0), cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} with ='), 'N'].values[0]])
    # mmlu_relative_improvement_table_rows.append([modelname, 'MMLU', 'False', round(without_eq,0), round(percentage_of_delta_without_eq*100,0), cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} w/o ='), 'N'].values[0]])
    mmlu_relative_improvement_table_rows.append([round(with_eq, 0), round(percentage_of_delta_with_eq*100, 0), cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} with ='), 'N'].values[0]])
    mmlu_relative_improvement_table_rows.append([round(without_eq,0), round(percentage_of_delta_without_eq*100,0), cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu {modelname} w/o ='), 'N'].values[0]])



current_x = bar_width
for modelname in model_names:
    # mmlu bars
    try:
        with_eq = \
        cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu_pro {modelname} with ='), 'delta_cot_direct'].values[
            0] * 100
    except Exception:
        print(f'No mmlu_pro {modelname} with =')
        continue
    without_eq = \
    cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu_pro {modelname} w/o ='), 'delta_cot_direct'].values[
        0] * 100

    ax.bar(current_x, with_eq, width=bar_width, color='blue', alpha=0.5, label='MMLU Pro With =', edgecolor='white', linewidth=3, hatch="x")
    # current_x += bar_width * 2
    ax.bar(current_x, without_eq, width=bar_width, color='blue', alpha=1.0, label='MMLU Pro Without =', edgecolor='white', linewidth=2)
    current_x += bar_width * 2 + bar_gap

    portion_of_improvement_with_eq = ((with_eq / 100) * cluster_metrics.loc[
        cluster_metrics.index.str.contains(f'mmlu_pro {modelname} with ='), 'N'].values[0]) / (
                                                 (with_eq + without_eq) * (cluster_metrics.loc[
                                             cluster_metrics.index.str.contains(
                                                 f'mmlu_pro {modelname} w/o ='), 'N'].values[0]) + cluster_metrics.loc[
                                                     cluster_metrics.index.str.contains(
                                                         f'mmlu_pro {modelname} with ='), 'N'].values[0])

    portion_of_improvement_without_eq = ((without_eq / 100) * cluster_metrics.loc[
        cluster_metrics.index.str.contains(f'mmlu_pro {modelname} w/o ='), 'N'].values[0]) / ((with_eq + without_eq) * (
    cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu_pro {modelname} w/o ='), 'N'].values[0]) +
                                                                                              cluster_metrics.loc[
                                                                                                  cluster_metrics.index.str.contains(
                                                                                                      f'mmlu_pro {modelname} with ='), 'N'].values[
                                                                                                  0])

    percentage_of_delta_with_eq = portion_of_improvement_with_eq / (
                portion_of_improvement_without_eq + portion_of_improvement_with_eq)
    percentage_of_delta_without_eq = portion_of_improvement_without_eq / (
                portion_of_improvement_without_eq + portion_of_improvement_with_eq)

    print(f'{modelname} MMLU Pro with =: {with_eq:.3f} ({percentage_of_delta_with_eq:.3f})')
    print(f'{modelname} MMLU Pro w/o =: {without_eq:.3f} ({percentage_of_delta_without_eq:.3f})')

    # relative_improvement_table_rows.append([modelname, 'MMLU Pro', 'True', round(with_eq, 0), round(percentage_of_delta_with_eq*100,0), cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu_pro {modelname} with ='), 'N'].values[0]])
    # relative_improvement_table_rows.append([modelname, 'MMLU Pro', 'False', round(without_eq, 0), round(percentage_of_delta_without_eq*100, 0), cluster_metrics.loc[cluster_metrics.index.str.contains(f'mmlu_pro {modelname} w/o ='), 'N'].values[0]])



# Customizing plot
num_ticks = len(model_names)
# tick_positions = np.linspace(0, current_x - bar_width, num_ticks)

import math
tick_positions =  [((bar_width*2)*i) + (bar_gap*i) + bar_width/2 for i in range(num_ticks)]
# tick_labels = [f'{model} with Equations' for model in model_names] + [f'{model} No Equations' for model in model_names]
tick_labels = []
for x in models:
    tick_labels.append(f'{x[2]}')
    # tick_labels.append(f'{x[1]} No Equations')

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=-20, ha='left', fontsize=12)
ax.set_ylabel('Delta (%)', fontsize=12)
ax.set_title('Improvement of CoT over direct on = vs. no =', fontsize=18, pad=20)

# Remove duplicate legend entries and adjust to show dataset color mapping
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]


by_label = dict(zip(labels, handles))

unique_labels = [label for i, label in enumerate(labels) if label not in labels[:i]]
unique_handles = [handles[labels.index(label)] for label in unique_labels]
ax.legend(unique_handles, unique_labels, fontsize=10, loc='upper left')

ax.tick_params(axis='y', labelsize=12)  # Set larger Y-axis tick labels

plt.tight_layout()
plt.grid()
# Place a margin the top a bit to give the title some space
plt.subplots_adjust(top=0.85)

# save as eq_clustering_perf
plt.savefig('eq_clustering_perf.png')
plt.show()


from tabulate import tabulate

# print(tabulate(relative_improvement_table_rows, headers=table_header, tablefmt='latex_booktabs'))