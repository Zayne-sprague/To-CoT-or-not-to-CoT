"""Example script to show some of the utility functions we have to make evaluating easier"""

from src import cache
cache.enable()
from pathlib import Path

from src.model.model import Model
from key_handler import KeyHandler
from eval_datasets.types.musr import MuSRDataset
from experiments.utils import rollout

KeyHandler.set_env_key()
model = Model.load_model('openai/gpt-4o-mini-2024-07-18')

# Define the dataset. Some datasets are thirdparty files and have their custom loaders, like MuSR.
dataset = MuSRDataset(str(Path(__file__).parent.parent / 'eval_datasets/thirdparty/musr/murder_mystery.json'))

example = dataset[10]

zs_cot_prompt = example['zs_cot_messages']
zs_directanswer_prompt = example['zs_cotless_messages']

# How many times do you want to sample from the model?
NUMBER_OF_SAMPLES = 10

# If the API can only handle N requests at a time, you can set the batch size
BATCH_SIZE = 4

# Standard model arguments
max_completion_length = 4096
temperature = 0.7
top_p = 0.95

# We can call rollout() which is a wrapper function around the model.inference call and the dataset.evaluate_response
# function. It will return the average accuracy over the samples given the example and prompt. It will also return the
# rate at which these samples had unparsable responses (meaning our answer parser could not determine the answer given
# back by the model).  Finally, it returns a dictionary of various metrics useful for evaluation.
cot_accuracy, cot_unparsable_rate, cot_metrics = rollout(
    model,
    zs_cot_prompt,
    example,
    num_rollouts=NUMBER_OF_SAMPLES,
    batch_size=BATCH_SIZE,
    completion_length=max_completion_length,
    temperature=temperature,
    top_p=top_p
)

directanswer_accuracy, directanswer_unparsable_rate, directanswer_metrics = rollout(
    model,
    zs_directanswer_prompt,
    example,
    num_rollouts=NUMBER_OF_SAMPLES,
    batch_size=BATCH_SIZE,
    completion_length=max_completion_length,
    temperature=temperature,
    top_p=top_p
)

print(f'Direct Answer vs CoT Accuracy (Unparseable Rate)\nDA: {directanswer_accuracy:.2f} ({directanswer_unparsable_rate:.2f})\nCoT:{cot_accuracy:.2f} ({cot_unparsable_rate:.2f})')