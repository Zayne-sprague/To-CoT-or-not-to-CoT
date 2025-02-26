"""Example script to show how the main components of our repo work."""

# Set up the cache (re-running the script with the same prompts will not call the thirdparty endpoint.)
from src import cache
cache.enable()

from src.model.model import Model
from key_handler import KeyHandler
from eval_datasets.types.gsm8k import GSM8KDataset

# Sets all of your environment variables, all that's needed for this script is OPENAI_API_KEY.
KeyHandler.set_env_key()

# Define the model
model = Model.load_model('openai/gpt-4o-mini-2024-07-18')

# Define the dataset
dataset = GSM8KDataset()

# Get an example and it's zero shot cot/direct answer prompts, then get the models response for both.
example = dataset[0]

zs_cot_prompt = example['zs_cot_messages']
zs_directanswer_prompt = example['zs_cotless_messages']

models_cot_response = model.parse_out(model.inference(zs_cot_prompt))
print(f'Models CoT Response:\n{models_cot_response[0]}')

models_directanswer_response = model.parse_out(model.inference(zs_directanswer_prompt))

# Answer parsing using our custom answer parsers (every dataset has their own special parsers, but a lot of them share
# the same rules)
examples_cot_metrics = dataset.evaluate_response(models_cot_response, example)
examples_directanswer_metrics = dataset.evaluate_response(models_directanswer_response, example)

print(f"The correct answer: {example['answer']}")

answer_span_in_cot_response = examples_cot_metrics[0]["answer_span"]
print(f'CoTs extracted answer: {examples_cot_metrics[0]["model_response"][answer_span_in_cot_response[0]:answer_span_in_cot_response[1]]}')

print (f"CoT was correct: {examples_cot_metrics[0]['correct']}")
print(f"Direct Answer was correct: {examples_directanswer_metrics[0]['correct']}")