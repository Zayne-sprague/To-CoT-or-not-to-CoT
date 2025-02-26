"""Example script to show some of the utility functions we have to make evaluating easier"""

from src import cache
cache.enable()

from src.model.model import Model
from key_handler import KeyHandler
from eval_datasets.types.gsm8k import GSM8KDataset
from experiments.utils import rollout

KeyHandler.set_env_key()

# Some models allow for partial completions. vLLM hosted APIs and Claude both do in our repo. OpenAI and Gemini do not
# in our repo.
model = Model.load_model('anthropic/claude-3-haiku-20240307')

dataset = GSM8KDataset()

example = dataset[0]

zs_cot_prompt = example['zs_cot_messages']

# A partial completion means the model will pick up exactly where it left off in the generation of the assistant
# message. Another way to think of this is that you are "Warm starting" the generation for the model. For example:

zs_cot_prompt.append({
    'role': 'assistant',
    'content': 'Oh wow. This is a difficult problem. Let me start by'
})

# Now, the claude haiku model will continue it's generation from "Oh wow. This is a difficult problem. Let me start by"
models_completion_of_partial_response = model.parse_out(model.inference(zs_cot_prompt))

print('Partial Prompt: Oh wow. This is a difficult problem. Let me start by')
print(f"Models output: {models_completion_of_partial_response[0]}")

# We use this functionality a lot to help models avoid generating chains of thought (especially for math questions)

zs_directanswer_prompt = example['zs_cotless_messages']
orig_directanswer_response = model.parse_out(model.inference(zs_directanswer_prompt))

print("System prompt:")
print(zs_directanswer_prompt[0]["content"])
print("Prompt:")
print(zs_directanswer_prompt[1]['content'])

print("Models original output:")
print(orig_directanswer_response[0])

zs_directanswer_prompt.append({
    'role': 'assistant',
    'content': '$\\boxed{'
})

new_directanswer_response = model.parse_out(model.inference(zs_directanswer_prompt))
print("New model output with the partial completion $\\boxed{")
print(new_directanswer_response[0])


