"""
This script just shows how to use the Model class to load and call models with a silly prompt as extra.
"""
from src import cache
cache.enable()

from src.model.model import Model
from key_handler import KeyHandler

KeyHandler.set_env_key()

with cache.static_context:
    print("DIRECT:")
    gpt4 = Model.load_model('openai/gpt-4o-2024-11-20')
    claude3 = Model.load_model('anthropic/claude-3-5-sonnet-20241022')
    # gemini = Model.load_model('gemini/google/gemini-1.5-pro-001')
    gemini = Model.load_model('gemini/google/gemini-2.0-flash-thinking-exp-1219')

    prompt = '''Yes or no. (Only answer yes or no)

If a giraffe had human legs with the ego of an actor and always dressed as a clown while using stilts perfectly, would it be able to eat food off a tree like other giraffes?'''

    gpt4_response = gpt4.parse_out(gpt4.inference(prompt, temperature=1.0, max_tokens=100))
    claude3_response = claude3.parse_out(claude3.inference(prompt, temperature=1.0, max_tokens=100))
    gemini_response = gemini.parse_out(gemini.inference(prompt, temperature=1.0, max_tokens=100))

    print(f'GPT-4o: {gpt4_response}')
    print(f'Claude-3.5 sonnet: {claude3_response}')
    print(f'Gemini: {gemini_response}')

    print("\n\n\nNO CONSTRAINT:")

    prompt = '''If a giraffe had human legs with the ego of an actor and always dressed as a clown while using stilts perfectly, would it be able to eat food off a tree like other giraffes?'''

    gpt4_response = gpt4.parse_out(gpt4.inference(prompt, temperature=1.0, max_tokens=100))
    claude3_response = claude3.parse_out(claude3.inference(prompt, temperature=1.0, max_tokens=1000))[0]
    gemini_response = gemini.parse_out(gemini.inference(prompt, max_tokens=8000, temperature=1.0))[0][0]['text']

    print(f'GPT-4o: {gpt4_response}')
    print(f'Claude-3.5 sonnet: {claude3_response}')
    print(f'Gemini: {gemini_response}')


    print("\n\n\nCOT CONSTRAINT:")

    prompt = '''Yes or no. If a giraffe had human legs with the ego of an actor and always dressed as a clown while using stilts perfectly, would it be able to eat food off a tree like other giraffes?\n\nThink step by step.'''

    gpt4_response = gpt4.parse_out(gpt4.inference(prompt, temperature=1.0, max_tokens=100))
    claude3_response = claude3.parse_out(claude3.inference(prompt, temperature=1.0, max_tokens=1000))[0]
    gemini_response = gemini.parse_out(gemini.inference(prompt, temperature=1.0, max_tokens=1000))[0][0]['text']

    print(f'GPT-4o: {gpt4_response}')
    print(f'Claude-3.5 sonnet: {claude3_response}')
    print(f'Gemini: {gemini_response}')

