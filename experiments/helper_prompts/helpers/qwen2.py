def qwen2_72b_gsm8k_zs_direct_prompt(prompt):
    ### Change system message
    prompt[0]['content'] += """ Your response should only contain a single numerical value as your final answer in the following format: $\\boxed{answer}$. You should not put any equation, explanation, or computation inside the boxed format."""
    ### Change the user message
    prompt[1]['content'] += """ Your response should only contain a single numerical value as your final answer in the following format: $\\boxed{answer}$. You should not put any equation, explanation, or computation inside the boxed format."""
    return [
        *prompt
    ]
def qwen2_7b_zs_cot_prompt(prompt):
    ### Check if a prefix has already been defined in the prompt
    assert prompt[1]['role'] == 'user'
    prompt[1]['content']+= " Your final answer following must be a single capital letter choice chosen from the answer options. If your final answer is a word or a sentence instead, you will receive 0 credit."
    return [
        *prompt,
        {
            'role':"assistant",
            'content': "Let\'s think step by step"
        }
    ]