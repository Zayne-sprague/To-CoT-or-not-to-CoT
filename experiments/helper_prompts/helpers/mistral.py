def mistral_zs_cot_prompt(prompt):
    ### Check if a prefix has already been defined in the prompt
    if prompt[-1]['role'] == 'assistant':
        return prompt
    ### Add a zs cot prefix
    prompt[1]['content']+= " Your final answer following must be a single capital letter choice chosen from the answer options. If your final answer is a word or a sentence instead, you will receive 0 credit."
    return [
        *prompt,
        {
            'role':"assistant",
            'content': "Let\'s think step by step"
        }
    ]