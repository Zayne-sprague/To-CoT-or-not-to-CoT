def gpt4o_mini_bbh_zs_direct_prompt(prompt):
    ### Change system message
    # prompt[0]['content'] += """ Do not provide any explanation to your answer. Your response should only contain the answer in the following format:\"Answer: <your answer>\""""
    prompt[-1]['content'] += """ Do not provide any explanation to your answer. Your response should only contain the answer in the following format:\"The best answer is [the_answer_letter]\", otherwise no credit will be given!"""
    return [
        {'role': 'system',
         'content': 'You only give answers and you always give them in the format "The best answer is [the_answer_letter]" otherwise no credit will be given!'},
        *prompt
    ]

def gpt4o_mini_folio_zs_direct_prompt(prompt):
    ### Change system message
    prompt[0]['content'] += """ Do not provide any explanation to your answer. Your response should only contain the answer in the following format:\"Answer: <your answer>\""""
    return [
        *prompt
    ]
def gpt4o_mini_gsm8k_zs_direct_prompt(prompt):
    ### Change system message
    prompt[0]['content'] += """ Your response should only contain a single numerical value as your final answer in the following format: $\\boxed{answer}$. You should not put any equation, explanation, or computation inside the boxed format. """
    prompt[-1]['content'] += ' Remember, you must answer by only saying $\\boxed{answer}$ where "answer" here is only the answer and nothing else. Otherwise you will receive no credit!'
    return [
        *prompt
    ]
def gpt4o_gsm8k_zs_direct_prompt(prompt):
    ### Change system message
    prompt[0]['content'] += """ Your response should only contain a single numerical value as your final answer in the following format: $\\boxed{answer}$. You should not put any equation, explanation, or computation inside the boxed format."""
    ### Change the user message
    prompt[1]['content'] += """ Your response should only contain a single numerical value as your final answer in the following format: $\\boxed{answer}$. You should not put any equation, explanation, or computation inside the boxed format."""
    return [
        *prompt
    ]
def gpt4o_agieval_zs_direct_prompt(prompt):
    ### Change system message
    prompt[0]['content'] += """Give your answer in the format \"The answer is therefore <A, B, C, D, E>\""""
    return [
        *prompt
    ]

# def ll2_gpqa_fs_cot_prompt(prompt):
    # prompt[-1]['content'] += ' Remember, you must always end your response with "The best answer is [the_answer_letter]" otherwise no credit will be given!'
    # return [{
    #     'role': 'system',
    #     'content': 'You will answer the following question and always end your response with "The best answer is [the_answer_letter]" otherwise no credit will be given'
    # },
    #     *prompt
    # ]


def o1_musr_zs_direct(prompt):
    prompt[-1]['content'] = prompt[-1]['content'].replace("Please only write the answer. ", '').rstrip()
    return prompt