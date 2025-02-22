def ll2_math_zs_cot_prompt(prompt):
    prompt[0]['content'] += ' Remember, you must format your final answer as $\\boxed{answer}$ otherwise you will receive no credit!'
    return [
        {'role': 'system', 'content': 'Explain your reasoning step-by-step and always give your answer in the format $\\boxed{answer}$ otherwise no credit will be given.'},
        *prompt
    ]

def mistral_math_zs_direct_prompt(prompt):
    prompt[0]['content'] += ' Remember, you must answer by only saying $\\boxed{answer}$ where "answer" here is only the answer and nothing else. Otherwise you will receive no credit!'
    return [
        {'role': 'system', 'content': 'You only give answers and you always give them in the format $\\boxed{answer}$. Not following these instructions result in no credit.'},
        *prompt
    ]


def mistral_gsm8k_hard_zs_direct_prompt(prompt):
    prompt[0]['content'] += ' Remember, you must answer by only saying $\\\\boxed{answer}$ where "answer" here is only the answer and nothing else. Otherwise you will receive no credit!'
    prompt[-1]['content'] = 'I will give the answer now. $\\\\boxed{'
    return [
        {'role': 'system', 'content': 'You only give answers and you always give them in the format $\\\\boxed{answer}$. Not following these instructions result in no credit.'},
        *prompt,
    ]

def mistral_agi_lat_ar_hard_zs_cot_prompt(prompt):
    prompt[0]['content'] += ' Every question has exactly 1 answer, you never abstain from answering, you always pick at least one choice.'
    prompt[-1]['content'] += ' Remember, you must always end your response with "The answer is therefore <answer letter>" and you must always give an answer. Otherwise no credit will be given!'
    return [
        *prompt
    ]



def ll2_gpqa_fs_cot_prompt(prompt):
    return prompt
    # prompt[-1]['content'] += ' Remember, you must always end your response with "The best answer is [the_answer_letter]" otherwise no credit will be given!'
    # return [{
    #     'role': 'system',
    #     'content': 'You will answer the following question and always end your response with "The best answer is [the_answer_letter]" otherwise no credit will be given'
    # },
    #     *prompt
    # ]

def mistral_handle_gpqa_fs_direct(prompt):
    if prompt[-1]['role'] == 'assistant':
        prompt[-1]['content'] = 'The best answer is '
    else:
        prompt.append({'role': 'assistant', 'content': 'The best answer is '})
    return [
        {'role': 'system', 'content': 'You only give answers and you always give them in the format "The best answer is [the_answer_letter]" otherwise no credit will be given!'},
        *prompt
    ]