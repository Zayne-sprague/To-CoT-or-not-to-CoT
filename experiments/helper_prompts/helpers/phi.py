def phi_gsm_zs_direct_prompt(prompt):
    prompt[0]['content'] += ' Remember, you must answer by only saying $\\boxed{answer}$ where "answer" here is only the answer and nothing else. Otherwise you will receive no credit!'
    return [
        {'role': 'system', 'content': 'You only give answers and you always give them in the format $\\boxed{answer}$. Not following these instructions result in no credit.'},
        *prompt,
        {'role': 'assistant', 'content': 'I will give the answer now. $\\\\boxed{'}
    ]

