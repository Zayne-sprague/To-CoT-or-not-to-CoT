def gemini_gsm8k_hard_zs_cot_prompt(prompt):
    prompt[-1]['content'] += ' Remember, you must give your answer in the format $\\\\boxed{answer}$ otherwise you will receive no credit!'
    prompt[-1]['content'] = prompt[-1]['content'].replace("Remember to box your final answer via $\\boxed{your answer}$", " You should format your final answer as \"The answer is $\\\\boxed{answer}$\" where answer is the numerical answer to the question." )
    return prompt

def gemini_pro_gsm8k_hard_zs_cot_prompt(prompt):
    prompt[0]['content'] = prompt[0]['content'].replace(" everytime!", " otherwise no credit will be given!")
    prompt[-1]['content'] += ' Remember, you must give your answer in the format $\\\\boxed{answer}$ otherwise you will receive no credit!'
    prompt[-1]['content'] = prompt[-1]['content'].replace("Remember to box your final answer via $\\boxed{your answer}$", " You should format your final answer as \"The answer is $\\\\boxed{answer}$\" where answer is the numerical answer to the question." )
    return prompt