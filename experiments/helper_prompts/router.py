"""
Not all models understand the instructions the same :P so we need to apply some minor changes to prompts in special cases
to make sure the models are generating answers we can parse consistently.
"""

from experiments.helper_prompts.helpers.llama2 import ll2_math_zs_cot_prompt
from experiments.helper_prompts.helpers.gpt4o import gpt4o_mini_folio_zs_direct_prompt,gpt4o_mini_gsm8k_zs_direct_prompt,gpt4o_gsm8k_zs_direct_prompt,gpt4o_agieval_zs_direct_prompt
from experiments.helper_prompts.helpers.qwen2 import qwen2_72b_gsm8k_zs_direct_prompt,qwen2_7b_zs_cot_prompt
from experiments.helper_prompts.helpers.mistral import mistral_zs_cot_prompt
from experiments.helper_prompts.helpers.llama2 import ll2_math_zs_cot_prompt, ll2_gpqa_fs_cot_prompt, mistral_math_zs_direct_prompt, \
    mistral_gsm8k_hard_zs_direct_prompt, mistral_agi_lat_ar_hard_zs_cot_prompt, mistral_handle_gpqa_fs_direct

from experiments.helper_prompts.helpers.llama3 import ll3_gsm_zs_direct_prompt
from experiments.helper_prompts.helpers.phi import phi_gsm_zs_direct_prompt
from experiments.helper_prompts.helpers.gemma import gemma_zs_cot_folio
from experiments.helper_prompts.helpers.gemini import gemini_gsm8k_hard_zs_cot_prompt, gemini_pro_gsm8k_hard_zs_cot_prompt

from experiments.helper_prompts.helpers.qwen import qwen_handle_mc_direct
from experiments.helper_prompts.helpers.gpt4o import gpt4o_mini_folio_zs_direct_prompt,gpt4o_mini_gsm8k_zs_direct_prompt,\
    gpt4o_gsm8k_zs_direct_prompt,gpt4o_agieval_zs_direct_prompt, gpt4o_mini_bbh_zs_direct_prompt, o1_musr_zs_direct

def switch(
        model,
        task,
        prompt_setting,
        prompt
):
    if ('Mistral-7B' in model) and 'math' in task and 'zs_direct' in prompt_setting:
        return mistral_math_zs_direct_prompt(prompt)
    if ('Mistral-7B' in model or 'Qwen/Qwen2-7B-Instruct' in model) and 'gsm8k_hard' in task and 'zs_direct' in prompt_setting:
        return mistral_gsm8k_hard_zs_direct_prompt(prompt)
    if ('Mistral-7B' in model) and 'agieval_lsat_ar' in task and 'zs_cot' in prompt_setting:
        return mistral_agi_lat_ar_hard_zs_cot_prompt(prompt)
    if ('Llama-2-7b-chat-hf' in model) and 'math' in task and 'zs_cot' in prompt_setting:
        return ll2_math_zs_cot_prompt(prompt)
    if 'Llama-2-7b-chat-hf' in model and 'gpqa' in task and 'fs_cot' in prompt_setting:
        return ll2_gpqa_fs_cot_prompt(prompt)
    if 'gpt4-mini' in model and 'folio' in task and 'zs_direct' in prompt_setting:
        return gpt4o_mini_folio_zs_direct_prompt(prompt)
    if ('gpt4-mini' in model or 'gemini-1.5' in model) and 'gsm8k_hard' in task and 'zs_direct' in prompt_setting:
        return gpt4o_mini_gsm8k_zs_direct_prompt(prompt)
    if 'gemini-1.5-flash' in model and 'gsm8k_hard' in task and 'zs_cot' in prompt_setting:
        return gemini_gsm8k_hard_zs_cot_prompt(prompt)
    if 'gemini-1.5-' in model and 'folio' in task and 'zs_cot' in prompt_setting:
        prompt[0]['content'] = prompt[0]['content'].replace('"Answer', '"The best answer is ')
        prompt[1]['content'] = prompt[1]['content'].replace('"Answer',
                                                            '"The best answer is ') + ' Remember, you must always end your response with "The best answer is [the_answer_letter]" otherwise no credit will be given!'
    if 'gemini-1.5-flash' in model and 'mm_musr' in task and 'fs_cot' in prompt_setting:
        prompt[-1]['content'] += ' Remember, you must pick one option and you must always format the answer as "ANSWER: <your choice>" where your choice is the letter of the answer you choose.'

    if 'gemini-1.5-pro' in model and 'gsm8k' in task and 'zs_cot' in prompt_setting:
        return gemini_pro_gsm8k_hard_zs_cot_prompt(prompt)
    if 'gemini-1.5-pro' in model and 'gsm8k' == task and 'zs_direct' in prompt_setting:
        return gpt4o_mini_gsm8k_zs_direct_prompt(prompt)
    if 'mini' in model and 'bbh' in task and 'zs_direct' in prompt_setting:
        return gpt4o_mini_bbh_zs_direct_prompt(prompt)
    if 'gpt-4o' in model and 'bbh' in task and 'zs_direct' in prompt_setting:
        return gpt4o_mini_bbh_zs_direct_prompt(prompt)
    if 'gpt-4o' in model and 'gsm8k_hard' in task and 'zs_direct' in prompt_setting:
        return gpt4o_gsm8k_zs_direct_prompt(prompt)
    if 'gpt-4o' in model and 'agieval' in task and 'zs_direct' in prompt_setting:
        return gpt4o_agieval_zs_direct_prompt(prompt)
    if 'Qwen2-72B-Instruct' in model and 'gsm8k_hard' in task and 'zs_direct' in prompt_setting:
        return qwen2_72b_gsm8k_zs_direct_prompt(prompt)
    if 'Mistral' in model and 'zs_cot' in prompt_setting:
        special_tasks = ['folio','csqa','siqa','piqa','winogrande','arc','stratqa']
        for stask in special_tasks:
            if stask in task: 
                return mistral_zs_cot_prompt(prompt)
    if 'Qwen2-7B-Instruct' in model and 'zs_cot' in prompt_setting:
        special_tasks = ['folio','siqa','piqa','csqa','winogrande','arc','stratqa']
        for stask in special_tasks:
            if stask in task: 
                return qwen2_7b_zs_cot_prompt(prompt)
    if 'Qwen2-7B-Instruct' in model and 'fs_cot' in prompt_setting and 'csqa' in task:
        return qwen2_7b_zs_cot_prompt(prompt)
    if ('meta-llama/Meta-Llama-3.1-70B-Instruct' in model) and 'gsm8k' == task and 'direct' in prompt_setting:
        return ll3_gsm_zs_direct_prompt(prompt)
    if 'Phi' in model and 'gsm8k' == task and 'zs_direct' in prompt_setting:
        return phi_gsm_zs_direct_prompt(prompt)
    if 'Phi' in model and 'agieval_lsat_rc' in task and 'fs_cot' in prompt_setting:
        # prompt.append({'role': 'assistant', 'content': 'Let\'s think step by step.'})
        prompt[-1]['content'] += ' Let\'s think step by step, remember to give your answer in the format \"The answer is therefore <A, B, C, D, E>\" at the end.'
    if 'Qwen/Qwen2-7B-Instruct' in model and task in ['siqa', 'folio', 'piqa', 'gpqa'] and 'direct' in prompt_setting:
        return qwen_handle_mc_direct(prompt)
    if 'Phi' in model and 'zs_cot' in prompt_setting and task in ['folio', 'siqa', 'piqa', 'winogrande', 'contexthub_deductive_level1', 'contexthub_deductive_level2', 'contexthub_abductive_level1', 'contexthub_abductive_level2']:
        if prompt[-1]['role'] != 'assistant':
            prompt.append({'role': 'assistant', 'content': "Let\'s think step by step."})
        else:
            if prompt[0]['role'] != 'system':
                prompt.insert(0, {'role': 'system', 'content': 'You always think step by step before giving an answer.'})
            else:
                raise ValueError('Prompt already has a system message.')
    if 'Qwen/Qwen2-7B-Instruct' in model and task in ['stratqa'] and 'zs_cot' in prompt_setting:
        prompt[-1]['content'] += ' Let\'s think step by step.'
    if 'Mistral' in model and task in ['gpqa'] and 'fs_direct' in prompt_setting:
        return mistral_handle_gpqa_fs_direct(prompt)
    if 'gemma' in model and 'contexthub_deductive_level2' in task and 'zs_direct' in prompt_setting:
        return qwen_handle_mc_direct(prompt)
    if 'gemma' in model and 'folio' in task and 'zs_cot' in prompt_setting:
        return gemma_zs_cot_folio(prompt)
    if 'o1-' in model and 'musr' in task and 'zs_direct' in prompt_setting:
        return o1_musr_zs_direct(prompt)
    if 'o1-' in model and ('stratqa' in task or 'csqa' in task):
        prompt[-1]['content'] = prompt[-1]['content'].replace('Only write the answer. ', '')

    return prompt