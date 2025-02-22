def qwen_handle_mc_direct(prompt):
    if prompt[-1]['role'] == 'assistant':
        prompt[-1]['content'] = 'The best answer is '
    else:
        prompt.append({'role': 'assistant', 'content': 'The best answer is '})
    return [
        {'role': 'system', 'content': 'You only give answers and you always give them in the format "The best answer is [the_answer_letter]" otherwise no credit will be given!'},
        *prompt
    ]