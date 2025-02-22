def gemma_zs_cot_folio(prompt):
    prompt[0]['content'] = prompt[0]['content'].replace('"Answer', '"The best answer is ')
    prompt[1]['content'] = prompt[1]['content'].replace('"Answer', '"The best answer is ') + ' Remember, you must always end your response with "The best answer is [the_answer_letter]" otherwise no credit will be given!'

    return [
        *prompt
    ]