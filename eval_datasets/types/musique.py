import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset


class MusiqueDataset(ReasoningDataset):
    # 1000 for olmo
    average_token_len = 1500

    def __init__(self, path_or_url='dgslibisey/MuSiQue', split='validation', subset="all", *args, **kwargs):
        super().__init__(path_or_url + ':' + split, generating_paraphrase=True, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.musique

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url)[split]]

        for ex in dataset:
            answers = [ex['answer']] + ex['answer_aliases']
            paragraph_context = [item['paragraph_text'] for item in ex['paragraphs'] if item['is_supporting']]
            paragraph_context = '\n\n'.join(paragraph_context) + "\n\n"
            prompt = f'''
You are given the following paragraphs:
{paragraph_context}

Based on the paragraphs above, please answer the question below:
{ex["question"]}

Think step by step before giving your final answer to the question. Note that there will always be an answer to the question and you should only base your answer on the paragraph provided. When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.
                        '''.strip()
            cotless_prompt = f'''
You are given the following paragraphs:
{paragraph_context}

Based on the paragraphs above, please answer the question below:
{ex["question"]}

Note that there will always be an answer to the question and you should only base your answer on the paragraph provided. You will only say the answer and the answer alone in the format: "Answer: <your answer>".  You must always give an answer at the end.
                        '''.strip()
            
            examples.append({
                **ex,
                'dataset_type': self.dataset_types.musique,
                'prompt_parts': {'zs_cot_prompt': prompt, 'zs_cotless_prompt': cotless_prompt},
                'paragraph_context': paragraph_context,
                'answers': answers,
            })
            
        random.Random(1).shuffle(examples)
        return examples


    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            *args, **kwargs
    ):
        answers = [item.lower() for item in example['answers']]

        returned_answers = []

        for resp in model_responses:
            try:
                ans = resp.split('Answer: ')[1].strip().lower()
                correct = ans in answers
                # print('hey0')
            except Exception as e:
                ans = None
                correct = False
            returned_answers.append({
                'model_response': resp,
                'answer_line': ans,
                'correct': correct,
                'answer_randomly_sampled': False,
                'model_answer': ans,
                **example
            })

        return returned_answers


    @classmethod
    def custom_evaluate_response(
            cls,
            model_responses,
            example,
            *args,
            **kwargs
    ):
        answers = [example['answer']]
        answer_labels = [f"{example['answerKey']}"]
        incorrect_answers = [*[x for x in example['choices']['text'] if x not in answers]]
        incorrect_answer_labels = [*[f'{x}' for x in example['choices']['label'] if x not in answer_labels]]

        returned_answers = []

        for resp in model_responses:
            parsed_resp = resp.strip()
            found = False

            try:
                line = [x for x in parsed_resp.split('\n') if x != ''][-1]  # Only check the last line/chunk of text.
            except Exception:
                continue

            potential_answer_segments = []
            if '(' in line and ')' in line:
                potential_answer_segments.append(line.split('(')[1].split(')')[0].strip())
            if '$' in line:
                try:
                    potential_answer_segments.append(line.split('$')[1].split('$')[0].strip())
                except Exception:
                    pass
            if 'Answer: ' in line:
                try:
                    potential_answer_segments.append(line.split('Answer: ')[1].split(' ')[0].strip())
                except Exception:
                    pass
            if 'ANSWER: ' in line:
                try:
                    potential_answer_segments.append(line.split('ANSWER: ')[1].split(' ')[0].strip())
                except Exception:
                    pass
            if 'answer: ' in line:
                try:
                    potential_answer_segments.append(line.split('answer: ')[1].split(' ')[0].strip())
                except Exception:
                    pass

            for line in potential_answer_segments:
                correct = any([x.lower() in line.lower() for x in answers])

                correct = correct or any(
                    [any([f'{x}' == cls.parse_ans(y) for y in
                          line.split(' ')]) for x
                     in answer_labels])

                incorrect = any(
                    [x.lower() in line.lower() for x
                     in incorrect_answers])

                incorrect = incorrect or any(
                    [any([f'{x}' == cls.parse_ans(y) for y in
                          line.split(' ')]) for x in incorrect_answer_labels])

                chosen_answers = []

                for xidx, x in enumerate(answers):
                    if x.lower() in line.lower():
                        chosen_answers.append(answer_labels[xidx])
                for x in answer_labels:
                    for y in line.split(' '):
                        if f'{x}' == cls.parse_ans(y):
                            chosen_answers.append(x)

                for xidx, x in enumerate(incorrect_answers):
                    if x.lower() in line.lower():
                        chosen_answers.append(incorrect_answer_labels[xidx])

                for x in incorrect_answer_labels:
                    for y in line.split(' '):
                        if f'{x}' == cls.parse_ans(y):
                            chosen_answers.append(x)

                if correct and not incorrect:
                    returned_answers.append(
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': line,
                         'correct': True,
                         'answer_randomly_sampled': False, 'model_answer': chosen_answers[0], **example})
                    found = True
                    break
                if correct and incorrect:
                    found = False
                    break
                elif incorrect:
                    returned_answers.append(
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': line,
                         'correct': False,
                         'answer_randomly_sampled': False, 'model_answer': chosen_answers[-1], **example})
                    found = True
                    break

            if found:
                continue

            # If randomly_select_when_unparsable is on, and we did not find an answer in the response, we will randomly
            #  assign an answer to this generation but we mark it as such.
            returned_answers.append({
                'model_response': resp,
                'answer_line': None,
                'correct': False,
                'answer_randomly_sampled': True,
                'model_answer': None,
                **example
            })
        return returned_answers


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = MusiqueDataset()

    ex = dataset[0]


    responses = [
"""Answer: $59,039	
"""
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])