import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset, CSQADataset


class PIQADataset(ReasoningDataset):
    average_token_len = 600

    def __init__(self, path_or_url='piqa', split='validation', *args, **kwargs):
        super().__init__(path_or_url + ':' + split, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.piqa

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url, trust_remote_code=True)[split]]

        for ex in dataset:
            choices = {'text': [ex['sol1'], ex['sol2']], 'label': ['A','B']}
            answer_index = ex['label']
            answer = ex['sol1'] if answer_index == 0 else ex['sol2']
            #prompt = f'Below is a sentence describing a goal and two possible solutions to achieve that goal.  Pick the solution that is most likely to achieve the goal. Think step by step before giving your final answer to the question. To think step-by-step, state the facts or premises you are using along with their deductions that yield the correct answer (even if those facts or premises are commonsense knowledge).  When you are ready to answer write the answer in the format: "Answer: <your answer choice (1 or 2)>".  You must always give an answer at the end.  You may only pick one answer choice.\n\nGoal:\n{ex["goal"]}\n\nPotential Solutions:\n{choices}\n\nYou must pick one option. Let\'s think step by step.'
            #             prompt = f'''
            # {ex["goal"]}
            #
            # {choices}
            #
            #  Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer (A or B)>".  You must always give an answer at the end.  You may only pick one answer choice.
            #             '''.strip()
            zs_cot_prompt = self.basic_prompt(ex["goal"], self.format_choices(choices))
            zs_cotless_prompt = self.basic_prompt(ex["goal"], self.format_choices(choices), direct=True)


            examples.append({
                'dataset_type': self.dataset_types.piqa,
                'prompt_parts': {
                    'zs_cot_prompt': zs_cot_prompt,
                    'zs_cotless_prompt': zs_cotless_prompt,
                    'cot_system_prompt': self.default_sys_mc_cot_prompt,
                    'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                },
                'choices': choices,
                'answer': answer,
                'question': ex["goal"],
            'answer_index': answer_index,
            'answer_choice_tokens': ['A', 'B'],
            'answerKey': 'A' if answer_index == 0 else 'B',

            **ex
            })

        return examples

    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            *args, **kwargs
    ):
        return CSQADataset.evaluate_response(model_responses, example, randomly_select_when_unparsable, *args, **kwargs)

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
            if '**' in line:
                potential_answer_segments.append(line.split('**')[1].strip())
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
            if 'answer:' in line.lower():
                potential_answer_segments.append(line)

            curr_tok_idx = sum([len(x) + 1 for x in parsed_resp.split('\n')[:-1]])
            potential_answer_segments = [[x, curr_tok_idx] for x in potential_answer_segments]

            curr_tok_idx = 0
            lines = []
            for l in resp.split('\n'):
                if l.strip() == '':
                    curr_tok_idx += len(l) + 1
                    continue
                else:
                    lines.append([l, curr_tok_idx])
                    curr_tok_idx += len(l) + 1

            for lidx, (l, tok_idx) in enumerate(lines):

                for s in l.split('.'):
                    seqs = ["i would recommend option", "I would choose option"]
                    if any([x.lower() in s.lower() for x in seqs]):
                        potential_answer_segments.append([s, tok_idx])


            for (line, tok_idx) in potential_answer_segments:
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
                        chosen_answers.append(answers[xidx])
                for x in answer_labels:
                    for y in line.split(' '):
                        if f'{x}' == cls.parse_ans(y):
                            chosen_answers.append(x)

                for xidx, x in enumerate(incorrect_answers):
                    if x.lower() in line.lower():
                        chosen_answers.append(incorrect_answers[xidx])

                for x in incorrect_answer_labels:
                    for y in line.split(' '):
                        if f'{x}' == cls.parse_ans(y):
                            chosen_answers.append(x)

                if correct and not incorrect:
                    returned_answers.append(
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': line,
                         'correct': True, 'answer_span': [tok_idx + cls.safe_find(parsed_resp[tok_idx:], chosen_answers[0]), tok_idx + cls.safe_find(parsed_resp[tok_idx:], chosen_answers[0]) + len(chosen_answers[0])],
                         'answer_randomly_sampled': False, 'model_answer': chosen_answers[0], **example})
                    found = True
                    break
                if correct and incorrect:
                    found = False
                    break
                elif incorrect:
                    returned_answers.append(
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': line,
                         'correct': False, 'answer_span': [tok_idx + cls.safe_find(parsed_resp[tok_idx:], chosen_answers[-1]), tok_idx + cls.safe_find(parsed_resp[tok_idx:], chosen_answers[-1]) + len(chosen_answers[-1])],
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
                'answer_span': None,
                **example
            })
        return returned_answers


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = PIQADataset()

    ex = dataset[1]


    responses = [
        'I think 1 2 the answer is 1',
        'ANSWER: 2\n\nBecause of..\n\nSo answer 1',
        'I think because...\n\nANSWER: 1',
        'I think because...\n\nANSWER: 1 or 2'
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])