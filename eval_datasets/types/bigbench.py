import random

from datasets import load_dataset, get_dataset_config_names

from eval_datasets import ReasoningDataset


class BigBenchDataset(ReasoningDataset):
    def __init__(self, path_or_url='tasksource/bigbench', subsets='all', split='validation', *args, **kwargs):
        self.total_skipped_examples = 0

        super().__init__(path_or_url + ':' + subsets + ':' + split, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.bigbench

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, subsets, split = path_or_url.split(':')
        configs = subsets.split(',')
        if 'all' in configs:
            configs = get_dataset_config_names(dataset_url)

        for c in configs:
            try:
                dataset = [x for x in load_dataset(dataset_url, c)[split]]
            except Exception as e:
                continue


            for ex in dataset:
                targets = ex['targets']
                if len(targets) > 1:
                    self.total_skipped_examples += 1
                    continue
                if len(ex['multiple_choice_targets']) == 0:
                    self.total_skipped_examples += 1
                    continue

                choice_formats = {
                    'text': [],
                    'label': []
                }
                for idx in range(len(ex['multiple_choice_targets'])):
                    choice_formats['text'].append(ex['multiple_choice_targets'][idx])
                    choice_formats['label'].append(chr(65 + idx))

                choices = '\n'.join([f'{choice_formats["label"][i]}: {choice_formats["text"][i]}' for i in range(len(choice_formats["label"]))])

                try:
                    answer_index = choice_formats['text'].index(ex['targets'][0])
                except Exception:
                    self.total_skipped_examples += 1
                    continue

                answerKey = choice_formats['label'][answer_index]
                answer = choice_formats['text'][answer_index]
                prompt = f'{ex["inputs"]}\n{choices}\nThink step by step before giving your final answer to the question. To think step-by-step, state the facts or premises you are using along with their deductions that yield the correct answer (even if those facts or premises are commonsense knowledge).  When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.  You may only pick one answer choice.'

                answer_choice_tokens = [f'{choice_formats["label"][i]}' for i in range(len(choice_formats["label"]))]

                examples.append({
                    'dataset_type': self.dataset_types.csqa,
                    'prompt_parts': {'user_context': prompt},
                    'choices': choice_formats,
                    'answer': answer,
                    'answer_index': answer_index,
                    'answerKey': answerKey,
                    'task_subset': c,
                    'answer_choice_tokens': answer_choice_tokens,
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
        answers = [example['answer']]
        answer_labels = [f"{example['answerKey']}"]
        incorrect_answers = [*[x for x in example['choices']['text'] if x not in answers]]
        incorrect_answer_labels = [*[f'{x}' for x in example['choices']['label'] if x not in answer_labels]]

        returned_answers = []

        for resp in model_responses:
            parsed_resp = resp.strip()
            found = False

            # We are going to choose the answer to be the first time we an answer choice or index mentioned...
            for line in reversed(parsed_resp.split('\n')):

                # so long as that answer is mentioned on a line with "answer:" in it.
                if 'answer:' not in line.lower():
                    continue

                correct = any([x.lower() in line.lower() for x in answers])

                correct = correct or any(
                    [any([f'{x.lower()}' == y.strip().replace(':', '').replace('.', '').lower() for y in
                          line.lower().split(' ')]) for x
                     in answer_labels])

                incorrect = any(
                    [x.lower() in line.lower() for x
                     in incorrect_answers])

                incorrect = incorrect or any(
                    [any([x.lower() == y.lower().strip().replace(':', '').replace('.', '').strip() for y in
                          line.lower().split(' ')]) for x in incorrect_answer_labels])


                chosen_answers = []

                for xidx, x in enumerate(answers):
                    if x.lower() in line.lower():
                        chosen_answers.append(answer_labels[xidx])
                for x in answer_labels:
                    for y in line.lower().split(' '):
                        if f'{x.lower()}' == y.strip().replace(':', '').replace('.', '').lower():
                            chosen_answers.append(x)

                for xidx, x in enumerate(incorrect_answers):
                    if x.lower() in line.lower():
                        chosen_answers.append(incorrect_answer_labels[xidx])

                for x in incorrect_answer_labels:
                    for y in line.lower().split(' '):
                        if x.lower() == y.lower().strip().replace(':', '').replace('.', '').strip():
                            chosen_answers.append(x)

                if correct and not incorrect:
                    returned_answers.append(
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': line, 'correct': True,
                         'answer_randomly_sampled': False, 'model_answer': chosen_answers[0], **example})
                    found = True

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
                'correct': True if random.random() <= 1 / len(
                    example['choices']) and randomly_select_when_unparsable else False,
                'answer_randomly_sampled': True,
                'model_answer': None,
                **example
            })
        return returned_answers


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = BigBenchDataset()

    ex = dataset[0]


    responses = [
        'I think A B C the answer is B',
        'ANSWER: A\n\nBecause of..\n\nSo answer B',
        'I think because...\n\nANSWER: A',
        'I think because...\n\nANSWER: A or B'
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])

    print(ex['messages'][0]['content'])
    print(f'TOTAL EXAMPLES: ', len(dataset))
    print(f'TOTAL SKIPPED EXAMPLES: {dataset.total_skipped_examples}')