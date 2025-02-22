import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset, CSQADataset


class WinograndeDataset(ReasoningDataset):
    def __init__(self, path_or_url='winogrande', split='validation', *args, **kwargs):
        super().__init__(path_or_url + ':' + split, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.winogrande

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url, 'winogrande_debiased', trust_remote_code=True)[split]]

        for ex in dataset:
            # choices = f'A: {ex["option1"]}\nB: {ex["option2"]}'
            choices = {"text": [ex['option1'], ex['option2']], "label": ["A", "B"]}
            answer_index = int(ex['answer']) - 1
            answer = ex['option1'] if answer_index == 0 else ex['option2']
            # prompt = f'Below is a sentence with a blank in it, fill in the blank with the most likely option. Think step by step before giving your final answer to the question. To think step-by-step, state the facts or premises you are using along with their deductions that yield the correct answer (even if those facts or premises are commonsense knowledge).  When you are ready to answer write the answer in the format: "Answer: <your answer choice (1 or 2)>".  You must always give an answer at the end.  You may only pick one answer choice.\n\nSentence:\n{ex["sentence"]}\n\nOptions:\n{choices}'
#             prompt = f'''
# {ex["sentence"]}
#
# {choices}
#
# Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer (A or B)>".  You must always give an answer at the end.  You may only pick one answer choice.
#             '''.strip()
            zs_cot_prompt = self.basic_prompt(ex["sentence"], self.format_choices(choices))
            zs_cotless_prompt = self.basic_prompt(ex["sentence"], self.format_choices(choices), direct=True)

#             fs_cot_prompt = f'''
# {self.icl()}
#
# Sentence: "{ex["sentence"]}"
#
# Answer Choices:
# {choices}
#
# Output:
#                         '''.strip()

            examples.append({
                **ex,
                'dataset_type': self.dataset_types.winogrande,
                'prompt_parts': {
                    'zs_cot_prompt': zs_cot_prompt,
                    'zs_cotless_prompt': zs_cotless_prompt,
                    'cot_system_prompt': self.default_sys_mc_cot_prompt,
                    'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                },
                'choices': choices,
                'question': ex["sentence"],
                'answerKey': 'A' if answer_index == 0 else 'B',
                'answer': answer,
                'answer_index': answer_index,
                'answer_choice_tokens': ['A', 'B'],

                })

        return examples

    def icl(self):
        # TODO - do this for all of them?
        examples = [
            {'question': 'John moved the couch from the garage to the backyard to create space. The _ is small.', 'choices': [{'label': ['A', 'B'], 'text': ['garage', 'backyard']}], 'answer': 'A', 'cot': "Since John is moving something out of the garage to make space, the garage must be what was small and that's why he was moving the couch out of it."},

        ]
        return '\n'.join([x.strip() for x in f'''Sentence: "

Answer Choices: 
A: garage
B: backyard

Output: Since John is moving something out of the garage to make space, the garage must be what was small and that's why he was moving the couch out of it.

ANSWER: A

Sentence: "The doctor diagnosed Justin with bipolar and Robert with anxiety. _ had terrible nerves recently."

Answer Choices: 
A: Justin
B: Robert

Output: Anxiety is usually associated with being nervous so the person with anxiety would have terrible nerves.

ANSWER: B

Sentence: "Dennis drew up a business proposal to present to Logan because _ wants his investment."

Answer Choices: 
A: Dennis
B: Logan 

Output: Because Dennis is drawing up hte proposal we can assume he is the one that wants the investment.

ANSWER: A

Sentence: "Felicia unexpectedly made fried eggs for breakfast in the morning for Katrina and now _ owes a favor."

Answer Choices: 
A: Felicia
B: Katrina

Output: Usually when someone does something for someone else, the person who did the favor is the one who is owed a favor. Since here, Felicia made breakfast for Katrina, Katrina is the one who owes a favor.

ANSWER: B

Sentence: "My shampoo did not lather easily on my Afro hair because the _ is too dirty."

Answer Choices: 
A: shampoo 
B: hair 

Output: Shampoo is often not considered clean or dirty, but hair can be dirty. 

Answer: B
        '''.strip().split('\n')])

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
    def custom_evaluate_response(cls, model_responses, example, *args, model=None, **kwargs):
        return None


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = WinograndeDataset(split='train')

    ex = dataset[0]


    responses = [
        'I think 1 2 the answer is 1',
        'ANSWER: 2\n\nBecause of..\n\nSo answer 1',
        'I think because...\n\nANSWER: 1',
        'I think because...\n\nANSWER: 1 or 2'
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])