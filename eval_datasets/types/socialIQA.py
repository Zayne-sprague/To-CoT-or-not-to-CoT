import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset, CSQADataset


class SocialIQADataset(ReasoningDataset):
    average_token_len = 300

    def __init__(self, path_or_url='social_i_qa', split='validation', *args, **kwargs):
        super().__init__(path_or_url + ':' + split, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.socialiqa

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url, trust_remote_code=True)[split]]

        for ex in dataset:
            # choices = f'A: {ex["answerA"]}\nB: {ex["answerB"]}\nC: {ex["answerC"]}'
            choices = {'text': [ex['answerA'], ex['answerB'], ex['answerC']], 'label': ['A', 'B', 'C']}
            answer_index = int(ex['label']) - 1
            answer = [ex['answerA'], ex['answerB'], ex['answerC']][answer_index]
            # prompt = f'Below you are given some context and question, you must choose the best answer to the question given the current context. Think step by step before giving your final answer to the question. To think step-by-step, state the facts or premises you are using along with their deductions that yield the correct answer (even if those facts or premises are commonsense knowledge).  When you are ready to answer write the answer in the format: "Answer: <your answer choice (1, 2, or 3)>".  You must always give an answer at the end.  You may only pick one answer choice.\n\nContext:\n{ex["context"]}\n\nQuestion:\n{ex["context"]}\n\nAnswers:\n{choices}\n\nYou must pick one answer. Let\'s think step by step.'
#             prompt = f'''
# {ex["context"]}
#
# {choices}
#
#  Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer (A, B, or C)>".  You must always give an answer at the end.  You may only pick one answer choice.
#                         '''.strip()
            zs_cot_prompt = self.basic_prompt(ex["context"], self.format_choices(choices))
            zs_cotless_prompt = self.basic_prompt(ex["context"], self.format_choices(choices), direct=True)

            examples.append({
                **ex,
                'dataset_type': self.dataset_types.socialiqa,
                'prompt_parts': {
                    'zs_cot_prompt': zs_cot_prompt,
                    'zs_cotless_prompt': zs_cotless_prompt,
                    'cot_system_prompt': self.default_sys_mc_cot_prompt,
                    'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                },
                'choices': choices,
                'answer': answer,
                'answer_index': answer_index,
                'answer_choice_tokens': ['A', 'B', 'C'],
                'answerKey': ['A', 'B', 'C'][answer_index],
                'og_question': ex['question'],
                'question': ex['context'] + '\n\n' + ex['question'],
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
    def custom_evaluate_response(cls, model_responses, example, *args, **kwargs):
        return None
    
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = SocialIQADataset()

    ex = dataset[0]


    responses = [
        'I think 1 2 the answer is 1',
        'ANSWER: 2\n\nBecause of..\n\nSo answer 1',
        'I think because...\n\nANSWER: 3',
        'I think because...\n\nANSWER: 1 or 2'
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])