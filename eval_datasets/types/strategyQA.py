import random

import json

from eval_datasets import ReasoningDataset, CSQADataset


class StrategyQADataset(ReasoningDataset):
    # 600 usually
    # 1500 for icl?
    average_token_len = 1500

    def __init__(self, path_or_url, *args, **kwargs):
        super().__init__(path_or_url, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.strategyqa

    def load_dataset(self, path_or_url):
        data = json.load(open(path_or_url, 'r'))
        examples = []

        for raw_ex in data:
            ex = {}
            ex['dataset_type'] = self.dataset_types.strategyqa
            ex['answer_choice_tokens'] = ['1', '2']
            ex['choices'] = {"label": ["A","B"], "text": ['True', 'False']}
            ex['answerKey'] = 'A' if raw_ex['answer'] is True else 'B'
            ex['answer_index'] = 0 if raw_ex['answer'] is True else 1
            ex['answer'] = 'True' if raw_ex['answer'] is True else 'False'
            ex['answer_choice_tokens'] = ['1', '2']
            ex['question'] = raw_ex['question']


            choices = self.format_choices(ex['choices'])
            zs_cot_prompt = f'Question: {raw_ex["question"]}\n\nAnswer Choices:\n{choices}\n\nThink step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'
            zs_cotless_prompt = f'Question: {raw_ex["question"]}\n\nAnswer Choices:\n{choices}\n\nOnly write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'
            fs_cot_prompt = f'{prompt_cot_examples()}\n\nQuestion: {raw_ex["question"]}\n\nAnswer Choices:\n{choices}\n\nPlease use the examples above as a guide when solving this problem. Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'
            fs_cotless_prompt = f'{prompt_cotless_examples()}\n\nQuestion: {raw_ex["question"]}\n\nAnswer Choices:\n{choices}\n\nPlease use the examples above as a guide when solving this problem. Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'



            examples.append({
                'prompt_parts': {
                    'zs_cot_prompt': zs_cot_prompt,
                    'zs_cotless_prompt': zs_cotless_prompt,
                    'fs_cot_prompt': fs_cot_prompt,
                    'fs_cotless_prompt': fs_cotless_prompt,
                    'cot_system_prompt': self.default_sys_mc_cot_prompt,
                    'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                },
                **raw_ex,
                **ex,
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
    def custom_evaluate_response(cls, model_responses, example, *args, model=None, **kwargs):
        return CSQADataset.custom_evaluate_response(model_responses, example, *args, model=model, **kwargs)


def prompt_cot_examples():
    return '''
Question: Do hamsters provide food for any animals?

Answer Choices:
( A ) Yes
( B ) No

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.

Answer: A

Question: Could Brooke Shields succeed at University of Pennsylvania?

Answer Choices:
( A ) Yes
( B ) No

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. 

Answer: A

Question: Hydrogen’s atomic number squared exceeds number of Spice Girls?

Answer Choices:
( A ) Yes
( B ) No

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5.

Answer: B

Question: Is it common to see frost during some college commencements?

Answer Choices:
( A ) Yes
( B ) No

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.

Answer: A

Question: Could a llama birth twice during War in Vietnam (1945-46)?

Answer Choices:
( A ) Yes
( B ) No

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. 

Answer: B

Question: Would a pear sink in water?

Answer Choices:
( A ) Yes
( B ) No

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float.

Answer: B'''.strip()


def prompt_cotless_examples():
    return '''
Question: Do hamsters provide food for any animals?

Answer Choices:
( A ) Yes
( B ) No

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: A

Question: Could Brooke Shields succeed at University of Pennsylvania?

Answer Choices:
( A ) Yes
( B ) No

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: A

Question: Hydrogen’s atomic number squared exceeds number of Spice Girls?

Answer Choices:
( A ) Yes
( B ) No

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: B

Question: Is it common to see frost during some college commencements?

Answer Choices:
( A ) Yes
( B ) No

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: A

Question: Could a llama birth twice during War in Vietnam (1945-46)?

Answer Choices:
( A ) Yes
( B ) No

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: B

Question: Would a pear sink in water?

Answer Choices:
( A ) Yes
( B ) No

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: B'''.strip()

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = StrategyQADataset('../../../reasoning_error_experiments/strategyqa_dataset/strategyqa_train.json')

    ex = dataset[0]


    responses = [
        'I think the answer is TRUE, 1',
        'ANSWER: 1\n\nBecause of..\n\nSo answer TRUE',
        'I think because...\n\nANSWER: True',
        'I think because...\n\nANSWER: 1 or 2'
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])

