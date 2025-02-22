# import sys
# sys.path.insert(0, '/u/fcyin/TrainingReasoning')
import random
from copy import deepcopy

from pathlib import Path
import json
from src.utils.paths import ROOT_FOLDER

from eval_datasets import ReasoningDataset, CSQADataset

random.seed(0)


class ProntoQADataset(ReasoningDataset):
    average_token_len = 1000

    def __init__(self, path_or_url=ROOT_FOLDER / 'eval_datasets/thirdparty/prontoqa', *args, level: int = 1, ontology: str = 'true', **kwargs):
        self.ontology = ontology
        if isinstance(path_or_url, str):
            path_or_url = Path(path_or_url)

        if not str(path_or_url).endswith('.json'):
            if ontology == 'fictional' or ontology == 'symbolic':
                path_or_url = path_or_url / ontology / (f"{str(level)}hop.json")
            elif ontology == 'paraphrase':
                path_or_url = path_or_url / ontology / (f"{str(level)}hop_trueontology.json")
            else:
                path_or_url = path_or_url / ontology / (f"{str(level)}hop_{ontology}ontology.json")

        super().__init__(path_or_url, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.prontoqa

    def load_dataset(self, path_or_url):
        data = json.load(open(str(path_or_url), 'r'))
        examples = []

        for raw_ex in data:
            if self.ontology == 'paraphrase':
                root_ex = {**raw_ex}
                q_str = root_ex['paraphrase_question']
                choices = root_ex['choices']
                prompt = self.basic_prompt(q_str, self.format_choices(choices))

                root_ex['prompt_parts'] = {'user_context': prompt}
                root_ex["question_format"] = q_str
                examples.append(root_ex)
                continue
            raw_ex = data[raw_ex]
            root_ex = {}
            root_ex['in_context_examples'] = []
            for key in raw_ex:
                if 'in_context_example' in key:
                    root_ex['in_context_examples'].append(raw_ex[key])
            raw_ex = raw_ex['test_example']

            answer = raw_ex['answer']
            choices = {"label": ["A","B"], "text": ['True', 'False']}

            root_ex['choices'] = choices
            root_ex['answer'] = answer
            root_ex['answer_index'] = 0 if raw_ex['answer'] == 'True' else 1
            root_ex['answer_choice_tokens'] = ['A', 'B']
            root_ex['answerKey'] = 'A' if raw_ex['answer'] == 'True' else 'B'
            root_ex["dataset_type"] = self.dataset_types.prontoqa
            root_ex['question_format'] = raw_ex["question"] + " " + raw_ex['query']
            root_ex['gold_cot'] = raw_ex['chain_of_thought']

            

            prompt = self.basic_prompt(root_ex['question_format'], self.format_choices(choices))

            root_ex['prompt_parts'] = {'user_context': prompt}

            examples.append({
                **root_ex,
            })

        random.shuffle(examples)
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
    def custom_evaluate_response(self, model_responses, example, *args, **kwargs):
        return None


if __name__ == '__main__':
    # from torch.utils.data import DataLoader

    dataset = ProntoQADataset()

    ex = dataset[0]


    responses = [
        'I think the answer is TRUE, A',
        'ANSWER: A\n\nBecause of..\n\nSo answer TRUE',
        'I think because...\n\nANSWER: True',
        'I think because...\n\nANSWER: A or B'
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])
    print(metrics)
    print(ex)
