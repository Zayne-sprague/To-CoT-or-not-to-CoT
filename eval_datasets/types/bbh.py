import random

from datasets import load_dataset, get_dataset_config_names

from eval_datasets import ReasoningDataset
from eval_datasets.types.csqa import CSQADataset


class BigBenchHardDataset(ReasoningDataset):
    def __init__(self, path_or_url='maveriq/bigbenchhard', split='train', *args, **kwargs):
        super().__init__(path_or_url + ':' + split, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.bbh

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split = path_or_url.split(':')
        configs = get_dataset_config_names(dataset_url)

        for config in configs:
            dataset = [x for x in load_dataset(dataset_url, config)[split]]

            for ex in dataset:
                try:
                    ex['dataset_type'] = self.dataset_types.bbh
                    ex = self.setup_ex(ex, config)
                    ex['subset'] = config
                    examples.append(ex)
                except Exception:
                    pass

        random.seed(42)
        random.shuffle(examples)
        return examples

    @classmethod
    def evaluate_response(cls, model_responses, example, *args, **kwargs):
        if len(example['choices']['text']) > 1:
            return CSQADataset.evaluate_response(model_responses, example, *args, **kwargs)
        else:
            parsed_answer = []
            for model_response in model_responses:
                if 'The best answer is' not in model_response:
                    parsed_answer.append({
                        'answer_label': None, 'model_response': model_response, 'answer_line': None,
                         'correct': False, 'answer_span': None, 'raw_model_answer': None,
                         'answer_randomly_sampled': False, 'model_answer': None, **example
                    })
                else:
                    answer = model_response.split("The best answer is")[-1].strip().split('.')[0]
                    if example['answer'].lower() in answer.lower():
                        parsed_answer.append({
                            'answer_label': None,
                            'model_response': model_response,
                            'answer_line': answer,
                            'correct': True,
                            'answer_span': [model_response.index(answer), model_response.index(answer) + len(answer)],
                            'answer_randomly_sampled': False,
                            'model_answer': answer,
                            'raw_model_answer': answer,
                            **example
                        })
                    else:
                        parsed_answer.append({
                            'answer_label': None,
                            'model_response': model_response,
                            'answer_line': answer,
                            'correct': False,
                            'answer_span': [model_response.index(answer), model_response.index(answer) + len(answer)],
                            'answer_randomly_sampled': False,
                            'model_answer': answer,
                            'raw_model_answer': answer,
                            **example
                        })

            return parsed_answer

    @classmethod
    def custom_evaluate_response(cls, model_responses, example, *args, **kwargs):
        return None


    def setup_ex(self, ex, subset):
        question = None
        choices = None

        if subset == 'boolean_expressions':
            choices = {'text': ['True', 'False'], 'label': ['A', 'B']}
            question = ex['input']
            target = ex['target']
        elif subset == 'causal_judgement' or subset == 'navigate' or subset == 'sports_understanding' or subset=='web_of_lies':
            choices = {'text': ['Yes', 'No'], 'label': ['A', 'B']}
            question = ex['input'].split("Options:")[0].strip()
            target = ex['target']
        elif subset == 'formal_fallacies':
            choices = {'text': ['valid', 'invalid'], 'label': ['A', 'B']}
            question = ex['input'].split("Options:")[0].strip()
            target = ex['target']
        elif subset in [
            'date_understanding',
            'disambiguation_qa',
            'geometric_shapes',
            'hyperbaton',
            'logical_deduction_five_objects',
            'logical_deduction_seven_objects',
            'logical_deduction_three_objects',
            'movie_recommendation',
            'penguins_in_a_table',
            'reasoning_about_colored_objects',
            'ruin_names',
            'salient_translation_error_detection',
            'snarks',
            'temporal_sequences',
            'tracking_shuffled_objects_five_objects',
            'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_three_objects',
        ]:
            raw_question = ex['input']
            question, choices = raw_question.split('Options:')
            question = question.strip()
            choice_texts = [x.split(')')[1].strip() for x in choices.split('(') if x.strip() != '']
            choices = {'text': choice_texts, 'label': [chr(65 + i) for i in range(len(choice_texts))]}
            target = ex['target'].replace('(','').replace(')','')
        elif subset == 'dyck_languages' or subset == 'multistep_arithmetic_two' or subset == 'object_counting' or subset=='word_sorting':
            question = ex['input']
            target = ex['target']
            choices = None


        else:
            raise Exception('Invalid subset')

        def get_index_and_label(target, choices):
            for idx, (text, label) in enumerate(zip(choices['text'], choices['label'])):
                if target in text or target in label:
                    return idx, label
            return None


        if question is not None and choices is not None:
            answer_index, answer_label = get_index_and_label(target, choices)
            return {
                **ex,
                'question': question,
                'answer': target,
                'answerKey': answer_label,
                'answer_index': answer_index,
                'choices': choices,
                'prompt_parts': {
                    'zs_cot_prompt': self.basic_prompt(question, self.format_choices(choices), direct=False),
                    'zs_cotless_prompt': self.basic_prompt(question, self.format_choices(choices), direct=True),
                    'cot_system_prompt': self.default_sys_mc_cot_prompt,
                    'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                },
            }
        elif question is not None:
            return {
                **ex,
                'question': question,
                'answer': target,
                'answerKey': target,
                'choices': {'text': [target], 'label': ['A']},
                'prompt_parts': {
                    'zs_cot_prompt': self.basic_prompt_short_answer(question, direct=False),
                    'zs_cotless_prompt': self.basic_prompt_short_answer(question, direct=True),
                    'cot_system_prompt': self.default_sys_short_ans_cot_prompt,
                    'cotless_system_prompt': self.default_sys_short_ans_cotless_prompt
                },
            }

    def basic_prompt_short_answer(self, question, direct=False):
        if direct:
            return f'Question: {question}\n\nWrite the answer in the following format: \"The best answer is <your answer>\" where <your answer> is a short answer to the question.'
        return f'Question: {question}\n\nAnswer the question above using this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nAlways conclude with:\nThe best answer is [your answer].\nwhere the [your answer] is a short answer response to the question\n\nLet\'s think step by step.'

    @property
    def default_sys_short_ans_cot_prompt(self):
        return 'You are a helpful AI assistant that will answer reasoning questions. You may reason over the question but you will always say at the end "The best answer is <your answer>.", where <your answer> is a short answer response to the question.  The answer should be very short. You must end your response with "The best answer is <your answer>" everytime or you will receive no credit!'

    @property
    def default_sys_short_ans_cotless_prompt(self):
        return 'You are a helpful AI assistant that will answer reasoning questions. You will always say at the end "The best answer is <your answer>.", where <your answer> is a short answer response to the question.  The answer should be very short. You must respond with "The best answer is <your answer>" everytime or you will receive no credit!'




if __name__ == "__main__":
    from src import cache
    cache.enable()

    dataset = BigBenchHardDataset()
    print('hi')



