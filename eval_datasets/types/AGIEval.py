import random


from eval_datasets import ReasoningDataset, CSQADataset
from src.utils.paths import ROOT_FOLDER

import os

LOADED = False
err = None
try:
    from eval_datasets.thirdparty.AGIEval.src import dataset_loader
    from eval_datasets.thirdparty.AGIEval.src.utils import read_jsonl, save_jsonl, extract_answer
    from eval_datasets.thirdparty.AGIEval.src.constructions import ChatGPTSchema, \
        ResultsForHumanSchema

    LOADED = True
except Exception as e:
    err = e
    pass


class AGIEvalDataset(ReasoningDataset):

    # 500 for olmo
    # 2k for anything larger than 4k windows, 750 for anything 4k
    average_token_len: int = 1250

    def __init__(self, path_or_url='', prompt_setting='zero-shot-CoT', slice: str = 'all', *args, **kwargs):
        assert LOADED, f'Please install AGIEval in the eval_datasets/thirdparty folder.  {e}'
        assert slice in ["lsat-ar", "lsat-lr", "lsat-rc", "logiqa-en", "sat-math", "sat-en", "aqua-rat", "sat-en-without-passage", "gaokao-english", "all"], 'incorrect slice.'
        self.total_skipped_examples = 0
        self.prompt_setting = prompt_setting
        self.slice = slice

        super().__init__(path_or_url, *args, **kwargs)


    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.agieval

    def load_dataset(self, path_or_url):
        examples = []

        datasets = dataset_loader.english_qa_datasets
        for dataset in datasets:
            if self.slice != 'all' and dataset != self.slice:
                continue
            raw_examples = self.agi_load_dataset(dataset, self.prompt_setting, str(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/v1'),  max_tokens=2048, prompt_path=(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/few_shot_prompts.csv'))
            cotless_raw_examples = self.agi_load_dataset(dataset, "zero-shot", str(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/v1'),  max_tokens=2048, prompt_path=(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/few_shot_prompts.csv'))
            few_shot_cot_examples = self.agi_load_dataset(dataset, "few-shot-CoT", str(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/v1'),  max_tokens=2048, prompt_path=(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/few_shot_prompts.csv'))
            few_shot_cotless_examples = self.agi_load_dataset(dataset, "few-shot", str(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/v1'),  max_tokens=2048, prompt_path=(ROOT_FOLDER / 'eval_datasets/thirdparty/AGIEval/data/few_shot_prompts.csv'))

            raw_examples = list(sorted(raw_examples, key=lambda x: x['metadata']))
            cotless_raw_examples = list(sorted(cotless_raw_examples, key=lambda x: x['metadata']))

            for (ex, cotless_ex, fs_ex, fs_cotless_ex) in zip(raw_examples, cotless_raw_examples, few_shot_cot_examples, few_shot_cotless_examples):
                # check if any metadata doesn't match
                if any([ex['metadata'] != x['metadata'] for x in [cotless_ex, fs_ex, fs_cotless_ex]]):
                    continue

                zs_cot_prompt = ex['context']
                zs_cotless_prompt = cotless_ex['context']
                fs_cot_prompt = fs_ex['context']
                fs_cotless_prompt = fs_cotless_ex['context']

                answerKey = ex['label']
                choices = ex['choices']

                try:
                    answer_index = choices['label'].index(answerKey)
                except Exception:
                    self.total_skipped_examples+= 1
                    continue

                def parse_fs_into_msgs(fs):
                    questions = fs.split('\n\n')
                    questions = [questions[0]+'\n\n'+questions[1], *questions[2:]]
                    if 'Explanation for' in fs:
                        question_answers = [
                            [x.split('Explanation for')[0], 'Explanation for' + x.split('Explanation for')[1]] for x in questions[:-1]
                        ]
                    else:
                        question_answers = [
                            [x.split('The answer is therefore')[0], 'The answer is therefore' + x.split('The answer is therefore')[1]] for x in questions[:-1]
                        ]

                    messages = []
                    for q, a in question_answers:
                        messages.append({
                            'role': 'user',
                            'content': q
                        })
                        messages.append({
                            'role': 'assistant',
                            'content': a
                        })
                    messages.append({
                        'role': 'user',
                        'content': questions[-1]
                    })
                    return messages

                examples.append({
                    **ex,
                    'dataset_type': self.dataset_types.agieval,
                    'question': ex['context'].split(' Failure to comply with the answer formatting will result in no credit.')[-1].split(' Answer Choices:')[0].strip(),
                    'prompt_parts': {
                        'zs_cot_prompt': zs_cot_prompt,
                        'zs_cotless_prompt': zs_cotless_prompt,
                        'fs_cot_prompt': fs_cot_prompt,
                        'fs_cotless_prompt': fs_cotless_prompt,
                        'cot_system_prompt': self.default_sys_mc_cot_prompt,
                        'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                    },
                    'answer': choices['text'][answer_index],
                    'answer_index': answer_index,
                    'dataset_filename': dataset,
                    'choices': choices,
                    'answerKey': answerKey,
                    'answer_choice_tokens': [f'{x}' for x in choices['label']],
                    "llama_3_1_eval":  {
                        "prompts": {
                            "fs_cot": parse_fs_into_msgs(fs_cot_prompt),
                            "fs_direct": parse_fs_into_msgs(fs_cotless_prompt),
                            "zs_cot": [{'role': 'user', 'content': zs_cot_prompt}],
                            "zs_direct": [{'role': 'user', 'content': zs_cotless_prompt}]
                        }
                    }
                })
        return examples

    def agi_load_dataset(self, dataset_name, setting_name, parent_path, prompt_path=None, max_tokens=None, end_of_example="\n",
                 chat_mode=False, verbose=False):
        test_path = os.path.join(parent_path, dataset_name + ".jsonl")
        loaded_jsonl = read_jsonl(test_path)
        processed = []
        if setting_name == "few-shot-CoT" or setting_name == "few-shot":
            # process demo once if it is few-shot-CoT
            processed_demos = dataset_loader.combine_prompt(prompt_path, dataset_name, load_explanation=setting_name == 'few-shot-CoT',
                                             chat_mode=chat_mode)
            if chat_mode:
                chosen_prompt, n_shot = dataset_loader.concat_prompt_chat_mode(
                    processed_demos, dataset_name, max_tokens, end_of_example, verbose=verbose)
            else:
                chosen_prompt, n_shot = dataset_loader.concat_prompt(
                    processed_demos, dataset_name, max_tokens, end_of_example, verbose=verbose)
        for meta_idx, line in enumerate(loaded_jsonl):
            if setting_name == "zero-shot":
                ctxt = dataset_loader.convert_zero_shot(line, dataset_name)
            elif setting_name == "zero-shot-CoT":
                ctxt = dataset_loader.convert_zero_shot_CoT_stage1(line, dataset_name)
            elif setting_name == "few-shot-CoT" or setting_name == "few-shot":
                ctxt = dataset_loader.convert_few_shot(line, dataset_name, chosen_prompt, n_shot, chat_mode)
            try:
                new_instance = ChatGPTSchema(context=ctxt, metadata=meta_idx)
                raw_ex = new_instance.to_dict()
                raw_ex['label'] = line['label']

                raw_ex['choices'] = {
                    'text': [],
                    'label': []
                }

                for opt in line['options']:

                    if opt.startswith('('):
                        raw_ex['choices']['text'].append(')'.join(opt.split(')')[1:]))
                        raw_ex['choices']['label'].append(opt.split(')')[0].replace('(', '')[0])
                    else:
                        raw_ex['choices']['text'].append(' '.join(opt.split(' ')[1:]))
                        raw_ex['choices']['label'].append(opt[0])

                if any([x not in ['A', 'B', 'C', 'D', 'E'] for x in raw_ex['choices']['label']]):
                    # TODO - this must be why they have phase 2.  The questions are poorly formatted.
                    self.total_skipped_examples+= 1
                    continue

                choice_str = ', '.join([f'{raw_ex["choices"]["label"][i]}' for i in range(len(raw_ex["choices"]["label"]))])

                if ctxt.startswith('Here are the answers for the problems in the exam.'):
                    ctxt = ctxt.replace('Here are the answers for the problems in the exam.', '')
                    ctxt = f'Here are the answers for the problems in the exam. Explain your reasoning step-by-step for each question before answering.  Give your final answer in the format "The answer is therefore <{choice_str}>".  Failure to comply with the answer formatting will result in no credit.\n' + ctxt
                else:
                    ctxt = f'Explain your reasoning step-by-step for each question before answering.  Give your final answer in the format "The answer is therefore <{choice_str}>".  Failure to comply with the answer formatting will result in no credit.\n' + ctxt

                if 'cot' not in setting_name.lower():
                    ctxt = ctxt.replace('Explain your reasoning step-by-step for each question before answering.  Give your final answer in the format ', 'Give your answer in the format ').lstrip()
                raw_ex['context'] = ctxt


                processed.append(raw_ex)
            except NameError:
                print("Dataset not defined.")
        return processed


    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            *args, **kwargs
    ):
        return CSQADataset.evaluate_response(model_responses, example, randomly_select_when_unparsable=randomly_select_when_unparsable, *args, **kwargs)

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
            if 'boxed{' in line:
                potential_answer_segments.append(line.split('boxed{')[1][0].strip())

            curr_tok_idx = sum([len(x) + 1 for x in parsed_resp.split('\n')[:-1]])
            potential_answer_segments = [[x, curr_tok_idx] for x in potential_answer_segments]

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

    ds = AGIEvalDataset(prompt_setting='zero-shot-CoT', slice='lsat-ar')

    ex = ds[0]

    responses = [
        '''Dostoyevsky believed that art should be well-written and not serve a particular political view. He believed that art should be judged on its artistic merit, not its political usefulness. Therefore, the answer is (D) A work of literature that is well-written cannot serve any particular political view. 

Here's my reasoning:

1. Dostoyevsky believed that art should be well-written. (Paragraph 1)
2. Dostoyevsky believed that art should not serve a particular political view. (Paragraph 2)
3. Therefore, a work of literature that is well-written cannot serve any particular political view. (Conclusion)

Answer: (D) A work of literature that is well-written cannot serve any particular political view.

'''
    ]

    metrics = ds.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])

    print(f'SKIPPED EXAMPLES: {ds.total_skipped_examples}')