import random

from datasets import load_dataset
import yaml
import json

from eval_datasets import ReasoningDataset, CSQADataset
from src.utils.paths import ROOT_FOLDER
import re, sys, unicodedata


class MMLUDataset(ReasoningDataset):
    # 1000 for olmo
    average_token_len = 1500

    def __init__(self, path_or_url='hails/mmlu_no_train', split='test', subset="all", raw_llama31_prompts: bool = False, use_llama_3_1_prompts: bool = True, reload_llama_3_1_prompts: bool = False, llama_3_1_prompts_cache_file: str = ROOT_FOLDER / 'eval_datasets/third_party/llama_3_1_prompts_tmp_folder/mmlu.json', use_lm_eval_prompts: bool = False, reload_lm_eval_prompts: bool = False, lm_eval_prompts_cache_file: str = ROOT_FOLDER / 'eval_datasets/lm_eval_prompts_tmp_folder/mmlu.json', *args, **kwargs):
        self.use_lm_eval_prompts = use_lm_eval_prompts
        self.reload_lm_eval_prompts = reload_lm_eval_prompts
        lm_eval_prompts_cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.lm_eval_prompts_cache_file = lm_eval_prompts_cache_file

        self.llama_3_1_prompts_cache_file = llama_3_1_prompts_cache_file
        self.raw_llama31_prompts = raw_llama31_prompts
        self.use_llama_3_1_prompts = use_llama_3_1_prompts
        self.reload_llama_3_1_prompts = reload_llama_3_1_prompts
        llama_3_1_prompts_cache_file.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path_or_url + ':' + split + ":" + subset, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.mmlu

    def load_llama_3_1_prompts(self, examples):
        if self.llama_3_1_prompts_cache_file.is_file() and not self.reload_llama_3_1_prompts:
            llama_prompts_per_task = json.load(open(self.llama_3_1_prompts_cache_file, 'r'))
        else:
            llama_prompts_per_task = {}
            raw_prompt_dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
                                              "Meta-Llama-3.1-8B-Instruct-evals__mmlu__details")["latest"]
            for x in raw_prompt_dataset:
                subject = x['subtask_name'].replace('mmlu_chat.', '')
                llama_prompts_per_task.setdefault(subject, []).append(
                    {'prompt': x['input_final_prompts'], 'question': x['input_question'],
                     'eval_config': x['eval_config'], 'was_correct': x['is_correct']})

        for exidx, ex in enumerate(examples):
            question = ex['question']
            subject = ex['subject']

            if self.use_llama_3_1_prompts:
                for prompt in llama_prompts_per_task[subject]:
                    if prompt['question'] == question:
                        llama_3_1_prompt = prompt['prompt'][0]

                        llama_3_1_eval_config = prompt['eval_config']

                        few_shot_direct_prompt = llama_3_1_prompt
                        zero_shot_direct_prompt = "<|start_header_id|>user<|end_header_id|>" + llama_3_1_prompt.split("<|start_header_id|>user<|end_header_id|>")[-1].strip()

                        few_shot_cot_prompt = self.build_fs_cot_prompt_from_llama_3_1_eval(ex, llama_3_1_prompt)
                        zero_shot_cot_prompt = self.build_zs_cot_prompt_from_llama_3_1_eval(ex, llama_3_1_prompt)

                        if not self.raw_llama31_prompts:
                            few_shot_direct_prompt = self.convert_raw_to_messages(few_shot_direct_prompt)
                            zero_shot_direct_prompt = self.convert_raw_to_messages(zero_shot_direct_prompt)
                            few_shot_cot_prompt = self.convert_raw_to_messages(few_shot_cot_prompt)
                            zero_shot_cot_prompt = self.convert_raw_to_messages(zero_shot_cot_prompt)


                        examples[exidx]['llama_3_1_eval'] = {'prompts': {
                            'fs_cot': few_shot_cot_prompt,
                            'fs_direct': few_shot_direct_prompt,
                            'zs_cot': zero_shot_cot_prompt,
                            'zs_direct': zero_shot_direct_prompt,
                        }, 'eval_config': llama_3_1_eval_config, 'few_shot_direct_was_correct': prompt['was_correct']}
        return examples

    def convert_raw_to_messages(self, raw):
        messages = []
        user_messages = [x for x in raw.split("<|start_header_id|>user<|end_header_id|>\n\n") if x!='']
        for user_message in user_messages:
            msgs = user_message.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')

            if len(msgs) == 2:

                messages.append({'role': 'user', 'content': msgs[0].replace("<|eot_id|>", "").replace("<|start_header_id|>user<|end_header_id|>", "")})
                if msgs[1] != '':
                    messages.append({'role': 'assistant', 'content': msgs[1].replace("<|eot_id|>", "")})
            else:
                messages.append({'role': 'user', 'content': user_message.replace("<|eot_id|>", "").replace("<|start_header_id|>user<|end_header_id|>", "")})
        return messages


    def build_fs_cot_prompt_from_llama_3_1_eval(self, example, llama_3_1_prompt):
        instructions = """\n\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nGive a description of your reasoning before you answer.\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

        gen_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        shots = [x[2:] if xidx > 0 else x for xidx, x in enumerate(llama_3_1_prompt.split('\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is '))]
        few_shot_responses = example["few_shot_question_response_pairs"]["cot"]
        for sidx, shot in enumerate(shots[:-1]):
            few_shot_response = [x[1] for x in few_shot_responses if x[0].strip().lower() in shot.strip().lower()]
            if len(few_shot_response) > 1:
                few_shot_response = [few_shot_response[0]]

            if len(few_shot_response) != 1:
                # Llama 3 has fixed MMLU ICL examples so they do not always directl

                counts = []
                max_ct = 0
                for fs in few_shot_responses:

                    words = fs[0].split()
                    ct = 0
                    for word in words:
                        if word in shot:
                            ct += 1

                    ct = ct/len(words)
                    counts.append([fs, ct])
                    if ct > max_ct:
                        max_ct = ct

                few_shot_response = [x[0][1] for x in counts if x[1] == max_ct]

            assert len(few_shot_response) == 1, f"Few shot response not found for shot: {shot}"
            few_shot_response = few_shot_response[0]
            # few_shot_response = ''.join([(f'## Step {xidx+1}:\n' + x + '.\n' if 'The best answer is' not in x else '\n' + x) for xidx, x in enumerate(few_shot_response.replace('Output: ', '').replace("Let's think step by step. ", '').replace('Answer:', 'The best answer is').split('. '))])
            few_shot_response = few_shot_response.replace("Output: ", "").replace("Let's think step by step. ", '').replace("Answer:", "The best answer is") + '.'


            shots[sidx] = shots[sidx] + instructions + few_shot_response.replace('Output: ', '')

        shots[-1] = shots[-1].replace("""Your response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is""", instructions)
        return "".join(shots)

    def build_zs_cot_prompt_from_llama_3_1_eval(self, example, llama_3_1_prompt):
        instructions = """\n\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

        question = '<|start_header_id|>user<|end_header_id|>' + llama_3_1_prompt.split('<|start_header_id|>user<|end_header_id|>')[-1].split("\nYour response should end with")[0] + instructions
        return question

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split, subset = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url, subset, trust_remote_code=True)[split]]

        few_shot_configs_folder = ROOT_FOLDER / 'eval_datasets/thirdparty/lm_eval_mmlu_few_shot'
        loaded_configs = {}
        few_shot_prompts = {}

        choice_labels = ['A', 'B', 'C', 'D']
        for ex in dataset:
            subject = ex['subject']
            if subject not in loaded_configs:
                loaded_configs[subject] = yaml.safe_load(open(few_shot_configs_folder / f'mmlu_{subject}.yaml', 'r'))
                few_shot_prompts[subject] = self.build_fewshot_context(loaded_configs[subject])
                fs_messages = few_shot_prompts[subject]
            else:
                fs_messages = few_shot_prompts[subject]

            choices = {"label": choice_labels, "text": ex['choices']}
            choice_str = self.format_choices(choices)
            answer_index = ex['answer']
            question = ex['question']

            answer = f"{ex['choices'][answer_index]}"

            lm_eval_messages = []

            cot_prompt = f'''
        
{ex["question"]}

{choice_str}

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.  You may only pick one answer choice. 
                        '''.strip()
            direct_prompt = f'''
{ex["question"]}

{choice_str}

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.
            '''.strip()

            fs_cot_prompt = f'''{fs_messages['cot']}Question: {ex["question"]}

{choice_str}

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.  You may only pick one answer choice. 

Output:
                                    '''.strip()
            fs_direct_prompt = f'''{fs_messages['direct']}Question: {ex["question"]}

{choice_str}

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.
                        '''.strip()

            examples.append({
                **ex,
                'dataset_type': self.dataset_types.mmlu,
                'prompt_parts': {
                    'zs_cot_prompt': cot_prompt,
                    'zs_cotless_prompt': direct_prompt,
                    'fs_cot_prompt': fs_cot_prompt,
                    'fs_cotless_prompt': fs_direct_prompt,
                    'cot_system_prompt': self.default_sys_mc_cot_prompt,
                    'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                },
                'lm_eval_messages': lm_eval_messages,
                'answer': answer,
                'answer_index': answer_index,
                'answerKey': choice_labels[answer_index],
                'answer_choice_tokens': choice_labels,
                "choices": choices,
                'few_shot_question_response_pairs': {'cot': fs_messages['cot_question_response_pair'], 'direct': fs_messages['ds_question_response_pair']},
            })


        random.Random(1).shuffle(examples)
        if self.use_llama_3_1_prompts:
            examples = self.load_llama_3_1_prompts(examples)

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


    def build_fewshot_context(self, yaml):
        cot_fs_ctx = ''
        direct_fs_ctx = ''
        cot_question_response_pair = []
        ds_question_response_pair = []

        samples = yaml['fewshot_config']['samples']
        for sample in samples:
            question = sample['question']
            target = sample['target']
            question_parts = question.split('\n(A)')
            question = question_parts[0].strip()
            choices = '(A) ' + question_parts[1].strip()
            choices = choices.replace('(A)', '( A )').replace('(B)', '( B )').replace('(C)', '( C )').replace('(D)', '( D )').replace('(E)', "( E )")
            choices = ['(' + x for x in choices.split('(') if x != '']
            choices = '\n'.join(choices)

            fs_cot_prompt = f'''
Question: {question}

{choices}

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.  You may only pick one answer choice. 
            '''.strip()

            fs_direct_prompt = f'''
Question: {question}

{choices}

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.
            '''.strip()


            response = 'Output: ' + target
            for char in ['A','B','C','D','E']:

                if f'The answer is ({char}).' in response:
                    response = response.replace(f'The answer is ({char}).', f'Answer: {char}')
                    direct_fs_ctx += f"{fs_direct_prompt}\n\nAnswer: {char}\n\n\n\n"

                    cot_question_response_pair.append([question, response])
                    ds_question_response_pair.append([question, f'Answer: {char}'])

            cot_fs_ctx += f"{fs_cot_prompt}\n\n{response}\n\n\n\n"

        return {'cot': cot_fs_ctx, 'direct': direct_fs_ctx, 'cot_question_response_pair': cot_question_response_pair, 'ds_question_response_pair': ds_question_response_pair}


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = MMLUDataset()

    ex = dataset[0]


    responses = [
"""ANSWER: The components of memory can be thought of as working one at a time (A). However, it's also important to note that they can work as a system (C) and take turns (D), as different types of memory interact and rely on each other. But the initial response focuses on the individual nature of each component.
"""
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])
