import random


from datasets import load_dataset


from eval_datasets import ReasoningDataset, CSQADataset
import json
from pathlib import Path
from src.utils.paths import ROOT_FOLDER


class MMLUProDataset(ReasoningDataset):
    # 1000 for olmo
    average_token_len = 1500

    def __init__(self, path_or_url='TIGER-Lab/MMLU-Pro', split='test',  raw_llama31_prompts: bool = False, use_llama_3_1_prompts: bool = True, reload_llama_3_1_prompts: bool = False, llama_3_1_prompts_cache_file: str = ROOT_FOLDER / 'eval_datasets/third_party/llama_3_1_prompts_tmp_folder/mmlu_pro.json', *args, **kwargs):
        self.llama_3_1_prompts_cache_file = llama_3_1_prompts_cache_file
        self.use_llama_3_1_prompts = use_llama_3_1_prompts
        self.reload_llama_3_1_prompts = reload_llama_3_1_prompts
        self.raw_llama31_prompts = raw_llama31_prompts
        llama_3_1_prompts_cache_file.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(path_or_url + ':' + split, *args, **kwargs)
        random.seed(0)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.mmlu_pro

    def load_llama_3_1_prompts(self, examples):
        if self.llama_3_1_prompts_cache_file.is_file() and not self.reload_llama_3_1_prompts:
            llama_prompts_per_task = json.load(open(self.llama_3_1_prompts_cache_file, 'r'))
        else:
            llama_prompts_per_task = {}
            raw_prompt_dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
                                              "Meta-Llama-3.1-8B-Instruct-evals__mmlu_pro__details")["latest"]
            for x in raw_prompt_dataset:
                subject = x['subtask_name'].replace('mmlu_pro_chat.', '')
                llama_prompts_per_task.setdefault(subject, []).append(
                    {'prompt': x['input_final_prompts'], 'question': x['input_question'],
                     'eval_config': x['eval_config'], 'was_correct': x['is_correct']})

        for exidx, ex in enumerate(examples):
            question = ex['question']
            subject = ex['category'].replace(" ", "_")

            if self.use_llama_3_1_prompts:
                found = False
                words = question.split()
                llama_3_1_prompt = None
                found_prompt = None
                for prompt in llama_prompts_per_task[subject]:
                    if prompt['question'] == question:
                        found = True
                        llama_3_1_prompt = prompt['prompt'][0]
                        found_prompt = prompt
                if not found:
                    for prompt in llama_prompts_per_task[subject]:
                        if sum([x in prompt['question'] for x in words])/len(words) > 0.9:
                            found = True
                            llama_3_1_prompt = prompt['prompt'][0]
                            found_prompt = prompt
                if found:
                    llama_3_1_msgs = self.convert_raw_to_messages(llama_3_1_prompt)

                    few_shot_cot_prompt = llama_3_1_msgs
                    few_shot_direct_prompt = self.build_few_shot_direct_prompt(llama_3_1_msgs) + [{'role': 'assistant', 'content': 'The best answer is '}]
                    zero_shot_cot_prompt = [llama_3_1_msgs[-1]]
                    zero_shot_direct_prompt = [few_shot_direct_prompt[-2]] + [{'role': 'assistant', 'content': 'The best answer is '}]

                    examples[exidx]['llama_3_1_eval'] = {'prompts': {
                        'fs_cot': few_shot_cot_prompt if not self.raw_llama31_prompts else llama_3_1_prompt,
                        'fs_direct': few_shot_direct_prompt,
                        'zs_cot': zero_shot_cot_prompt,
                        'zs_direct': zero_shot_direct_prompt,
                        'raw_fs_cot_prompt': llama_3_1_prompt
                    }, 'eval_config': {"max_gen_len": "1024", "max_prompt_len": "3840", "num_few_shot": "5", "num_generations": "1", "temperature": "0.0", "top_k": "0", "top_p": "0"}, 'few_shot_cot_was_correct': found_prompt['was_correct']}

                if not found:
                    print('hi')

        return examples

    def build_few_shot_direct_prompt(self, llama_3_1_messages):
        messages = []
        for message in llama_3_1_messages:
            if message['role'] == 'user':
                messages.append({
                    'role': 'user',
                    'content': message['content'].replace("\n\nLet's think step by step.", "").replace("Your response should end with", "You should respond with")
                })
            else:
                messages.append({
                    'role': 'assistant',
                    'content': "The best answer is " + message['content'].split("The best answer is ")[-1]
                })
        return messages

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
        dataset_url, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url,download_mode="force_redownload")[split]]
        fs_dataset = [x for x in load_dataset(dataset_url,download_mode="force_redownload")["validation"]]
        fs_prompts_per_category = {}

        for ex in dataset:
            category = ex['category']
            if category not in fs_prompts_per_category:
                fs_example_rows = [x for x in fs_dataset if x['category'] == category]
                fs_prompts_per_category[category] = self.build_fs_context(fs_example_rows)
            fs_prompts = fs_prompts_per_category[category]

            choice_labels = [x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[0:len(ex['options'])]]
            choices = '\n'.join([f'{choice_labels[i]}: {ex["options"][i]}' for i in range(len(ex['options']))])
            answer_index = ex['answer_index']
            answer = f"{ex['options'][answer_index]}"
            # prompt = f'{ex["question"]}\n{choices}\nThink step by step before giving your final answer to the question. To think step-by-step, state the facts or premises you are using along with their deductions that yield the correct answer (even if those facts or premises are commonsense knowledge).  When you are ready to answer write the answer in the format: "Answer: <your answer letter>".  You must always give an answer at the end.  You may only pick one answer choice.  You must pick an answer letter.  Let\'s think step by step.'
            cot_prompt = f'''
{ex["question"]}

{choices}

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.  You may only pick one answer choice. 
                        '''.strip()
            direct_prompt = f'''
{ex["question"]}

{choices}

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.
                        '''.strip()

            fs_cot_prompt = f'''{fs_prompts['cot']}Question: {ex["question"]}

{choices}

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.  You may only pick one answer choice. 
                                                '''.strip()
            fs_direct_prompt = f'''{fs_prompts['direct']}Question: {ex["question"]}

{choices}

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
                'answer': answer,
                'answer_index': answer_index,
                'answerKey': choice_labels[answer_index],
                'answer_choice_tokens': choice_labels,
                "choices": {"label": choice_labels, "text": ex['options']},

            })

        random.Random(3).shuffle(examples)
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

    def build_fs_context(self, examples):
        cot_fs_ctx = ''
        direct_fs_ctx = ''

        for example in examples:
            question = example['question']
            target = example['cot_content']
            choice_labels = [x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[0:len(example['options'])]]
            choices = '\n'.join([f'{choice_labels[i]}: {example["options"][i]}' for i in range(len(example['options']))])

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

            response = 'Output: ' + target[3:]
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if f'The answer is ({letter}).' in response:
                    response = response.replace(f'The answer is ({letter}).', f'Answer: {letter}')
                    direct_fs_ctx += f"{fs_direct_prompt}\n\nAnswer: {letter}\n\n\n\n"
                    break


            cot_fs_ctx += f"{fs_cot_prompt}\n\n{response}\n\n\n\n"

        return {'cot': cot_fs_ctx, 'direct': direct_fs_ctx}

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = MMLUProDataset()

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