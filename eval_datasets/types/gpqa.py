import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset, CSQADataset
import json
from pathlib import Path
from src.utils.paths import ROOT_FOLDER

class GPQADataset(ReasoningDataset):
    def __init__(self, path_or_url='Idavidrein/gpqa', subset='gpqa_main', raw_llama31_prompts: bool = False, use_llama_3_1_prompts: bool = True, reload_llama_3_1_prompts: bool = True, llama_3_1_prompts_cache_file: str = ROOT_FOLDER / 'eval_datasets/third_party/llama_3_1_prompts_tmp_folder/gpqa.json', *args, **kwargs):
        self.llama_3_1_prompts_cache_file = llama_3_1_prompts_cache_file
        self.use_llama_3_1_prompts = use_llama_3_1_prompts
        self.reload_llama_3_1_prompts = reload_llama_3_1_prompts
        self.raw_llama31_prompts = raw_llama31_prompts
        llama_3_1_prompts_cache_file.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(path_or_url + ':' + subset + ':train', *args, **kwargs)
        random.seed(0)

    # 900 for olmo
    average_token_len = 2000

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.gpqa

    def load_llama_3_1_prompts(self, examples):
        if self.llama_3_1_prompts_cache_file.is_file() and not self.reload_llama_3_1_prompts:
            llama_prompts_per_task = json.load(open(self.llama_3_1_prompts_cache_file, 'r'))
        else:
            llama_prompts_per_task = []
            raw_prompt_dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
                                              "Meta-Llama-3.1-8B-Instruct-evals__gpqa__details")["latest"]
            for x in raw_prompt_dataset:
                llama_prompts_per_task.append(
                    {'prompt': x['input_final_prompts'], 'question': x['input_question'],
                     'eval_config': x['eval_config'], 'was_correct': x['is_correct'], 'model_output': x['output_prediction_text'][0], 'correct_answer_choice': x['input_correct_responses'][0]})

        for exidx, ex in enumerate(examples):
            question = ex['question']

            if self.use_llama_3_1_prompts:
                found = False
                llama_3_1_prompt = None
                found_prompt = None
                for prompt in llama_prompts_per_task:
                    if prompt['question'] == question:
                        found = True
                        llama_3_1_prompt = prompt['prompt'][0]
                        found_prompt = prompt

                if found:
                    llama_3_1_msgs = self.convert_raw_to_messages(llama_3_1_prompt)

                    N_shots = 3
                    zero_shot_cot_prompt = llama_3_1_msgs
                    zero_shot_direct_prompt = self.build_zs_direct_prompt(llama_3_1_prompt)

                    few_shot_cot_prompt = []
                    few_shot_direct_prompt = []
                    count = 0
                    for possible_shot in llama_prompts_per_task:
                        if possible_shot['question'] != question and possible_shot['was_correct']:
                            few_shot_cot_prompt.extend(self.convert_raw_to_messages(possible_shot['prompt'][0]))
                            few_shot_cot_prompt.append({'role': 'assistant', 'content': possible_shot['model_output']})
                            few_shot_direct_prompt.extend(self.build_zs_direct_prompt(possible_shot['prompt'][0]))
                            few_shot_direct_prompt.append({'role': 'assistant', 'content': f'The best answer is {possible_shot["correct_answer_choice"]}.'})
                            count += 1
                            if count == N_shots:
                                break

                    few_shot_cot_prompt.extend(llama_3_1_msgs)
                    few_shot_direct_prompt.extend(self.build_zs_direct_prompt(llama_3_1_prompt))

                    zero_shot_direct_prompt.append({'role': 'assistant', 'content': f'The best answer is '})
                    few_shot_direct_prompt.append({'role': 'assistant', 'content': 'The best answer is '})





                    examples[exidx]['llama_3_1_eval'] = {'prompts': {
                        'fs_cot': few_shot_cot_prompt,
                        'fs_direct': few_shot_direct_prompt,
                        'zs_cot': zero_shot_cot_prompt if not self.raw_llama31_prompts else llama_3_1_prompt,
                        'zs_direct': zero_shot_direct_prompt,
                        'raw_zs_cot_prompt': llama_3_1_prompt
                    }, 'eval_config': {"max_gen_len": "2048", "max_prompt_len": "4096", "num_few_shot": "0", "num_generations": "1", "temperature": "0.0", "top_k": "0", "top_p": "0"}, 'zero_shot_cot_was_correct': found_prompt['was_correct']}

                if not found:
                    print('hi')

        return examples

    def build_zs_direct_prompt(self, raw):
        msgs = self.convert_raw_to_messages(raw)
        msgs[0]['content'] = msgs[0]['content'].split('- For simple problems')[0].strip() + '\n\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.'
        return msgs

    def convert_raw_to_messages(self, raw):
        messages = []
        user_messages = [x for x in raw.split("<|start_header_id|>user<|end_header_id|>\n\n") if x != '']
        for user_message in user_messages:
            msgs = user_message.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')

            if len(msgs) == 2:

                messages.append({'role': 'user', 'content': msgs[0].replace("<|eot_id|>", "").replace(
                    "<|start_header_id|>user<|end_header_id|>", "")})
                if msgs[1] != '':
                    messages.append({'role': 'assistant', 'content': msgs[1].replace("<|eot_id|>", "")})
            else:
                messages.append({'role': 'user', 'content': user_message.replace("<|eot_id|>", "").replace(
                    "<|start_header_id|>user<|end_header_id|>", "")})
        return messages

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, subset, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url, subset)[split]]

        r = random.Random(1)
        for ex in dataset:
            choice_formats = {
                'text': [],
                'label': []
            }
            raw_choices = [ex['Correct Answer'], ex['Incorrect Answer 1'], ex['Incorrect Answer 2'], ex['Incorrect Answer 3']]
            cidx = [0, 1, 2, 3]
            r.shuffle(cidx)
            for lidx, idx in enumerate(cidx):
                choice_formats['text'].append(raw_choices[idx])
                choice_formats['label'].append(chr(65 + lidx))

            choices = self.format_choices(choice_formats) #'\n'.join([f'{choice_formats["label"][i]}: {choice_formats["text"][i]}' for i in range(len(choice_formats["label"]))])

            answer_index = choice_formats['text'].index(ex['Correct Answer'])
            answerKey = choice_formats['label'][answer_index]
            answer = choice_formats['text'][answer_index]
            #prompt = f'{ex["Question"]}\n{choices}\nThink step by step before giving your final answer to the question. To think step-by-step, state the facts or premises you are using along with their deductions that yield the correct answer (even if those facts or premises are commonsense knowledge).  When you are ready to answer write the answer in the format: "Answer: <your answer>".  Failure to follow the answer format will result in no credit. You must always give an answer at the end.  You may only pick one answer choice. You must pick an answer and that answer must be one of the multiple choice options.  Let\'s think step by step.'
#             prompt = f'''
# {ex["Question"]}
#
# {choices}
#
# Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end.  You may only pick one answer choice.
#             '''.strip()
            prompt = self.basic_prompt(ex["Question"], choices)
#             prompt = f'''
# Give an answer for the question below.  Write your answer in the format "Answer: <your answer>". You must always given an answer and you may only choose one answer choice.
#
# Question:
# {ex["Question"]}
#
# Answer Choices:
# {choices}
#
# Remember to give your answer in the format, "Answer: <your answer choice>".
#                     '''.strip()

            examples.append({
                **ex,
                'question': ex["Question"],
                'choices': choice_formats,
                'dataset_type': self.dataset_types.gpqa,
                'prompt_parts': {'user_context': prompt},
                'answer': answer,
                'answer_index': answer_index,
                'answerKey': answerKey,
                'answer_choice_tokens': choice_formats['label']
            })

        if self.use_llama_3_1_prompts:
            examples = self.load_llama_3_1_prompts(examples)
        return examples

    def basic_prompt(self, question, choices):
            return f'''Answer the question below. You must give an answer by saying "Answer: <Letter>".
{question}

{choices}

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.
                '''.strip()

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

            if '**' in line:
                potential_answer_segments.append(line.split('**')[1].strip())

            if '$' in line:
                try:
                    potential_answer_segments.append(line.split('$')[1].split('$')[0].strip())
                except Exception:
                    pass
            if 'Answer: ' in line:
                try:
                    potential_answer_segments.append(line.split('Answer: ')[1].split('.')[0].strip())
                except Exception:
                    pass
            if 'ANSWER: ' in line:
                try:
                    potential_answer_segments.append(line.split('ANSWER: ')[1].split('.')[0].strip())
                except Exception:
                    pass
            if 'answer: ' in line:
                try:
                    potential_answer_segments.append(line.split('answer: ')[1].split('.')[0].strip())
                except Exception:
                    pass

            for line in potential_answer_segments:
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
                         'correct': True,
                         'answer_randomly_sampled': False, 'model_answer': chosen_answers[0], **example})
                    found = True
                    break
                if correct and incorrect:
                    found = False
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
                'correct': False,
                'answer_randomly_sampled': True,
                'model_answer': None,
                **example
            })
        return returned_answers


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = GPQADataset()

    ex = dataset[0]


    responses = [
        """A three-body bound state is a state where all three nucleons are bound together, unlike two-body bound states which involve only two nucleons.

The presence of two-body bound states does not guarantee the existence of three-body bound states. In quantum mechanics, the formation of a three-body bound state is influenced by the interplay of the two-body interactions and the three-body force.

If the three-body force is attractive enough to overcome the repulsive forces between the nucleons, then a three-body bound state can form. However, if the three-body force is not strong enough, or if it is repulsive, then a three-body bound state will not form, even if two-body bound states exist.

Therefore, the presence of two-body bound states does not necessarily imply the presence of three-body bound states.

Answer: D) A three-body bound state may occur regardless of whether two-body bound states occur.

"""
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])

    print(ex['messages'][0]['content'])
