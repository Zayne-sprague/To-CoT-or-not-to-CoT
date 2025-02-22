import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset
from src.utils.paths import ROOT_FOLDER
import json
import re
import string
from openai import OpenAI
from key_handler import KeyHandler


KeyHandler.set_env_key()
client = OpenAI()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    try:
        return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()
    except:
        return s

def gpt_normalize_answer(cot_answer, candidate_answer):
    messages = [
        {
            "role": "system",
            "content": "Here are one answer to a multi-hop reasoning question and one candidate answer. Tell me whether the answer is the same as the candidate answer. You should only look at the content of the answers and don't focus on 100% verbatim match. You should answer 'yes' if they are the same and 'no' if they are different and nothing else."
        },
        {
            "role": "user",
            "content": "The answer is: " + normalize_answer(cot_answer) + '\n' + "The candidate answer is: " + normalize_answer(candidate_answer)
        },
    ]
            
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,  # Ensure messages is a list
    )
    # print(response.choices[0].message)
    if response.choices[0].message.content == 'yes':
        # print(normalize_answer(cot_answer), normalize_answer(candidate_answer))
        cot_answer = candidate_answer
        return True, cot_answer
    else:
        return False, cot_answer

class MusiqueDatasetAll(ReasoningDataset):
    # 1000 for olmo
    average_token_len = 1500

    def __init__(self, path_or_url=ROOT_FOLDER / 'eval_datasets/thirdparty/musique', split='validation', subset="all", *args, **kwargs):
        if subset == "all":
            path_or_url = path_or_url / 'musique_full_v1.0_dev.jsonl'
            # path_or_url = path_or_url / 'musique_full_v1.0_dev_substitute.jsonl'
        super().__init__(path_or_url, *args, generating_paraphrase=True, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.musique_all

    def load_dataset(self, path_or_url):
        examples = []
        for line in open(str(path_or_url), 'r').readlines():
            ex = json.loads(line)
            if ex['answerable']:
                answers = [ex['answer']] + ex['answer_aliases']
                paragraph_context = [item['paragraph_text'] for item in ex['paragraphs'] if item['is_supporting']]
                # paragraph_context = [item['paragraph_text'] for item in ex['paragraphs']]
            else:
                answers = ["It's unanswerable"]
                n_paragraphs = random.choice([2, 3])
                paragraph_context = random.sample([item['paragraph_text'] for item in ex['paragraphs']], n_paragraphs)
                # paragraph_context = [item['paragraph_text'] for item in ex['paragraphs']]

            paragraph_context = '\n\n'.join(paragraph_context) + "\n\n"
            # the baseline cot prompt
            prompt = f'''
You are given the following paragraphs:
{paragraph_context}

Based on the paragraphs above, and not your internal knowledge, please answer the question below:
{ex["question"]}

Think step by step before giving your final answer to the question. If you think the question is unanswerable, write "Answer: It's unanswerable". When you are ready to answer write the answer in the format: "Answer: <your answer>".  You must always give an answer at the end. And I emphezise that you should only base your answer on the paragraph provided. 
                        '''.strip()
            
            cotless_prompt = f'''
You are given the following paragraphs:
{paragraph_context}

Based on the paragraphs above, and not your internal knowledge, please answer the question below:
{ex["question"]}

You must answer the question based on the paragraph provided. If you think the question is unanswerable, write "Answer: It's unanswerable". You will only give the answer and the answer alone in the format: "Answer: <your answer>".  You must always give an answer at the end. And I emphezise that you should only base your answer on the paragraph provided.
                        '''.strip()


            examples.append({
                **ex,
                'dataset_type': self.dataset_types.musique_all,
                'prompt_parts': {'zs_cot_prompt': prompt, 'zs_cotless_prompt': cotless_prompt},
                'paragraph_context': paragraph_context,
                'answers': answers,
                'question': ex['question'],
                'answerKey': "||".join(answers),
                'answer': "||".join(answers)
            })
            
        random.Random(1).shuffle(examples)
        return examples


    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            *args, **kwargs
    ):
        answers = [normalize_answer(item.lower()) for item in example['answers']]

        returned_answers = []

        for resp in model_responses:
            try:
                ans = normalize_answer(resp.split('Answer: ')[1].strip().lower())
                correct = ans in answers
                if not correct:
                    # use gpt to check whether the two answers are essentially the same
                    result, ans = gpt_normalize_answer(ans, answers[0])
                    if result:
                        correct = True
            except Exception as e:
                ans = None
                correct = False
                # print("error")
            ans_span = [-1, None]
            if ans is not None:
                ans_span = [len(ans)-1, None]
            returned_answers.append({
                'model_response': resp,
                'answer_line': ans,
                'correct': correct,
                'answer_randomly_sampled': False,
                'model_answer': ans,
                'raw_model_answer': ans,
                'answer_span': ans_span,
                **example
            })

        return returned_answers


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

    dataset = MusiqueDatasetAll()

    ex = dataset[0]


    responses = [
"""Answer: $59,039	
"""
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])
