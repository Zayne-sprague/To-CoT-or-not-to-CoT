import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset


class CSQADataset(ReasoningDataset):
    def __init__(self, path_or_url='commonsense_qa', split='validation', *args, **kwargs):
        super().__init__(path_or_url + ':' + split, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.csqa

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url)[split]]

        for ex in dataset:
            choices = self.format_choices(ex["choices"]) #'\n'.join([f'( {ex["choices"]["label"][i]} ) {ex["choices"]["text"][i]}' for i in range(len(ex["choices"]["label"]))])
            answer_index = ex['choices']['label'].index(ex['answerKey'])
            answer = ex['choices']['text'][answer_index]

            answer_choice_tokens = [f'{ex["choices"]["label"][i]}' for i in range(len(ex["choices"]["label"]))]

            zs_cot_prompt = f'Question: {ex["question"]}\n\nAnswer Choices:\n{choices}\n\nThink step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'
            zs_cotless_prompt = f'Question: {ex["question"]}\n\nAnswer Choices:\n{choices}\n\nOnly write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'
            fs_cot_prompt = f'{prompt_cot_examples()}\n\nQuestion: {ex["question"]}\n\nAnswer Choices:\n{choices}\n\nPlease use the examples above as a guide when solving this problem. Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'
            fs_cotless_prompt = f'{prompt_cotless_examples()}\n\nQuestion: {ex["question"]}\n\nAnswer Choices:\n{choices}\n\nPlease use the examples above as a guide when solving this problem. Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.'

            examples.append({
                'dataset_type': self.dataset_types.csqa,
                'prompt_parts': {
                    'zs_cot_prompt': zs_cot_prompt,
                    'zs_cotless_prompt': zs_cotless_prompt,
                    'fs_cot_prompt': fs_cot_prompt,
                    'fs_cotless_prompt': fs_cotless_prompt,
                    'cot_system_prompt': self.default_sys_mc_cot_prompt,
                    'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                },
                'answer_choice_tokens': answer_choice_tokens,
                'answer': answer,
                'answer_index': answer_index,
                **ex
            })

        return examples



    @classmethod
    def evaluate_with_model(cls, model_resp, example, *args, model=None, **kwargs):
        assert model is not None, 'evaluating with a model requires the model kwarg to be passed through.'

        choices = '\n'.join(
            [f'{example["choices"]["text"][i]}' for i in range(len(example["choices"]["label"]))])

        ex1_choices = 'bank\nlibrary\ndepartment store\nmall\nnew york'
        ex2_choices = 'doctor\nbookstore\nmarket\ntrain station\nmortuary'

        instructions = f'Parse out the multiple choice label and answer in this response. The model is given the following choices,\n[choices]\n\nparse out the letter choice written in the response along with the actual choice reported in the response (these may differ in some responses). Use the choices to select which answer is reported on in the response. Letter choices should be A, B, C, D, E, or None (None for when no answer is selected or multiple answers are selected) and may differ from the selected letter choice in the response.  You should response only with "Answer: <letter choice> | <actual choice>".'


        prompt = f'''
EXAMPLE

Answer: B - library.

Here's my reasoning:

A revolving door is a door that rotates to allow people to enter and exit a building in both directions. This means that a revolving door can facilitate easy movement in and out of a building, making it convenient for people to enter and exit.

Now, think about a place where security is a concern. A library is a place where valuable materials and assets are stored, and it's important to control who can enter and exit the building. A revolving door can serve as a security measure in a library because it allows only authorized personnel to enter and exit, while still allowing patrons to easily enter and exit the building.

So, the answer is B library.

----

{instructions.replace('[choices]', ex1_choices)}

Output:

Answer: B | library

====

EXAMPLE

Sure, I'm ready to help!

A revolving door is convenient for two-way travel, allowing people to enter and exit a building easily without having to go around it. However, it also serves as a security measure in a:

Bank

Here's why:

* Banks often have sensitive information and valuable assets, and a revolving door can help to control the flow of people in and out of the building.
* By requiring people to enter and exit through the same door, a revolving door can help to prevent unauthorized access to the bank's secure areas.
* Additionally, a revolving door can help to reduce the risk of tailgating, where someone follows someone else into a secure area without proper authorization.

Therefore, the answer is (B) Bank.

----

{instructions.replace('[choices]', ex1_choices)}

Output:

Answer: B | bank

====

EXAMPLE

Sure, I'm ready to help! Here are the facts and premises related to the question:

A revolving door is a device that allows people to pass through it in both directions, making it convenient for two-way travel.

A revolving door is often used as a security measure in various places, such as:

* Banks
* Libraries
* Department stores
* Malls

Now, let's think step-by-step to determine the answer:

Since the question states that the revolving door serves as a security measure, we can conclude that the answer must be one of the options that include "security measure" in the description.

Option A mentions "bank", which does not include "security measure" in its description. Therefore, we can eliminate option A.

Option B mentions "library", which does not include "security measure" in its description. Therefore, we can eliminate option B.

Option C mentions "department store", which does include "security measure" in its description. Therefore, we can choose option C as the answer.

So, the answer is:

Answer: Option C - department store

----

{instructions.replace('[choices]', ex1_choices)}

Output:

ANSWER: C | department store

====

EXAMPLE

{model_resp}

----

{instructions.replace('[choices]', choices)}

Output:
            '''.strip()

        out = model.parse_out(model.inference(prompt, temperature=0, top_p=0.99))

        parsed = out[0].replace("ANSWER: ", "").split('|')
        proposed_label = parsed[0].strip()

        if proposed_label.lower() in [x.lower() for x in example['choices']['text']]:
            proposed_label = example['choices']['label'][
                [x.lower() for x in example['choices']['text']].index(proposed_label.lower())]

        proposed_choice = parsed[1].strip().lower()

        return proposed_label, proposed_choice


    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            *args, model=None, **kwargs
    ):


        answers = [example['answer']]
        answer_labels = [f"{example['answerKey']}"]
        incorrect_answers = [*[x for x in example['choices']['text'] if x not in answers]]
        incorrect_answer_labels = [*[f'{x}' for x in example['choices']['label'] if x not in answer_labels]]

        returned_answers = []

        def parse_ans(word):
            strip_outs = [':', '.', ',', '<', '>', '!', '?', '(', ')', '[', ']', '{', '}', '\'', '\"', '’', '‘', '“',
                          '”']
            for x in strip_outs:
                word = word.strip().replace(x, '')
            return word

        for resp in model_responses:
            if resp is None or resp=='':
                returned_answers.append({'errored': True, 'correct': False, 'model_response': None, 'answer_line': None, 'model_answer': None, **example})
                continue

            parsed_resp = resp.strip()
            found = False

            if f'[invalid] - configuration was not run' in parsed_resp:
                returned_answers.append(
                    {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line':None,
                     'correct': False, 'answer_span': None,
                     'answer_randomly_sampled': False, 'model_answer': None, **example})
                found = True
                continue

            found_answer = None
            for x in answer_labels + answers:
                if f'ANSWER: {x}' in parsed_resp:
                    found_answer = \
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'ANSWER: {x}',
                         'correct': True, 'answer_span': [parsed_resp.index(f'ANSWER: {x}') + len("ANSWER: "), parsed_resp.index(f'ANSWER: {x}') + len("ANSWER: ") + len(f'{x}')],
                         'answer_randomly_sampled': False, 'model_answer': x, **example}
                    found = True
                    break
            for x in incorrect_answer_labels + incorrect_answers:
                if f'ANSWER: {x}' in parsed_resp:
                    found_answer = \
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line':  f'ANSWER: {x}',
                         'correct': False, 'answer_span': [parsed_resp.index(f'ANSWER: {x}') + len("ANSWER: "), parsed_resp.index(f'ANSWER: {x}') + len("ANSWER: ") + len(f'{x}')],
                         'answer_randomly_sampled': False, 'model_answer': f'{x}', **example}
                    found = True
                    break
            if found:
                returned_answers.append(found_answer)
                continue

            for x in answer_labels + answers:
                if f'Answer: {x}' in parsed_resp:
                    found_answer = \
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'Answer: {x}',
                         'correct': True, 'answer_span': [parsed_resp.index(f'Answer: {x}') + len("Answer: "), parsed_resp.index(f'Answer: {x}') + len("Answer: ") + len(f'{x}')],
                         'answer_randomly_sampled': False, 'model_answer': x, **example}
                    found = True
                    break
            for x in incorrect_answer_labels + incorrect_answers:
                if f'Answer: {x}' in parsed_resp:
                    found_answer = \
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line':  f'Answer: {x}',
                         'correct': False, 'answer_span': [parsed_resp.index(f'Answer: {x}') + len("Answer: "), parsed_resp.index(f'Answer: {x}') + len("Answer: ") + len(f'{x}')],
                         'answer_randomly_sampled': False, 'model_answer': f'{x}', **example}
                    found = True
                    break
            if found:
                returned_answers.append(found_answer)
                continue


            for x in answer_labels + answers:
                if f'answer: {x}' in parsed_resp:
                    found_answer = \
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'answer: {x}',
                         'correct': True, 'answer_span': [parsed_resp.index(f'answer: {x}') + len("answer: "), parsed_resp.index(f'answer: {x}') + len("answer: ") + len(f'{x}')],
                         'answer_randomly_sampled': False, 'model_answer': x, **example}
                    found = True
                    break
            for x in incorrect_answer_labels + incorrect_answers:
                if f'answer: {x}' in parsed_resp:
                    found_answer = \
                        {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line':  f'answer: {x}',
                         'correct': False, 'answer_span': [parsed_resp.index(f'answer: {x}') + len("answer: "), parsed_resp.index(f'answer: {x}') + len("answer: ") + len(f'{x}')],
                         'answer_randomly_sampled': False, 'model_answer': f'{x}', **example}
                    found = True
                    break
            if found:
                returned_answers.append(found_answer)
                continue


            # We are going to choose the answer to be the first time we an answer choice or index mentioned...
            #early escape following llama 3 eval
            if 'he best answer is' in parsed_resp and not found:

                choice = parsed_resp.split('he best answer is')[-1].split('.')[0].strip()
                ret_ans = None

                chosen_choice_format = None

                for x in answer_labels:
                    chosen_choice_format = x
                    if f'({x.lower()})' in choice:
                        chosen_choice_format = f'({x.lower()})'
                    elif f'{x.lower()})' in choice:
                        chosen_choice_format = f'{x.lower()})'

                    if chosen_choice_format in choice:
                        ret_ans = {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{chosen_choice_format}', 'correct': True,
                             'answer_randomly_sampled': False, 'model_answer': x, **example}
                        found = True
                        break
                if not found:
                    for x in incorrect_answer_labels:
                        chosen_choice_format = x
                        if f'({x.lower()})' in choice:
                            chosen_choice_format = f'({x.lower()})'
                        elif f'{x.lower()})' in choice:
                            chosen_choice_format = f'{x.lower()})'

                        if chosen_choice_format in choice:
                            ret_ans = {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{chosen_choice_format}', 'correct': False,
                                 'answer_randomly_sampled': False, 'model_answer': x, **example}
                            found = True
                            break
                if not found:
                    for x in answers:
                        if x in choice:
                            ret_ans = {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{choice}', 'correct': True,
                                 'answer_randomly_sampled': False, 'model_answer': x, **example}
                            found = True
                            break
                if not found:
                    for x in incorrect_answers:
                        if x in choice:
                            ret_ans = {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{choice}', 'correct': False,
                                 'answer_randomly_sampled': False, 'model_answer': x, **example}
                            found = True
                            break

                if ret_ans is not None:
                    try:
                        curr_loc = [parsed_resp.index(choice) + choice.index(ret_ans['answer_line']), parsed_resp.index(choice) + choice.index(ret_ans['answer_line']) + len(ret_ans['answer_line'])]
                    except Exception:
                        print('hi')
                    ret_ans['answer_span'] = curr_loc
                    returned_answers.append(ret_ans)
                # else:
                #     print('hi')

            if 'he answer is therefore ' in parsed_resp and not found:
                if 'he answer is therefore **' in parsed_resp:
                    choice = parsed_resp.split('he answer is therefore **')[1].split('**')[0].strip()
                else:
                    choice = parsed_resp.split('he answer is therefore ')[1][0].strip()

                for x in answer_labels:
                    if f'{x}' in choice:
                        returned_answers.append(
                            {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{choice}', 'correct': True,
                             'answer_span': [parsed_resp.index(choice) + choice.index(x), parsed_resp.index(choice) + choice.index(x) + len(x)],
                             'answer_randomly_sampled': False, 'model_answer': x, **example})
                        found = True
                        break
                if not found:
                    for x in incorrect_answer_labels:
                        if x in choice:
                            returned_answers.append(
                                {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{choice}', 'correct': False,
                                 'answer_span': [parsed_resp.index(choice) + choice.index(x),
                                                 parsed_resp.index(choice) + choice.index(x) + len(x)],
                                 'answer_randomly_sampled': False, 'model_answer': x, **example})
                            found = True
                            break
                if not found:
                    for x in answers:
                        if x in choice:
                            returned_answers.append(
                                {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{choice}', 'correct': True,
                                 'answer_span': [parsed_resp.index(choice) + choice.index(x),
                                                 parsed_resp.index(choice) + choice.index(x) + len(x)],
                                 'answer_randomly_sampled': False, 'model_answer': x, **example})
                            found = True
                            break
                if not found:
                    for x in incorrect_answers:
                        if x in choice:
                            returned_answers.append(
                                {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': f'{choice}', 'correct': False,
                                 'answer_span': [parsed_resp.index(choice) + choice.index(x),
                                                 parsed_resp.index(choice) + choice.index(x) + len(x)],
                                 'answer_randomly_sampled': False, 'model_answer': x, **example})
                            found = True
                            break

            if not found:
                for lidx, line in enumerate(reversed(parsed_resp.split('\n'))):
                    curr_idx_count = sum([len(x)+1 for x in parsed_resp.split('\n')[:len(parsed_resp.split('\n')) - lidx - 1]])

                    # so long as that answer is mentioned on a line with "answer:" in it.
                    if 'answer' not in line.lower():
                        continue

                    correct = any(
                        [any([f'{x}' == parse_ans(y) for y in
                              line.split(' ')]) for x
                         in answer_labels])

                    incorrect = any(
                        [any([x == parse_ans(y) for y in
                              line.split(' ')]) for x in incorrect_answer_labels])

                    if not incorrect and not correct:
                        incorrect = any(
                            [x.lower() in line.lower() for x
                             in incorrect_answers])

                        correct = any([x.lower() in line.lower() for x in answers])

                    chosen_answers = []

                    for xidx, x in enumerate(answers):
                        if x.lower() in line.lower():
                            chosen_answers.append(answers[xidx])
                    for x in answer_labels:
                        for y in line.split(' '):
                            if f'{x}' == parse_ans(y):
                                chosen_answers.append(x)

                    for xidx, x in enumerate(incorrect_answers):
                        if x.lower() in line.lower():
                            chosen_answers.append(incorrect_answers[xidx])

                    for x in incorrect_answer_labels:
                        for y in line.split(' '):
                            if x == parse_ans(y):
                                chosen_answers.append(x)

                    if correct and not incorrect:
                        returned_answers.append(
                            {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': line, 'correct': True,
                             'answer_randomly_sampled': False, 'model_answer': chosen_answers[0],
                             'answer_span': [curr_idx_count + cls.safe_find(parsed_resp[curr_idx_count:], chosen_answers[0]), curr_idx_count + cls.safe_find(parsed_resp[curr_idx_count:], chosen_answers[0]) + len(chosen_answers[0])],
                             **example})
                        found = True

                        break
                    if correct and incorrect:
                        found = False
                        break
                    elif incorrect:
                        returned_answers.append(
                            {'answer_label': answer_labels[0], 'model_response': resp, 'answer_line': line,
                             'correct': False,
                             'answer_randomly_sampled': False, 'model_answer': chosen_answers[-1],
                             'answer_span': [curr_idx_count + cls.safe_find(parsed_resp[curr_idx_count:], chosen_answers[-1]),
                                             curr_idx_count + cls.safe_find(parsed_resp[curr_idx_count:], chosen_answers[-1]) + len(
                                                 chosen_answers[-1])],
                             **example})
                        found = True
                        break

            if found:
                continue
            else:
                # Fallback, may be specific for different datasets.
                from experiments.utils import get_custom_eval_response

                fn = get_custom_eval_response(example)
                out = fn([resp], example, returned_answers)
                if out is not None:
                    returned_answers.extend(out)
                    continue

            # If randomly_select_when_unparsable is on, and we did not find an answer in the response, we will randomly
            #  assign an answer to this generation but we mark it as such.
            returned_answers.append({
                'model_response': resp,
                'answer_line': None,
                'correct': True if random.random() <= 1 / len(
                    example['choices']) and randomly_select_when_unparsable else False,
                'answer_randomly_sampled': True,
                'model_answer': None,
                'answer_span': None,
                **example
            })

        for x in returned_answers:
            span = x.get('answer_span')
            if span is not None:
                x['parsed_answer_span'] = x['model_response'][x['answer_span'][0]:x['answer_span'][1]]
                # if x['parsed_answer_span'].lower() != x['model_answer'].lower():
                #     print('hit')
                #     raise Exception('Span does not match model answer!!!')
            # if x['model_answer'] is not None and span is None:
            #     raise Exception('Model answer is not None but span is None!!!')

            labels = x['choices']['label']
            choices = [x.lower() for x in x['choices']['text']]
            answer = x['model_answer']
            x['raw_model_answer'] = x['model_answer']
            if answer not in labels and answer is not None:
                x['model_answer'] = labels[choices.index(answer.lower())]

        if len(returned_answers) != len(model_responses):
            print('hi')
        assert len(returned_answers) == len(model_responses) or len(returned_answers) == 0, 'Returned answers does not match model responses.'
        return returned_answers

    @classmethod
    def custom_evaluate_response(cls, model_responses, example, *args, model=None, **kwargs):
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
                returned_answers.append({
                    'model_response': resp,
                    'answer_line': None,
                    'correct': False,
                    'answer_randomly_sampled': True,
                    'model_answer': None,
                    'answer_span': None,
                    **example
                })
                continue

            potential_answer_segments = []

            for _l in resp.split('\n'):
                for l in _l.split('.'):
                    if 'answer:' in l.lower():
                        potential_answer_segments.append(l)

            if len(resp.split('.')) <= 2:
                potential_answer_segments.append(line)
            for sent in resp.split('.'):
                if any([x in sent.lower() for x in [
                    'the answer is', 'final answer', 'the best answer is',
                    'answer is', 'is the answer', "would choose option", "would answer", "my answer is",
                    "i will choose", "the answer to the question", "based on the information provided", "my answer",
                ]]):
                    potential_answer_segments.append(sent)
            if len(resp.split('.')) > 3:
                potential_answer_segments.append(line.split('.')[-1])
            if '**' in line:
                potential_answer_segments.append(line.split('**')[1].strip())
            if '(' in line and ')' in line:
                potential_answer_segments.append(line.split('(')[1].split(')')[0].strip())
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
            if 'the best answer would be ' in line:
                potential_answer_segments.append(line.split('the best answer would be ')[1].split(' ')[0])

            curr_tok_idx = sum([len(x) + 1 for x in parsed_resp.split('\n')[:-1]])
            potential_answer_segments = [[x, curr_tok_idx] for x in potential_answer_segments]

            # split line by new line and check if any line matches the format (option letter): (option text)
            potential_ans_line = None
            escape = False

            curr_tok_idx = 0
            for sent in resp.split('\n'):
                letters = example['choices']['label']
                texts = example['choices']['text']
                for idx, letter in enumerate(letters):
                    if f'{letter}: {texts[idx]}'.lower() == sent.lower().strip():
                        if potential_ans_line is not None:
                            potential_ans_line = None
                            escape = True
                            break
                        potential_ans_line = line
                if escape:
                    break
                curr_tok_idx += len(sent) + 1

            if not escape and potential_ans_line is not None:
                potential_answer_segments.append([potential_ans_line, curr_tok_idx])


            def parse_ans(word):
                strip_outs = [':', '.', ',', '<', '>', '!', '?', '(', ')', '[', ']', '{', '}', '\'', '\"', '’', '‘', '“', '”']
                for x in strip_outs:
                    word = word.strip().replace(x, '')
                return word

            for (line, tok_idx) in potential_answer_segments:
                correct = any([x.lower() in line.lower() for x in answers])

                correct = correct or any(
                    [any([f'{x}' == parse_ans(y) for y
                          in
                          line.split(' ')]) for x
                     in answer_labels])

                incorrect = any(
                    [x.lower() in line.lower() for x
                     in incorrect_answers])

                incorrect = incorrect or any(
                    [any([x == parse_ans(y)
                          for y in
                          line.split(' ')]) for x in incorrect_answer_labels])

                chosen_answers = []

                for xidx, x in enumerate(answers):
                    if x.lower() in line.lower():
                        chosen_answers.append(answer_labels[xidx])
                for x in answer_labels:
                    for y in line.split(' '):
                        if f'{x}' == parse_ans(y):
                            chosen_answers.append(x)

                for xidx, x in enumerate(incorrect_answers):
                    if x.lower() in line.lower():
                        chosen_answers.append(incorrect_answers[xidx])

                for x in incorrect_answer_labels:
                    for y in line.split(' '):
                        if x == parse_ans(y):
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

def prompt_cot_examples():
        return f'''Question: What do people use to absorb extra ink from a fountain pen? 

Answer Choices:
( A ) shirt pocket 
( B ) calligrapher’s hand 
( C ) inkwell
( D ) desk drawer
( E ) blotter

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink.

Answer: E

Question: What home entertainment equipment requires cable?

Answer Choices: 
( A ) radio shack 
( B ) substation
( C ) television
( D ) cabinet

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The answer must require cable. Of the above choices, only television requires cable. 

Answer: C

Question: The fox walked from the city into the forest, what was it looking for? 

Answer Choices: 
( A ) pretty flowers 
( B ) hen house 
( C ) natural habitat 
( D ) storybook

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. 

Answer: B

Question: Sammy wanted to go to where the people were. Where might he go? 

Answer Choices: 
( A ) populated areas
( B ) race track 
( C ) desert 
( D ) apartment 
( E ) roadblock

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of
people. 

Answer: A

Question: Where do you put your grapes just before checking out? 

Answer Choices: 
( A ) mouth 
( B ) grocery cart 
( C ) super market 
( D ) fruit basket 
( E ) fruit market

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. 

Answer: B

Question: Google Maps and other highway and street GPS services have replaced what? 

Answer Choices: 
( A ) united states 
( B ) mexico 
( C ) countryside 
( D ) atlas

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. 

Answer: D

Question: Before getting a divorce, what did the wife feel who was doing all the work? 

Answer Choices: 
( A ) harder 
( B ) anguish 
( C ) bitterness 
( D ) tears 
( E ) sadness

Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Output: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. 

Answer: C
    '''.strip()


def prompt_cotless_examples():
    return f'''Question: What do people use to absorb extra ink from a fountain pen? 

Answer Choices:
( A ) shirt pocket 
( B ) calligrapher’s hand 
( C ) inkwell
( D ) desk drawer
( E ) blotter

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: E

Question: What home entertainment equipment requires cable?

Answer Choices: 
( A ) radio shack 
( B ) substation
( C ) television
( D ) cabinet

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: C

Question: The fox walked from the city into the forest, what was it looking for? 

Answer Choices: 
( A ) pretty flowers 
( B ) hen house 
( C ) natural habitat 
( D ) storybook

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: B

Question: Sammy wanted to go to where the people were. Where might he go? 

Answer Choices: 
( A ) populated areas
( B ) race track 
( C ) desert 
( D ) apartment 
( E ) roadblock

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: A

Question: Where do you put your grapes just before checking out? 

Answer Choices: 
( A ) mouth 
( B ) grocery cart 
( C ) super market 
( D ) fruit basket 
( E ) fruit market

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: B

Question: Google Maps and other highway and street GPS services have replaced what? 

Answer Choices: 
( A ) united states 
( B ) mexico 
( C ) countryside 
( D ) atlas

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: D

Question: Before getting a divorce, what did the wife feel who was doing all the work? 

Answer Choices: 
( A ) harder 
( B ) anguish 
( C ) bitterness 
( D ) tears 
( E ) sadness

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.

Answer: C
    '''.strip()



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = CSQADataset()

    ex = dataset[0]


    responses = [
        """
To watch a movie without leaving home, we need a device or service that can stream or play movies. The options given are:

( A ) drive in movie - This is not a device or service for home use.
( B ) drive in movie - Again, this is not a device or service for home use.
( C ) television - While a television can be used to watch movies, it requires a separate device like a DVD player, cable box, or streaming device to play movies at home.
( D ) video store - This is a physical location where you can rent or buy movies on DVD or Blu-ray, but it doesn't allow you to watch movies at home without additional equipment.
( E ) show - This term is too vague to be the correct answer.

The best answer is:

Answer: C) television (with a streaming device or cable box) or E) (a streaming service, such as Netflix, Hulu, Amazon Prime Video, etc.)

        """
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])

import json
from key_handler import KeyHandler
from datasets import load_dataset
def get_analysis_ready():

    dataset = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project", "meta-llama__Meta-Llama-3.1-8B-Instruct",
                           token=KeyHandler.hf_key)
    examples = [x for x in dataset['biggen_bench']]
    return dataset, examples

def filter_exs(dataset, key = None):
    if key:
        directs = [x['zero_shot_direct_model_response'] + '\n\n===\n' + x['zero_shot_direct_parsed_model_answer'] for x in
                   dataset['biggen_bench'] if json.loads(x['additional_information'])['task'] == key]
        cots = [x['zero_shot_cot_model_response'] + '\n\n===\n' + x['zero_shot_cot_parsed_model_answer'] for x in
                dataset['biggen_bench'] if json.loads(x['additional_information'])['task'] == key]
    else:
        directs = [x['zero_shot_direct_model_response'] + '\n\n===\n' + x['zero_shot_direct_parsed_model_answer'] for x in
                   dataset['biggen_bench']]
        cots = [x['zero_shot_cot_model_response'] + '\n\n===\n' + x['zero_shot_cot_parsed_model_answer'] for x in
                dataset['biggen_bench']]

    return directs, cots

# dataset, examples = get_analysis_ready()