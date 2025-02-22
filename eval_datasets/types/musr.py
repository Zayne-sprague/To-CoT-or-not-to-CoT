import random

from eval_datasets import ReasoningDataset, CSQADataset
from src.logic_tree.tree import LogicTree
from copy import deepcopy
import jsonlines

from eval_datasets.thirdparty.musr.op_icl import object_placements_solved_ex, object_placements_solved_ex_2, object_placements_solved_ex_3, object_placements_solved_ex_cotless, object_placements_solved_ex_2_cotless, object_placements_solved_ex_3_cotless
from eval_datasets.thirdparty.musr.ta_icl import team_allocation_solved_ex, team_allocation_solved_ex_2, team_allocation_solved_ex_3, team_allocation_solved_ex_cotless, team_allocation_solved_ex_2_cotless, team_allocation_solved_ex_3_cotless
from eval_datasets.thirdparty.musr.mm_icl import murder_mystery_solved_ex, murder_mystery_solved_ex_2, murder_mystery_solved_ex_3, murder_mystery_solved_ex_cotless, murder_mystery_solved_ex_2_cotless, murder_mystery_solved_ex_3_cotless


class MuSRDataset(ReasoningDataset):
    # 1000 for olmo
    # 1000 for icl
    # 500 for llama 2 + icl (except for mm)
    # 1500 otherwise
    average_token_len = 1500

    def __init__(self, path_or_url, *args, prompt_fn=None, give_full_trees: bool = False, **kwargs):
        self.give_full_trees = give_full_trees
        super().__init__(path_or_url, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.musr

    def load_dataset(self, path_or_url):
        examples = []
        data = [x for x in jsonlines.open(path_or_url, 'r').read()]
        for ex in data:
            for q in ex['questions']:
                if 'murderer' in q['question']:
                    user_ctx_fn = self.build_mm_user_context
                elif 'efficiently' in q['question']:
                    user_ctx_fn = self.build_ta_user_context
                elif 'location' in q['question']:
                    user_ctx_fn = self.build_op_user_context


                ans_index = q['answer']
                choices = q['choices']
                ans = choices[ans_index]
                question = q['question']
                context = ex['context']
                labels = ['A','B','C','D','E','F'][0:len(choices)]

                choice_str = '\n'.join(
                    [f'{labels[idx]}: {choices[idx]}' for idx in range(len(choices))])
                zs_cot_prompt = user_ctx_fn(choice_str, context, question, use_cot=True, few_shot=False)
                zs_cotless_prompt = user_ctx_fn(choice_str, context, question, use_cot=False, few_shot=False)

                fs_cot_prompt = user_ctx_fn(choice_str, context, question, use_cot=True, few_shot=True)
                fs_cotless_prompt = user_ctx_fn(choice_str, context, question, use_cot=False, few_shot=True)

                example = {
                    'dataset_type': self.dataset_types.musr,
                    'context': context,
                    'question': question,
                    # 'choices': choices,
                    'choices': {'text': choices, 'label': labels},
                    'answerKey': labels[ans_index],
                    'answer': ans,
                    'answer_index': ans_index,
                    'prompt_parts': {
                        'zs_cot_prompt': zs_cot_prompt,
                        'zs_cotless_prompt': zs_cotless_prompt,
                        'fs_cot_prompt': fs_cot_prompt,
                        'fs_cotless_prompt': fs_cotless_prompt,
                        'cot_system_prompt': self.default_sys_mc_cot_prompt,
                        'cotless_system_prompt': self.default_sys_mc_cotless_prompt
                    },
                    'answer_choice_tokens': [f'{idx + 1}' for idx in range(len(choices))],
                    'extra_info': {'story_info': q['intermediate_data'][0]['victim_info'], 'tree_trace': self.get_example_tree_trace(q, depth=3, replace_if_missing=True), 'fact_lookup': self.get_example_fact_lookup(q)} if 'murder' in q['question'] else {}, 'full_trees': self.get_full_trees(q) if 'murder' in q['question'] and self.give_full_trees else {}
                }
                examples.append(example)

        return examples

    def build_mm_user_context(self, choices, context, question, use_cot=True, few_shot=False):
        if few_shot:
            if use_cot:
                return f"""
{murder_mystery_solved_ex}

{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Use the example above as a guide. Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
                """
            else:
                return f"""
{murder_mystery_solved_ex_cotless}

{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Use the example above as a guide. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Please state your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.      
                """
        else:
            if use_cot:
                return f"""
{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
                """.strip()
            else:
                return f"""
{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Please state your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.      
                            """
    def build_ta_user_context(self, choices, context, question, use_cot=True, few_shot=False):
        if few_shot:
            if use_cot:
                return f"""
{team_allocation_solved_ex}

{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Use the example above as a guide. Before selecting a choice, explain your reasoning step by step. The story should allow you to determine how good each person is at a skill. Roughly, each person is either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks that uses their skills as well as possible. In addition, one task will have to have two people assigned to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the overall quality of the assignment.

When two people need to work on a task and one is bad at it, they don’t necessarily benefit from the other person being good, unless they work well together.

With different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team to find the single assignment to ensure that the tasks overall are completed as effectively as possible. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"             
                """
            else:
                return f"""
{team_allocation_solved_ex_cotless}

{team_allocation_solved_ex_2_cotless}

{team_allocation_solved_ex_3_cotless}

{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Use the example above as a guide. The story should allow you to determine how good each person is at a skill. Roughly, each person is either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks that uses their skills as well as possible. In addition, one task will have to have two people assigned to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the overall quality of the assignment.

When two people need to work on a task and one is bad at it, they don’t necessarily benefit from the other person being good, unless they work well together.

With different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team to find the single assignment to ensure that the tasks overall are completed as effectively as possible.

Please generate your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.
                                """
        else:
            if not use_cot:
                return f"""
{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. The story should allow you to determine how good each person is at a skill. Roughly, each person is either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks that uses their skills as well as possible. In addition, one task will have to have two people assigned to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the overall quality of the assignment.

When two people need to work on a task and one is bad at it, they don’t necessarily benefit from the other person being good, unless they work well together.

With different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team to find the single assignment to ensure that the tasks overall are completed as effectively as possible. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"

Please generate your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.

            """.strip()
            else:
                return f"""
{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Before selecting a choice, explain your reasoning step by step. The story should allow you to determine how good each person is at a skill. Roughly, each person is either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks that uses their skills as well as possible. In addition, one task will have to have two people assigned to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the overall quality of the assignment.

When two people need to work on a task and one is bad at it, they don’t necessarily benefit from the other person being good, unless they work well together.

With different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team to find the single assignment to ensure that the tasks overall are completed as effectively as possible. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
                            """.strip()

    def build_op_user_context(self, choices, context, question, use_cot=True, few_shot=False):
        if few_shot:
            if use_cot:
                return f"""
{object_placements_solved_ex}
                
{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Use the example above as a guide. Before selecting a choice, explain your reasoning step by step. Based on this story, we want to identify where someone believes that a certain object is at the end of the story. In order to do that, you need to read the story and keep track of where they think the object is at each point. When an object is moved, the person may observe its new location if they saw it move.

To see where an object ends up, they must be able to see the location that it moves to and not be too distracted by what they are doing. If they do not observe the object moving, then they will still believe it to be in the last location where they observed it. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
                """.strip()
            else:
                return f"""
{object_placements_solved_ex_cotless}

{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Use the example above as a guide. Please generate your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.
""".strip()
        else:
            if use_cot:
                return f"""
{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Before selecting a choice, explain your reasoning step by step. Based on this story, we want to identify where someone believes that a certain object is at the end of the story. In order to do that, you need to read the story and keep track of where they think the object is at each point. When an object is moved, the person may observe its new location if they saw it move.

To see where an object ends up, they must be able to see the location that it moves to and not be too distracted by what they are doing. If they do not observe the object moving, then they will still believe it to be in the last location where they observed it. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
                            """.strip()
            else:
                return f"""
{context.strip()}

{question.strip()}

{choices.strip()}

You must pick one option. Please generate your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.
            """.strip()

    def get_full_trees(self, ex):
        t1 = LogicTree.from_json(ex['intermediate_trees'][0])
        t2 = LogicTree.from_json(ex['intermediate_trees'][1])
        return [t1, t2]

    def get_example_tree_trace(self, ex, depth=1, replace_if_missing: bool = False):
        get_gold_trace = lambda x, depth=1: x.print_for_gpt(ignore_value_before_depth=1, ignore_value_after_depth=depth,
                                                            print_only_nodes_with_value=True, print_forward=True,
                                                            only_print_values=True)

        t1 = LogicTree.from_json(ex['intermediate_trees'][0])
        t2 = LogicTree.from_json(ex['intermediate_trees'][1])

        if replace_if_missing:
            for t in [t1, t2]:
                new_children = []
                branches = t.nodes[0].children
                vals = [x.value.lower() for x in branches]
                for c in branches:
                    if 'suspicious' not in c.value.lower():
                        new_children.append(c)
                if all(['means' not in v.lower() for v in vals]):
                    prev = deepcopy(new_children[-1])
                    prev.children = []
                    prev.value = prev.value.split("has")[0].strip() + " does not have a means."
                    new_children.append(prev)
                elif all(['opportunity' not in v.lower() for v in vals]):
                    prev = deepcopy(new_children[-1])
                    prev.children = []
                    prev.value = prev.value.split("has")[0].strip() + " does not have an opportunity."
                    new_children.append(prev)
                elif all(['motive' not in v.lower() for v in vals]):
                    prev = deepcopy(new_children[-1])
                    prev.children = []
                    prev.value = prev.value.split("has")[0].strip() + " does not have a motive."
                    new_children.append(prev)
                t.nodes[0].children = new_children

        t1_trace = get_gold_trace(t1, depth).strip()
        t2_trace = get_gold_trace(t2, depth).strip()

        return f'{t1_trace}\n\n{t2_trace}'

    def get_example_fact_lookup(self, ex):
        t1 = LogicTree.from_json(ex['intermediate_trees'][0])
        t2 = LogicTree.from_json(ex['intermediate_trees'][1])

        choices = ex['choices']

        facts = {
            choices[0]: {'means': False, 'motive': False, 'opportunity': False},
            choices[1]: {'means': False, 'motive': False, 'opportunity': False},
        }

        if any(['means' in x.value.lower() for x in t1.nodes[0].children]):
            facts[choices[0]]['means'] = True
        if any(['motive' in x.value.lower() for x in t1.nodes[0].children]):
            facts[choices[0]]['motive'] = True
        if any(['opportunity' in x.value.lower() for x in t1.nodes[0].children]):
            facts[choices[0]]['opportunity'] = True

        if any(['means' in x.value.lower() for x in t2.nodes[0].children]):
            facts[choices[1]]['means'] = True
        if any(['motive' in x.value.lower() for x in t2.nodes[0].children]):
            facts[choices[1]]['motive'] = True
        if any(['opportunity' in x.value.lower() for x in t2.nodes[0].children]):
            facts[choices[1]]['opportunity'] = True


        return facts

    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            *args,
            **kwargs
    ):
        return CSQADataset.evaluate_response(model_responses, example, *args, **kwargs)

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
            if 'answer:' in line.lower():
                potential_answer_segments.append(line)
            if 'the optimal assignment is ' in line:
                potential_answer_segments.append(line.split('the optimal assignment is ')[1].split(' ')[0])
            if 'the best option is ' in line:
                potential_answer_segments.append(line.split('the best option is ')[1].split(' ')[0])
            if 'the optimal assignment would be Option ' in line:
                potential_answer_segments.append(line.split('the optimal assignment would be Option ')[1].split(' ')[0])

            for sent in line.split('.'):
                if 'the most likely murderer is' in sent.lower():
                    potential_answer_segments.append(sent)
                elif all([x in sent.lower() for x in "i think the murderer is".split(' ')]):
                    potential_answer_segments.append(sent)
                elif 'most likely murderer' in sent.lower():
                    potential_answer_segments.append(sent)

            curr_tok_idx = sum([len(x) + 1 for x in parsed_resp.split('\n')[:-1]])
            potential_answer_segments = [[x, curr_tok_idx] for x in potential_answer_segments]

            curr_tok_idx = 0
            lines = []
            for l in resp.split('\n'):
                if l.strip() == '':
                    curr_tok_idx += len(l) + 1
                    continue
                else:
                    lines.append([l, curr_tok_idx])
                    curr_tok_idx += len(l) + 1

            for lidx, (l, tok_idx) in enumerate(lines):
                ending_seqs = [
                    'i think the best allocation would be:',
                    'the best assignment would be:',
                    'the optimal assignment would be:',
                    'the optimal assignment of roles would be:',
                    'the optimal assignment is:',
                ]
                if any([l.strip().lower().endswith(x.lower()) for x in ending_seqs]) and lidx < len(lines)-1:
                    potential_answer_segments.append((lines[lidx+1][0], tok_idx))
                for s in l.split('.'):
                    if "i would choose option" in s.lower() or "i would choose answer" in s.lower():
                        potential_answer_segments.append((s, tok_idx))
                curr_tok_idx += len(l) + 1

            for (line, tok_idx) in potential_answer_segments:
                try:
                    correct = any([x.lower() in line.lower() for x in answers])
                except Exception:
                    print('hi')

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
        return [returned_answers[0]]


if __name__ == '__main__':
    mm_path = '../../../datasets/murder_mystery.json'
    mm = MuSRDataset(mm_path)

    ta_path = '../../../datasets/team_allocation.json'
    ta = MuSRDataset(ta_path)

    op_path = '../../../datasets/object_placements.json'
    op = MuSRDataset(op_path)

    mm_ex = mm[2]
    ta_ex = ta[0]
    op_ex = op[0]

    responses = ["""1. Richard carries the flight manual to his office. (A: cockpit)
2. Tom follows Richard with the same purpose. (A: cockpit)
3. Lisa is seen with a bundle of safety booklets. (C: passenger seating area)
4. Tom navigates the plane, making his move amid the quiet of lesser trodden areas of the aircraft. (D: storage)
5. Lisa restocks the passenger seating area with safety booklets. (C: passenger seating area)
6. Richard is seen engrossed in pre-flight checks located in another section of the plane. (A: cockpit)
7. Tom and Richard discuss painstaking flight procedures. (B: office)

Based on the story, the most likely place Lisa would look to find the flight manual is in the cockpit (A: cockpit). This is because Richard and Tom are both seen going to the cockpit to access the flight manual, and Lisa is seen restocking the passenger seating area with safety booklets, which suggests that she may have previously checked the cockpit for the flight manual. Additionally, the story highlights the importance of the flight manual and safety booklets being easily accessible, which further supports the idea that Lisa would look for the flight manual in the cockpit.
"""
    ]

    metrics = op.evaluate_response(responses, op_ex)
    print([x['model_answer'] for x in metrics])
