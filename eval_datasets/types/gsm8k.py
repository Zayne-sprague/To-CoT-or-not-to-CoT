import random
import re
import pprint
import sympy as sp
import multiprocessing
from datasets import load_dataset

from eval_datasets import ReasoningDataset
from eval_datasets.sat_solver.arithmetic_solver import arithmetic_satlm_exec, arithmetic_proglm_exec
from sympy.parsing.latex import parse_latex
from sympy import simplify
import json
from pathlib import Path
from src.utils.paths import ROOT_FOLDER


class GSM8KDataset(ReasoningDataset):
    def __init__(self, path_or_url='openai/gsm8k', variant='original',subset='main', used_closed_source_prompting: bool = False, used_plan_solve_prompt: bool = False,used_cot_solver_prompt: bool =False, use_llama_3_1_prompts: bool = True, reload_llama_3_1_prompts: bool = False, llama_3_1_prompts_cache_file: str = ROOT_FOLDER / 'eval_datasets/third_party/llama_3_1_prompts_tmp_folder/gsm8k.json', *args, **kwargs):
        self.llama_3_1_prompts_cache_file = llama_3_1_prompts_cache_file
        self.use_llama_3_1_prompts = use_llama_3_1_prompts
        self.reload_llama_3_1_prompts = reload_llama_3_1_prompts
        self.used_plan_solve_prompt = used_plan_solve_prompt
        self.used_cot_solver_prompt = used_cot_solver_prompt
        self.used_closed_source_prompting = used_closed_source_prompting
        llama_3_1_prompts_cache_file.parent.mkdir(parents=True, exist_ok=True)

        self.variant = variant
        if variant == 'original':
            path_or_url = 'openai/gsm8k'
            subset = 'main'
            split = 'test'
        elif variant == 'plus':
            path_or_url = 'qintongli/GSM-Plus'
            subset = 'default'
            split = 'test'
        elif variant == 'hard':
            path_or_url = 'reasoning-machines/gsm-hard'
            subset = 'default'
            split = 'train'
        super().__init__(path_or_url + ':' + subset + ':' + split, *args, **kwargs)
        
        random.seed(0)

    average_token_len = 1000

    @classmethod
    @property
    def is_math(cls):
        return True
    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.gsm8k

    def load_llama_3_1_prompts(self, examples):
        # if self.variant != 'original':
        #     return examples

        if self.llama_3_1_prompts_cache_file.is_file() and not self.reload_llama_3_1_prompts:
            llama_prompts_per_task = json.load(open(self.llama_3_1_prompts_cache_file, 'r'))
        else:
            llama_prompts_per_task = []
            raw_prompt_dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
                                              "Meta-Llama-3.1-8B-Instruct-evals__gsm8k__details")["latest"]
            for x in raw_prompt_dataset:
                llama_prompts_per_task.append(
                    {'prompt': x['input_final_prompts'], 'question': x['input_question'],
                     'eval_config': x['eval_config'], 'input_correct_responses': x['input_correct_responses']})

        for exidx, ex in enumerate(examples):
            question = ex['question']

            if self.use_llama_3_1_prompts:
                found = False
                words = question.split()
                llama_3_1_prompt = None
                found_row = None
                for prompt in llama_prompts_per_task:
                    if prompt['question'] == question:
                        found = True
                        llama_3_1_prompt = prompt['prompt'][0]
                        found_row = prompt
                # if not found:
                #     for prompt in llama_prompts_per_task:
                #         if sum([x in prompt['question'] for x in words]) / len(words) > 0.9:
                #             found = True
                #             llama_3_1_prompt = prompt['prompt'][0]
                if found:
                    llama_3_1_msgs = self.convert_raw_to_messages(llama_3_1_prompt)
                else:
                    ### Use few-shot prompt from the first example in the llama3.1 eval dataset
                    llama_3_1_prompt = llama_prompts_per_task[0]
                    orig_question = llama_3_1_prompt['question']
                    assert orig_question in llama_3_1_prompt['prompt'][0]
                    llama_3_1_prompt = llama_3_1_prompt['prompt'][0].replace(orig_question,question)
                    llama_3_1_msgs = self.convert_raw_to_messages(llama_3_1_prompt)

                if self.used_closed_source_prompting:
                    custom_sys_message_for_direct = [{'role': 'system', 'content': 'You answer questions and only give the answer. You will always return "The final answer is $\\boxed{[answer_choice]}$." as your response where you fill in the answer choice with the correct answer.'}]
                    custom_sys_message_for_cot = [{'role': 'system', 'content': 'You answer questions and give explanations as well as calculations before giving the answer. You will always return "The final answer is $\\boxed{[answer_choice]}$." at the end of your response where you fill in the answer choice with the correct answer.'}]

                    few_shot_cot_prompt = custom_sys_message_for_cot + llama_3_1_msgs
                    few_shot_direct_prompt = custom_sys_message_for_direct + self.build_few_shot_direct_prompt(llama_3_1_msgs)
                    few_shot_direct_prompt[-1]['content'] += '\n\nYou may only give the answer. Begin your response with "The final answer is $\\boxed{".'
                    zero_shot_cot_prompt = custom_sys_message_for_cot + [llama_3_1_msgs[-1]]
                    zero_shot_direct_prompt = custom_sys_message_for_direct + [few_shot_direct_prompt[-1]]
                    zero_shot_direct_prompt[-1]['content'] += '\n\nYou may only give the answer. Begin your response with "The final answer is $\\boxed{".'
                else:
                    few_shot_cot_prompt = llama_3_1_msgs
                    few_shot_direct_prompt = [
                        {'role': 'system', 'content': 'You solve math questions. You only output "The final answer is $\\boxed{answer}$" where $\\boxed{answer}$ is the numerical answer to the problem'},
                        *self.build_few_shot_direct_prompt(llama_3_1_msgs)
                    ]

                    zero_shot_cot_prompt = [llama_3_1_msgs[-1]]
                    zero_shot_direct_prompt = [{'role': 'system', 'content': 'You solve math questions. You only output "The final answer is $\\boxed{answer}$" where $\\boxed{answer}$ is the numerical answer to the problem'}, few_shot_direct_prompt[-1]]
                if found:
                    answer_aliases =  found_row['input_correct_responses']
                else:
                    answer_aliases = [examples[exidx]['target']]
                examples[exidx]['llama_3_1_eval'] = {'prompts': {
                    'fs_cot': few_shot_cot_prompt,
                    'fs_direct': few_shot_direct_prompt,
                    'zs_cot': zero_shot_cot_prompt,
                    'zs_direct': zero_shot_direct_prompt
                }, 'eval_config': {"max_gen_len": "1024", "max_prompt_len": "3072", "num_few_shot": "8",
                                    "num_generations": "1", "temperature": "0.0", "top_k": "0", "top_p": "0"},
                'answer_aliases': answer_aliases}

                


        return examples

    def build_few_shot_direct_prompt(self, llama_3_1_messages):
        messages = []
        for message in llama_3_1_messages:
            if message['role'] == 'user':
                messages.append({
                    'role': 'user',
                    'content': message['content'].replace("Given the following problem, reason and give a final answer to the problem.", "Give the answer to the following problem.").replace("\n\nLet's think step by step.", "").replace(
                        "Your response should end with", "You should respond with").replace("[answer]", "$\\boxed{answer}$").replace("The final answer is", "The answer is").replace("is the final numerical answer to the problem.", "is the numerical answer to the problem.")
                })
            else:
                messages.append({
                    'role': 'assistant',
                    'content': "The answer is $" + message['content'].split("The final answer is ")[-1] + '$'
                })
        return messages

    def convert_raw_to_messages(self, raw):
        messages = []
        user_messages = [x for x in raw.split("<|start_header_id|>user<|end_header_id|>\n\n") if x != '']
        for user_message in user_messages:
            msgs = user_message.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')

            if len(msgs) == 2:

                messages.append({'role': 'user', 'content': msgs[0].replace("<|eot_id|>", "").replace(
                    "<|start_header_id|>user<|end_header_id|>", "").replace("[answer]", "$\\boxed{answer}$").replace("is the response to the", "is the final numerical answer to the")})
                if msgs[1] != '':
                    parts = msgs[1].split("The final answer is ")
                    new_msg = parts[0] + "The final answer is \\boxed{" + parts[1] + "}"
                    messages.append({'role': 'assistant', 'content': new_msg.replace("<|eot_id|>", "")})
            else:
                messages.append({'role': 'user', 'content': user_message.replace("<|eot_id|>", "").replace(
                    "<|start_header_id|>user<|end_header_id|>", "").replace("[answer]", "$\\boxed{answer}$").replace("is the response to the", "is the final numerical answer to the")})
        return messages


    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, subset, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url, subset)[split]]
        for ex in dataset:
            if self.variant in ['original', 'plus']:
                prompt = ex["question"]
                ### Format of GSM8K dataset
                answer = ex["answer"].split("#### ")[-1]
            elif self.variant == 'hard':
                prompt = ex["input"]
                answer = ex["target"]
            cot_sys_prompt = 'You are a helpful AI assistant that will answer reasoning questions. You will reason step by step and you will always say at the end $\\boxed{your answer}$". You must end your response with "\\boxed{your answer}" everytime!'
            cotless_sys_prompt = 'You are a helpful AI assistant that will answer reasoning questions. You will only say "\\boxed{your answer}". You must end your response with $\\boxed{your answer}$ everytime!'


            zs_cot_prompt = f"""Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
{prompt}
            """.strip() + """\n\nRemember to box your final answer via $\\boxed{your answer}$.
            """.strip()

            zs_cotless_prompt = f"""Solve the following math problem. Box your final answer: $\\boxed{{your answer}}$.

Problem:
{prompt}
            """.strip() + """\n\nRemember to box your final answer via $\\boxed{your answer}$.
            """

            fs_cot_prompt = f"""
{prompt_examples()}

Problem:
{prompt}
            """.strip()
            fs_cotless_prompt = f"""
{prompt_examples_cotless()}

Problem:
{prompt}
            """.strip()

            if self.used_plan_solve_prompt:
                ### Use prompt No.3 from Table 8 of Plan-and-solve paper
                zs_cot_prompt = f"""Let\'s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan.Then, let\'s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer. When you show the answer, box your final answer: $\\boxed{{your answer}}$.
            
            
Problem:
{prompt}
            """.strip() + """\n\nRemember to box your final answer via $\\boxed{your answer}$.
            """.strip()
            if self.used_cot_solver_prompt:
                zs_cot_prompt = f"""
{prompt_examples_cot_solver_chat()}

Q: {prompt}
            """.strip()

            examples.append({
                **ex,
                'answer': answer,
                "question": prompt,
                'dataset_type': self.dataset_types.gsm8k,
                'prompt_parts': {
                    'zs_cot_prompt': zs_cot_prompt,
                    'zs_cotless_prompt': zs_cotless_prompt,
                    'fs_cot_prompt': fs_cot_prompt,
                    'fs_cotless_prompt': fs_cotless_prompt,
                    'cot_system_prompt': cot_sys_prompt,
                    'cotless_system_prompt': cotless_sys_prompt
                },
                'answerKey': answer,
            })

        r = random.Random(1)
        r.shuffle(examples)
        if self.use_llama_3_1_prompts:
            examples = self.load_llama_3_1_prompts(examples)
        return examples


    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            answer_aliases: list[str] = (),
            *args, **kwargs
    ):
        answer = example['answer']

        returned_answers = []

        for resp in model_responses:
            slice = None
            try:
                out, slice = last_boxed_only_string(resp)
                ans = remove_boxed(out)

                ans = ans.replace("Your answer: ", "")
                ans = ans.replace("\$", "")
                ans = ans.replace("$", "")

                slice = [slice[0] + out.index(ans), slice[0] + out.index(ans) + len(ans)]

                correct = is_equiv(answer, ans)

                if not correct and len(answer_aliases) > 0:
                    if any([str(x).lower() in ans for x in answer_aliases]):
                        correct = True

                # print('hey0')
            except Exception as e:
                try:
                    resp = resp.replace('$\\boxed{', '').replace('$', '').lower().replace("the answer is ", '').replace('the final answer is ', '').replace(',', '')
                    resp = resp.split(' ')[0]
                    if '.' in resp:
                        ans = float(resp)
                    else:
                        ans = int(resp)
                    correct = is_equiv(answer, ans)

                except Exception as e:
                    ans = None
                    correct = False
            returned_answers.append({
                'model_response': resp,
                'answer_line': ans,
                'correct': correct,
                'answer_randomly_sampled': False,
                'answer_span': slice,
                'model_answer': ans,
                'raw_model_answer': ans,
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
        return None

def ans_parser(final_answer: str) -> str:
    SUBSTITUTIONS = [
        ('an', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''),
        (' ', ''), ('mbox', 'text'), (',\\text{and}', ''),
        ('\\text{and}', ','), ('\\text{m}', '\\text{}')
    ]

    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
        'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
        'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
        '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
        '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
        r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots',
    ]

    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('/')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)

    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \sqrtab -> \sqrt{a}{b}
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def simplify_with_timeout(equation, timeout=5):
    def solve_equation(equation, return_dict):
        try:
            solution = sp.simplify(equation)
            return_dict['solution'] = solution
        except Exception as e:
            return_dict['error'] = str(e)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(target=solve_equation, args=(equation, return_dict))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Sympy solve function timed out")

    if 'error' in return_dict:
        raise Exception(return_dict['error'])

    return return_dict.get('solution', None)
def prompt_examples():
    return """
Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
There are 15 trees in the grove. Grove workers will plant trees in the
      grove today. After they are done, there will be 21 trees. How many trees did
      the grove workers plant today?


Remember to box your final answer via $\\boxed{your answer}$.

Solution:
There are 15 trees originally. Then there were 21 trees after some more
      were planted. So there must have been 21 - 15 = 6. The answer is $\\boxed{6}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
If there are 3 cars in the parking lot and 2 more cars arrive, how many
      cars are in the parking lot?

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer
      is $\\boxed{{5}}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
Leah had 32 chocolates and her sister had 42. If they ate 35, how many
      pieces do they have left in total?

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
Originally, Leah had 32 chocolates. Her sister had 42. So in total they
      had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is $\\boxed{39}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12
      lollipops. How many lollipops did Jason give to Denny?

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
Jason started with 20 lollipops. Then he had 12 after giving some to Denny.
      So he gave Denny 20 - 12 = 8. The answer is $\\boxed{8}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: hawn has five toys. For Christmas, he got two toys each from his mom and
      dad. How many toys does he have now?

Solution: Shawn started with 5 toys. If he got 2 toys each from his mom and dad,
      then that is 4 more toys. 5 + 4 = 9. The answer is $\\boxed{{9}}$..

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: There were nine computers in the server room. Five more computers were
      installed each day, from monday to thursday. How many computers are now in the
      server room?

Solution: There were originally 9 computers. For each of 4 days, 5 more computers
      were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is
      $\\boxed{{29}}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday,
      he lost 2 more. How many golf balls did he have at the end of wednesday?

Solution: Michael started with 58 golf balls. After losing 23 on tuesday, he had
      58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer
      is $\\boxed{{33}}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: Olivia has $23. She bought five bagels for $3 each. How much money does
      she have left?

Solution: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15
      dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is $\\boxed{{8}}$.
""".strip()

def prompt_examples_cotless():
    return """SSolve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
There are 15 trees in the grove. Grove workers will plant trees in the
      grove today. After they are done, there will be 21 trees. How many trees did
      the grove workers plant today?


Remember to box your final answer via $\\boxed{your answer}$.

Solution:
The answer is $\\boxed{6}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
If there are 3 cars in the parking lot and 2 more cars arrive, how many
      cars are in the parking lot?

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
The answer is $\\boxed{{5}}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
Leah had 32 chocolates and her sister had 42. If they ate 35, how many
      pieces do they have left in total?

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
The answer is $\\boxed{39}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12
      lollipops. How many lollipops did Jason give to Denny?

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
The answer is $\\boxed{8}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: hawn has five toys. For Christmas, he got two toys each from his mom and
      dad. How many toys does he have now?

Solution: The answer is $\\boxed{{9}}$..

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: There were nine computers in the server room. Five more computers were
      installed each day, from monday to thursday. How many computers are now in the
      server room?

Solution: The answer is $\\boxed{{29}}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday,
      he lost 2 more. How many golf balls did he have at the end of wednesday?

Solution: The answer is $\\boxed{{33}}$.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem: Olivia has $23. She bought five bagels for $3 each. How much money does
      she have left?

Solution: The answer is $\\boxed{{8}}$.
""".strip()

def prompt_examples_satlm():
    return '''Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

# solution in code:
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result




Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

# solution in code:
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result




Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

# solution in code:
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    # 4 days between monday and thursday
    num_days = 4
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result




Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

# solution in code:
def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result




Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

# solution in code:
def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    lollipops_given = Variable()
    jason_lollipops_after = 12
    jason_lollipops_after = jason_lollipops_initial - lollipops_given
    result = lollipops_given
    return result




Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

# solution in code:
def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result




Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

# solution in code:
def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result




Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

# solution in code:
def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_planted = Variable()
    trees_after = 21
    trees_after = trees_initial + trees_planted
    result = trees_planted
    return result
'''.strip()


def prompt_examples_proglm():
    return '''Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

# solution in Python:
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result




Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

# solution in Python:
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result




Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

# solution in Python:
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result




Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

# solution in Python:
def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result




Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

# solution in Python:
def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result




Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

# solution in Python:
def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result




Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

# solution in Python:
def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result




Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

# solution in Python:
def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result
'''.strip()


def prompt_examples_proglm_chat():
    return '''
Let's use python to solve the following math problem. Please follow the python code format and define every variable clearly before using it. Please include proper indentations in your response. Do not generate any explanations for your python code. Here are three examples on how to do it.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# solution in Python:
def solution():
    # Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    # The money spent is the number of bagels times the cost of each bagel
    money_spent = bagels * bagel_cost
    # The money left is the initial money minus the money spent
    money_left = money_initial - money_spent
    result = money_left
    return result


Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# solution in Python:
def solution():
    # Michael had 58 golf balls
    golf_balls_initial = 58
    # On tuesday, he lost 23 golf balls. 
    golf_balls_lost_tuesday = 23
    # On wednesday, he lost 2 more. 
    golf_balls_lost_wednesday = 2
    # The number of golf balls left is the initial number of golf balls minus the number of golf balls lost on tuesday and wednesday
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result


Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# solution in Python:
def solution():
    # There were nine computers in the server room. How many computers are now in the server room?
    computers_initial = 9
    # Five more computers were installed each day
    computers_per_day = 5
    # 4 days between monday and thursday
    num_days = 4
    # The number of installed computers is the number of computers installed per day times the number of days
    computers_added = computers_per_day * num_days
    # The total number of computers is the initial number of computers plus the number of installed computers
    computers_total = computers_initial + computers_added
    result = computers_total
    return result


How about this question? Let's use python to solve the following math problem. Please follow the python code format and define every variable clearly before using it. Please include proper indentations in your response. Do not generate any explanations for your python code.
Q: {question}
'''.strip()


def prompt_examples_cot_solver_chat():
    return '''
Let's use python to solve the following math problem. Please follow the python code format and define every variable clearly before using it. Please include proper indentations in your response. Do not generate any explanations for your python code. Here are three examples on how to do it.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# solution in Python:
def solution():
    # Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    # The money spent is the number of bagels times the cost of each bagel
    money_spent = bagels * bagel_cost
    # The money left is the initial money minus the money spent
    money_left = money_initial - money_spent
    result = money_left
    # Solve the value of result
    # money_spent = bagels * bagel_cost = 5 * 3 = 15
    # money_left = money_initial - money_spent = 23 - 15 = 8
    # result = 8
    # The final answer is $\\boxed{8}$


Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# solution in Python:
def solution():
    # Michael had 58 golf balls
    golf_balls_initial = 58
    # On tuesday, he lost 23 golf balls. 
    golf_balls_lost_tuesday = 23
    # On wednesday, he lost 2 more. 
    golf_balls_lost_wednesday = 2
    # The number of golf balls left is the initial number of golf balls minus the number of golf balls lost on tuesday and wednesday
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    # Solve the value of result
    # golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday = 58 - 23 - 2 = 33
    # result = 33
    # The final answer is $\\boxed{33}$


Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# solution in Python:
def solution():
    # There were nine computers in the server room. How many computers are now in the server room?
    computers_initial = 9
    # Five more computers were installed each day
    computers_per_day = 5
    # 4 days between monday and thursday
    num_days = 4
    # The number of installed computers is the number of computers installed per day times the number of days
    computers_added = computers_per_day * num_days
    # The total number of computers is the initial number of computers plus the number of installed computers
    computers_total = computers_initial + computers_added
    result = computers_total
    # Solve the value of result
    # computers_added = computers_per_day * num_days = 5 * 4 = 20
    # computers_total = computers_initial + computers_added = 9 + 20 = 29
    # result = 29
    # The final answer is $\\boxed{29}$


How about this question? Let's use python to solve the following math problem. Please follow the examples below to write the python code in the proper format and define every variable clearly before using it. After writing the python code, please solve the value of the result and box the final answer in the comment. Remember to box your final answer via $\\boxed{your answer}$.
'''.strip()

def prompt_examples_proglm_chat_no_comment():
    return '''
Let's use python to solve the following math problem. Please follow the python code format and define every variable clearly before using it. Please include proper indentations in your response. Do not generate any explanations for your python code. Here are three examples on how to do it.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# solution in Python:
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result


Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# solution in Python:
def solution():
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result


Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# solution in Python:
def solution():
    computers_initial = 9
    computers_per_day = 5
    num_days = 4
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result


How about this question? Let's use python to solve the following math problem. Please follow the python code format and define every variable clearly before using it. Please include proper indentations in your response. Do not generate any explanations for your python code.
Q: {question}
'''.strip()

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = string.replace('amperes', '')

    # Replace \
    string = string.replace('\\','')

    # Replace commas
    string = string.replace(',', '')

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    
    try:
        ss1 = _strip_string(str(str1))
        ss2 = _strip_string(str(str2))
        if verbose:
            print(ss1, ss2)
        if ss1==ss2:
            return True
        else:
            return float(ss1) == float(ss2)
    except Exception as e:
        import traceback
        # print(e)
        # print(traceback.format_exc())
        # print(str1,str2)
        return str1 == str2


def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a, _ = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval, [idx, right_brace_idx+1]


def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None

    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break

    return tokens[:i]


def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)


def _clean_numbers(string):
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string


class GSM8KSymDataset(GSM8KDataset):
    # override to specify type of symbolic solver
    def __init__(self, path_or_url='openai/gsm8k', variant='original', subset='main', sym_solver='satlm', *args, **kwargs):
        assert sym_solver in ['satlm', 'proglm']
        self.sym_solver = sym_solver
        super().__init__(path_or_url, variant, subset, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.gsm8ksym

    # override to add satlm prompt
    def load_dataset(self, path_or_url):
        exs = super().load_dataset(path_or_url)
        for ex in exs:
            ex['dataset_type'] = self.dataset_types.gsm8ksym
            ex['sym_solver'] = self.sym_solver
            question = ex["question"]
            if self.sym_solver == 'satlm':
                ex['prompt_parts']['sat_system_prompt'] = 'You are a helpful AI assistant that will answer reasoning questions. You will write equations in the form of code to solve the problems.'
                fs_sat_prompt = prompt_examples_satlm() + "\n\n\n\n\n" + f"Q: {question}"
                ex['prompt_parts']['fs_sat_prompt'] = fs_sat_prompt
            elif self.sym_solver == 'proglm':
                # ex['prompt_parts']['sat_system_prompt'] = 'You are a helpful AI assistant that will answer reasoning questions. You will write Python code to solve the problems.'
                # fs_sat_prompt = prompt_examples_proglm() + "\n\n\n\n\n" + f"Q: {question}"
                # ex['prompt_parts']['sat_system_prompt'] = 'You will write python program to solve math problems. You will only write Python code.'
                ex['prompt_parts']['sat_system_prompt'] = 'You are a helpful AI assistant and you are good at writing Python codes. You will write python program to solve math problems. You will only write Python code.'
                fs_sat_prompt = prompt_examples_proglm_chat().format(question=question)
                # fs_sat_prompt = prompt_examples_proglm_chat_no_comment().format(question=question)
                # fs_sat_prompt = prompt_examples_proglm() + "\n\n\n\n\n" + f"Q: {question}"
                
                ex['prompt_parts']['fs_sat_prompt'] = fs_sat_prompt
        return exs

    # override to add satlm messages
    def process_example(self, example, *args, **kwargs):
        example = super().process_example(example, *args, **kwargs)
        def create_msgs(prompt, sys_prompt=None,multi_turn_few_shot=False):
            if prompt is None:
                return None
            if multi_turn_few_shot:
                icl_examples = prompt.split("\n\n\n")[:-1]
                question = prompt.split("\n\n\n")[-1]
                msg = []
                for i, icl_ex in enumerate(icl_examples):
                    icl_q = icl_ex.split("\n\n")[0]
                    icl_ans = icl_ex.split("\n\n")[1]
                    if sys_prompt:
                        msg.append({'role': 'system', 'content': sys_prompt})
                    msg.append({'role': 'user', 'content': icl_q})
                    msg.append({'role': 'assistant', 'content': icl_ans})
                if sys_prompt:
                    msg.append({'role': 'system', 'content': sys_prompt})
                msg.append({'role': 'user', 'content': question})
                return msg
            if sys_prompt:
                return [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': prompt}]
            return [{'role': 'user', 'content': prompt}]
        prompt_parts = example['prompt_parts']
        example['fs_sat_messages'] = create_msgs(prompt_parts.get('fs_sat_prompt'), prompt_parts.get('sat_system_prompt'),multi_turn_few_shot=False)
        print(example['fs_sat_messages'])
        return example

    # override to call symbolic solver
    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            *args, **kwargs
    ):
        answer = example['answer']

        returned_answers = []

        for resp in model_responses:
            if example['sym_solver'] == 'satlm':
                try:
                    ans = resp.split("def solution():")[1].strip()
                    ans = arithmetic_satlm_exec(ans)
                    ans = str(ans)
                    if ',' in ans:
                        ans = ans.replace(',', '')
                    correct = is_equiv(answer, ans)
                    print("Answer:", ans, "GT:", answer)
                except Exception as e:
                    ans = None
                    correct = False
                    print("SatLM Eval Exec:", e)
            elif example['sym_solver'] == 'proglm':
                try:
                    ans = resp[resp.index("def solution():"):].strip()
                    ans = ans.replace('```','')
                    ans = arithmetic_proglm_exec(ans)
                    ans = float(ans)
                    ans = str(ans)
                    correct = is_equiv(answer, ans)
                    print("Answer:", ans, "GT:", answer)
                except Exception as e:
                    ans = None

                    correct = False
                    import traceback
                    traceback.print_exc()
                    
                    print("ProgLM Eval Exec:", e)
                    print(resp)
            returned_answers.append({
                'model_response': resp,
                'answer_line': ans,
                'correct': correct,
                'answer_randomly_sampled': False,
                'model_answer': ans,
                **example
            })

        return returned_answers

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = GSM8KDataset()

    ex = dataset[0]


    responses = [
        """One two three \\boxed{1} there you go.

"""
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])

    print(ex['messages'][0]['content'])
