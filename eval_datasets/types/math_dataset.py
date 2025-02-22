"""
Broken as of now (unless you have a cached version) :P

Not sure if it will return to HF so a lot of editing will need to be done to fix this.
"""
import random
import re
import pprint
import sympy as sp
import multiprocessing
from datasets import load_dataset

from eval_datasets import ReasoningDataset
from sympy.parsing.latex import parse_latex
from sympy import simplify
import json
from pathlib import Path
from src.utils.paths import ROOT_FOLDER


class MATHDataset(ReasoningDataset):
    def __init__(self, path_or_url='lighteval/MATH', subset='all',  use_llama_3_1_prompts: bool = True, reload_llama_3_1_prompts: bool = False, llama_3_1_prompts_cache_file: str = ROOT_FOLDER / 'eval_datasets/third_party/llama_3_1_prompts_tmp_folder/math.json', *args, **kwargs):
        self.llama_3_1_prompts_cache_file = llama_3_1_prompts_cache_file
        self.use_llama_3_1_prompts = use_llama_3_1_prompts
        self.reload_llama_3_1_prompts = reload_llama_3_1_prompts
        llama_3_1_prompts_cache_file.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(path_or_url + ':' + subset + ':test', *args, **kwargs)
        random.seed(0)

    average_token_len = 1000

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.math
    @classmethod
    @property
    def is_math(cls):
        return True

    def load_llama_3_1_prompts(self, examples):
        for exidx, ex in enumerate(examples):
            few_shot_direct_prompt = self.build_fewshot_direct_prompt_for_llama_3_1_eval(ex)
            zero_shot_direct_prompt = self.build_direct_prompt_for_llama_3_1_eval(ex)

            few_shot_cot_prompt = self.build_fs_cot_prompt_from_llama_3_1_eval(ex)
            zero_shot_cot_prompt = self.build_cot_prompt_for_llama_3_1_eval(ex)



            examples[exidx]['llama_3_1_eval'] = {'prompts': {
                'fs_cot': few_shot_cot_prompt,
                'fs_direct': few_shot_direct_prompt,
                'zs_cot': zero_shot_cot_prompt,
                'zs_direct': zero_shot_direct_prompt
            }, 'eval_config': {"max_gen_len": "10", "max_prompt_len": "3840", "num_few_shot": "5", "num_generations": "1", "temperature": "0.0", "top_k": "0", "top_p": "0", "additional_info_from_zayne": "Why do they say 5-shot here but report 0 shot in the table (also no shots are in the prompt, maybe it's a typo hopefully in this config)"}}
        return examples

    def build_direct_prompt_for_llama_3_1_eval(self, example, solution = None):
        regular_instructions = """Solve the following math problem efficiently and clearly. Your response should end with:\n\nThe final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: """
        question = example['question']

        messages = [
            {'role': 'user', 'content': regular_instructions + question.strip()}
        ]
        if solution:
            messages.append({'role': 'assistant', 'content': solution})
        else:
            messages.append({'role': 'assistant', 'content': 'The final answer is: $\\boxed{'})

        return messages

    def build_fewshot_direct_prompt_for_llama_3_1_eval(self, example):
        questions = ["""Find the domain of the expression $\\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}""", """"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find
$\det (\mathbf{A} \mathbf{B}).$""",
"""Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound
weights instead, how many times must Terrell lift them in order to lift the
same total weight?""", """"If the system of equations

\\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\\frac{a}{b},$ assuming $b$ is nonzero."""
                     ]
        solutions = [
            """The final answer is $\\boxed{[2,5)}$. I hope it is correct.""",
            """The final answer is $\\boxed{24}$. I hope it is correct.""",
            """The final answer is $\\boxed{16}$. I hope it is correct.""",
            """The final answer is $\\boxed{-\\frac{2}{3}}$. I hope it is correct."""
        ]
        prompt = []
        for q, a in zip(questions, solutions):
            prompt.extend(self.build_direct_prompt_for_llama_3_1_eval({'question': q}, a))
        prompt.extend(self.build_direct_prompt_for_llama_3_1_eval(example))
        return prompt



    def build_cot_prompt_for_llama_3_1_eval(self, example, solution = None, use_regular_instructions=True):
        regular_instructions = """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: """
        few_shot_cot_instructions = """Solve the following math problem efficiently and clearly:\n\nProvide a concise solution with explanations and calculations.\n\nAlways conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: """
        question = example['question']

        messages = [
            {'role': 'user', 'content': (regular_instructions if use_regular_instructions else few_shot_cot_instructions) + question.strip()}
        ]
        if solution:
            messages.append({'role': 'assistant', 'content': solution})

        return messages

    def build_fs_cot_prompt_from_llama_3_1_eval(self, example):
        questions = ["""Find the domain of the expression $\\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}""", """"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find
$\det (\mathbf{A} \mathbf{B}).$""",
            """Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound
weights instead, how many times must Terrell lift them in order to lift the
same total weight?""", """"If the system of equations

\\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\\frac{a}{b},$ assuming $b$ is nonzero."""
                     ]
        solutions = ["""The expressions inside each square root must be non-negative. Therefore,
$x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator
cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of
the expression is $\\boxed{[2,5)}$.
Therefore, the final answer is $\\boxed{[2,5)}$. I hope it is correct.""",
            """We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B})
= (2)(12) = \\boxed{24}.$
Therefore, the final answer is $\\boxed{24}$. I hope it is correct.""",
            """If Terrell lifts two 20-pound weights 12 times, he lifts a total of
$2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound
weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$
pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\\boxed{16}
\end{align*}
Therefore, the final answer is $\\boxed{16}$. I hope it is correct.""",
            """If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have 49
$$-\\frac{3}{2}a=b\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Therefore, the final answer is $\\boxed{-\\frac{2}{3}}$. I hope it is correct."""
        ]

        prompt = []
        for q, a in zip(questions, solutions):
            prompt.extend(self.build_cot_prompt_for_llama_3_1_eval({'question': q}, a, use_regular_instructions=False))
        prompt.extend(self.build_cot_prompt_for_llama_3_1_eval(example, use_regular_instructions=False))
        return prompt


    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, subset, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url, subset)[split]]
        for ex in dataset:
            prompt = ex["problem"]

            out, _ = last_boxed_only_string(ex["solution"])
            answer = remove_boxed(out)

            cot_sys_prompt = 'You are a helpful AI assistant that will answer reasoning questions. You will reason step by step and you will always say at the end "\\boxed{your answer}". You must end your response with "\\boxed{your answer}" everytime!'
            cotless_sys_prompt = 'You are a helpful AI assistant that will answer reasoning questions. You will only say "\\boxed{your answer}". You must end your response with "\\boxed{your answer}" everytime!'


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

            examples.append({
                **ex,
                'answer': answer,
                'question': ex["problem"],
                'dataset_type': self.dataset_types.math,
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
        answer = example['answer']

        returned_answers = []

        for resp in model_responses:
            slice = None
            try:
                out, slice = last_boxed_only_string(resp)
                ans = remove_boxed(out)


                ans = ans.replace("Your answer: ", "")
                ans = ans.replace("$", "")

                slice = [slice[0] + out.index(ans), slice[0] + out.index(ans) + len(ans)]

                correct = is_equiv(answer, ans)
                # print('hey0')
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
Find the domain of the expression $\\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
The expressions inside each square root must be non-negative. Therefore,
$x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator
cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of
the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $\\boxed{[2,5)}$. I hope it is correct.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find
$\det (\mathbf{A} \mathbf{B}).$

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B})
= (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $\\boxed{24}$. I hope it is correct.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound
weights instead, how many times must Terrell lift them in order to lift the
same total weight?

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of
$2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound
weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$
pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\\boxed{16}
\end{align*}
Final Answer: The final answer is $\\boxed{16}$. I hope it is correct.

Solve the following math problem.  Explain your reasoning step by step.  When you are finished, box your final answer: $\\boxed{{your answer}}$.
            
Problem:
If the system of equations

\\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\\frac{a}{b},$ assuming $b$ is nonzero.

Remember to box your final answer via $\\boxed{your answer}$.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have 49
$$-\\frac{3}{2}a=b\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $\\boxed{-\\frac{2}{3}}$. I hope it is correct.
""".strip()

def prompt_examples_cotless():
    return """Solve the following math problem. Box your final answer: $\\boxed{{your answer}}$.
    
Problem:
Find the domain of the expression $\\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Remember to box your final answer via $\\boxed{your answer}$.

Final Answer: $\\boxed{[2,5)}$

Solve the following math problem. Box your final answer: $\\boxed{{your answer}}$.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find
$\det (\mathbf{A} \mathbf{B}).$

Remember to box your final answer via $\\boxed{your answer}$.

Final Answer: $\\boxed{24}$

Solve the following math problem. Box your final answer: $\\boxed{{your answer}}$.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound
weights instead, how many times must Terrell lift them in order to lift the
same total weight?

Remember to box your final answer via $\\boxed{your answer}$.

Final Answer: $\\boxed{16}$

Solve the following math problem. Box your final answer: $\\boxed{{your answer}}$.

Problem:
If the system of equations

\\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\\frac{a}{b},$ assuming $b$ is nonzero.

Remember to box your final answer via $\\boxed{your answer}$.

Final Answer: $\\boxed{-\\frac{2}{3}}$
""".strip()


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
        return s


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

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            idx = string.rfind('he final answer is:')
            if idx < 0:
                return None
            else:
                retval = string.split("he final answer is:")[-1].strip().replace('$$', '').split("I hope it is correct")[0].split("\n\n")[0].strip()
                if retval.startswith('$'):
                    retval = retval[1:-1]
                return retval, [string.index(retval), string.index(retval) + len(retval)]

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
    """
    Clean Numbers in the given string
    """

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

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = MATHDataset()

    ex = dataset[0]


    responses = [
        """One two three \\boxed{1} there you go.

"""
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])

    print(ex['messages'][0]['content'])
