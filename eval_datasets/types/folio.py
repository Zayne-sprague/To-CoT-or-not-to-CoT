import random
import os
from copy import deepcopy

from pathlib import Path
import json
from src.utils.paths import ROOT_FOLDER
from datasets import load_dataset
from eval_datasets import ReasoningDataset, CSQADataset
from functools import partial

from eval_datasets.sat_solver.fol_solver import fol_satlm_exec
from eval_datasets.sat_solver.fol_uninterp_solver import fol_uninterp_satlm_exec

random.seed(0)

# get the direcotry of the current file as a str
_FILE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

class FOLIODataset(ReasoningDataset):
    average_token_len = 2000

    def __init__(self, path_or_url=_FILE_PATH + '../thirdparty/folio/original.json', split='validation', subset='default', generating_paraphrase=False,symbolic=False,used_plan_solve_prompt: bool = False,used_cot_solver_prompt: bool =False,used_cot_solver_no_cot_prompt: bool=False,use_llama_3_1_prompts: bool = True,*args, **kwargs):
        self.generating_paraphrase = generating_paraphrase
        self.path_or_url = path_or_url
        self.used_plan_solve_prompt = used_plan_solve_prompt
        self.used_cot_solver_prompt = used_cot_solver_prompt
        self.used_cot_solver_no_cot_prompt = used_cot_solver_no_cot_prompt
        self.use_llama_3_1_prompts = use_llama_3_1_prompts
        self.symbolic = symbolic
        if not str(path_or_url).endswith('.json'):
            super().__init__(path_or_url + ':' + subset + ':' + split, generating_paraphrase = generating_paraphrase,*args, **kwargs)
        else:
            super().__init__(path_or_url, generating_paraphrase = generating_paraphrase,*args, **kwargs)
        

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.folio

    def load_dataset(self, path_or_url):
        if not str(path_or_url).endswith('.json'):
            dataset_url, subset, split = path_or_url.split(':')
            data = [x for x in load_dataset(dataset_url, subset)[split]]
        else:
            data = json.load(open(str(path_or_url), 'r'))
       
        examples = []

        for raw_ex in data:
            root_ex = {}
            if 'label' in raw_ex:
                answer = raw_ex['label']
            else:
                answer = raw_ex['answer']
            choices = {"label": ["A","B","C"], "text": ['True', 'False', "Unknown"]}

            root_ex['choices'] = choices
            root_ex['answer'] = answer if answer != 'Uncertain' else 'Unknown'
            root_ex['answer_index'] = 0 if answer == 'True' else (1 if answer == 'False' else 2)
            root_ex['answer_choice_tokens'] = ['A', 'B', 'C']
            root_ex['answerKey'] = 'A' if answer == 'True' else ('B' if answer == 'False' else 'C')
            root_ex["dataset_type"] = self.dataset_types.folio
            
            
            if not self.generating_paraphrase:
                question_temp = """Premises: {} \n
            Conclusion: {} \n
            Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown."""
                prompt_func = partial(self.basic_prompt,choices=self.format_choices(choices))
                
            else:
                question_temp = """Premises: {} \n\n
            Conclusion: {}"""
                prompt_func = partial(self.paraphrase_prompt,level='complex')
            if not self.symbolic:
                root_ex["question_format"] = question_temp.format(raw_ex["premises"],raw_ex["conclusion"])
            else:
                root_ex["question_format"] = question_temp.format(raw_ex["premises-FOL"],raw_ex["conclusion-FOL"])
            prompt = prompt_func(root_ex['question_format'])
            root_ex['prompt_parts'] = {'user_context': prompt}
            # root_ex['cot_prompt'] = "Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + root_ex["question_format"] + '\nA. True\nB. False\nC. Unknown\n\n' + "- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nGive a description of your reasoning before you answer.\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, or C.\n\nLet's think step by step."
            # root_ex['direct_prompt'] ="Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + root_ex["question_format"] + '\nA. True\nB. False\nC. Unknown\n\n' + "Your response should end with \"The best answer is [the_answer_letter].\" where the [the_answer_letter] is one of A, B, or C."
            root_ex['cot_prompt'] = self.basic_prompt(choices=self.format_choices(choices),direct=False,question = question_temp.format(raw_ex["premises"],raw_ex["conclusion"]))
            root_ex['direct_prompt'] = self.basic_prompt(choices=self.format_choices(choices),direct=True,question = question_temp.format(raw_ex["premises"],raw_ex["conclusion"]))
            if self.used_plan_solve_prompt:
                root_ex['cot_prompt'] = "Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + root_ex["question_format"] + '\nA. True\nB. False\nC. N/A\n\n' + "Let\'s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. You must explicitly list a plan and releant variables. Then, let\'s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer. When you are ready to answer write the answer in the format: \"Answer: <your answer>\". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."
                root_ex['prompt_parts'] = {'user_context': root_ex['cot_prompt']}
            if self.used_cot_solver_prompt:
                root_ex['cot_prompt'] = prompt_examples_satdev_v0_cot_solver().format(raw_ex["premises"],raw_ex["conclusion"])
                root_ex['prompt_parts'] = {'user_context': root_ex['cot_prompt']}
            if self.used_cot_solver_no_cot_prompt:
                root_ex['direct_prompt'] = prompt_examples_satdev_v0_cot_solver_no_cot().format(raw_ex["premises"],
                                                                                      raw_ex["conclusion"])
                root_ex['cot_prompt'] = prompt_examples_satdev_v0_cot_solver().format(raw_ex["premises"], raw_ex["conclusion"])
                root_ex['prompt_parts'] = {
                    'zs_cotless_prompt': root_ex['direct_prompt'],
                    'zs_cot_prompt': root_ex['cot_prompt']
                }

            root_ex['prompt_parts']['zs_cot_prompt'] = root_ex['cot_prompt']
            root_ex['prompt_parts']['zs_cotless_prompt'] = root_ex['direct_prompt']

            examples.append({
                **root_ex,
                'question': question_temp.format(raw_ex["premises"],raw_ex["conclusion"]),
                'premises': raw_ex['premises'],
                'conclusion': raw_ex['conclusion'],
            })

        random.shuffle(examples)
#         icl_exs = [
#         {'premises': """The Mona Lisa is a world's best-known painting.
# The Mona Lisa is a portrait painted by Leonardo da Vinci.
# Leonardo da Vinci was a scientist and painter.
# Painting genres can be history, portrait, animal, landscape, and still life.""",
#          'conclusion': """Leonardo da Vinci has artworks in the landscape genre.""",
#          'answerKey': "C",
#          'cot': """Based on the premises, Leonardo da Vinci was a scientist and painter. Based on the definition of painters, painters paints. Therefore, Leonardo da Vinci is likely to have paintings. Based on the premises, painting genres can be history, portrait, animal, landscape, and still life. Because Leonardo da Vinci is likely to have paintings, he is likely to have paintings in any of these genres that belong to the painting genre. Based on painting is a type of artwork, Leonardo da Vinci might have artworks in any of these genres that belong to the painting genre, including history, portrait, animal, landscape, and still life. However, there is no information that mentions that Leonardo da Vinci particularly has artworks in the landscape genre. Therefore, it is unknown that Leonardo da Vinci has artworks in the landscape genre. Thus, the conclusion is Unknown."""
# },
#          {'premises': """Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
# Any choral conductor is a musician.
# Some musicians love music.
# Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.""",
#          'conclusion': """A Czech published a book in 1946.""",
#          'answerKey': "A",
#          'cot': """Based on the premises, Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant, and Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Because Miroslav Venhoda published a book in 1946 and Miroslav Venhoda was a Czech, we can conclude that the statement that a Czech published a book in 1946 is True. Thus, the conclusion is True."""},
#          {'premises': """The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
# All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
# Airbus made more revenue than Boeing last year.""",
#          'conclusion': """There does not exist a United Airlines plane produced by Boeing.""",
#          'answerKey': "C",
#          'cot': """Based on the premises, all American Airlines planes are from the world's major large passenger aircraft manufacturers, and the world's only major large passenger aircraft manufacturers are Boeing and Airbus. Therefore, all American Airlines planes are from Boeing or Airbus. However, there is no information related to United Airlines planes in the premises, and there is no premise that shows the logical relationship between United Airlines and American Airlines. Therefore, we cannot deduce any information about United Airlines planes based on the premises related to American Airlines planes. So, it is unknown that there does not exist a United Airlines plane produced by Boeing. Thus, the conclusion is unknown."""},
#          {'premises': """Edwin Smith was a New Zealand rower from Auckland.
# Edwin Smith was also known as Ted Smith.
# Edwin Smith went to Rose Road Primary School, located in Grey Lynn.
# Edwin Smith was a sergeant who served with the New Zealand 24th battalion in Italy and Egypt.
# Broadway Sheetmetals was a business run and owned by Edwin Smith, a sheet metal worker.""",
#          'conclusion': """No sergeants were from Auckland.""",
#          'answerKey': "B",
#          'cot': """Based on the premises, Edwin Smith was a New Zealand rower from Auckland, and Edwin Smith was a sergeant who served with the New Zealand 24th battalion in Italy and Egypt. Thus, we know that Edwin Smith was from Auckland, New Zealand, and Edwin Smith was a sergeant. Therefore, there existed at least one sergeant from Auckland because Edwin Smith was a sergeant from Auckland. This contradicts the conclusion that no sergeants were from Auckland. Thus, the conclusion is False."""},
#     ]
        icl_exs = [
        {'premises': """The Mona Lisa is a world's best-known painting.
The Mona Lisa is a portrait painted by Leonardo da Vinci.
Leonardo da Vinci was a scientist and painter.
Painting genres can be history, portrait, animal, landscape, and still life.""",
         'conclusion': """Leonardo da Vinci has artworks in the landscape genre.""",
         'answerKey': "C",
         'cot': """Based on the premises, Leonardo da Vinci was a scientist and painter. Because painter paints, he is likely to have paintings.  Based on the premises, painting genres can be history, portrait, animal, landscape, and still life. Thus, Leonardo da Vinci might have artworks in any of these genres that belong to the painting genre. However, there is no information that mentions that Leonardo da Vinci has artworks in the landscape genre. Therefore, it is unknown that Leonardo da Vinci has artworks in the landscape genre."""
},
         {'premises': """Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.""",
         'conclusion': """A Czech published a book in 1946.""",
         'answerKey': "A",
         'cot': """Based on the premises, Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant, and Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Therefore, a Czech published a book in 1946. Thus, the conclusion is True."""},
         {'premises': """The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
Airbus made more revenue than Boeing last year.""",
         'conclusion': """There does not exist a United Airlines plane produced by Boeing.""",
         'answerKey': "C",
         'cot': """Based on the premises, all American Airlines planes are from the world's major large passenger aircraft manufacturers and the world's only major large passenger aircraft manufacturers are Boeing and Airbus. Therefore, all American Airlines planes are from Boeing or Airbus. However, there is no information related to United Airlines planes in the premises. So, it is unknown that there does not exist a United Airlines plane produced by Boeing."""},
         {'premises': """Edwin Smith was a New Zealand rower from Auckland.
Edwin Smith was also known as Ted Smith.
Edwin Smith went to Rose Road Primary School, located in Grey Lynn.
Edwin Smith was a sergeant who served with the New Zealand 24th battalion in Italy and Egypt.
Broadway Sheetmetals was a business run and owned by Edwin Smith, a sheet metal worker.""",
         'conclusion': """No sergeants were from Auckland.""",
         'answerKey': "B",
         'cot': """Based on the premises, Edwin Smith was a New Zealand rower from Auckland, and Edwin Smith was a sergeant who served with the New Zealand 24th battalion in Italy and Egypt. Thus, we know that Edwin Smith was a sergeant and was from Auckland. Therefore, there exists at least one sergeant from Auckland, who is Edwin Smith. This contradicts the conclusion. So the conclusion is False."""},
    ]
        for i, ex in enumerate(icl_exs):
            icl_exs[i]['cot_prompt'] = self.basic_prompt(choices=self.format_choices(choices),direct=False,question = question_temp.format(ex["premises"],ex["conclusion"]))
            icl_exs[i]['direct_prompt'] = self.basic_prompt(choices=self.format_choices(choices),direct=True,question = question_temp.format(ex["premises"],ex["conclusion"]))
        if self.use_llama_3_1_prompts:
            for example in examples:
                # prompts = build_icl(example,question_temp)
                prompts = build_icl(example,icl_exs)
                example['llama_3_1_eval'] = {'prompts': prompts}
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
    def custom_evaluate_response(self, model_responses, example, *args, **kwargs):
        return None

def build_icl(curr_ex,icl_exs):
    sys_prompt = "You are a helpful AI assistant that will answer reasoning questions. You will only say \"Answer: <Your Answer Letter Choice>\". You may only pick one answer and you must end your response with \"Answer: <Your Answer Letter Choice>\" everytime!"
    sys_prompt = {
            'role': 'system',
            'content': sys_prompt
        }
    cot_messages = []
    direct_messages = []
    cot_messages.append(sys_prompt)
    direct_messages.append(sys_prompt)
    for ex in icl_exs:
        
        cot_messages.append({
            'role': 'user',
            'content': ex['cot_prompt']
        })
        cot_messages.append({
            'role': 'assistant',
            'content': ex['cot'] + ' Answer: ' + ex['answerKey'] + '.'
        })

        direct_messages.append({
            'role': 'user',
            'content': ex['direct_prompt']
        })
        direct_messages.append({
            'role': 'assistant',
            'content': 'Answer: ' + ex['answerKey'] + '.'
        })

    cot_messages.append({
        'role': 'user',
        'content': curr_ex['cot_prompt']
    })

    direct_messages.append({
        'role': 'user',
        'content': curr_ex['direct_prompt']
    })

    return  {
        'zs_direct': [sys_prompt,direct_messages[-1], {'role': 'assistant', 'content': 'Answer: '}],
        'zs_cot': [sys_prompt,cot_messages[-1]],
        'fs_direct': direct_messages + [{'role': 'assistant', 'content': 'Answer: '}],
        'fs_cot': cot_messages
    }
def prompt_examples_satlm():
    return """
Problem: Based on the premises, is the conclusion true, false, or unknown?

Premises:
The Mona Lisa is a world's best-known painting.
The Mona Lisa is a portrait painted by Leonardo da Vinci.
Leonardo da Vinci was a scientist and painter.
Painting genres can be history, portrait, animal, landscape, and still life.

Conclusion:
Leonardo da Vinci has artworks in the landscape genre.

# solution in code:
def solution():
    # The Mona Lisa is a world's best-known painting.
    And(Painting(Mona_Lisa), Best_Known(Mona_Lisa))
    # The Mona Lisa is a portrait painted by Leonardo da Vinci.
    And(Portrait(Mona_Lisa), Painted_By(Mona_Lisa, Leonardo_da_Vinci))
    # Leonardo da Vinci was a scientist and painter.
    And(Scientist(Leonardo_da_Vinci), Painter(Leonardo_da_Vinci))
    # Painting genres can be history, portrait, animal, landscape, and still life.
    ForAll([x], Implies(Painting(x), Or(History(x), Portrait(x), Animal(x), Landscape(x), Still_Life(x))))

    # Leonardo da Vinci has artworks in the landscape genre.
    return Exists([x], And(Landscape(x), Painted_By(x, Leonardo_da_Vinci)))



Problem: Based on the premises, is the conclusion true, false, or unknown?

Premises:
Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.

Conclusion:
A Czech published a book in 1946.

# solution in code:
def solution():
    # Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    And(Czech(Miroslav_Venhoda), Choral_Conductor(Miroslav_Venhoda), Specialized_In(Miroslav_Venhoda, Renaissance_and_Baroque_Music))
    # Any choral conductor is a musician.
    ForAll([x], Implies(Choral_Conductor(x), Musician(x)))
    # Some musicians love music.
    Exists([x], And(Musician(x), Love(x, Music)))
    # Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Published_a_Book_in_1946(Miroslav_Venhoda)

    # A Czech published a book in 1946.
    return Exists([x], And(Czech(x), Published_a_Book_in_1946(x)))



Problem: Based on the premises, is the conclusion true, false, or unknown?

Premises:
The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
Airbus made more revenue than Boeing last year.

Conclusion:
There does not exist a United Airlines plane produced by Boeing.

# solution in code:
def solution():
    # The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
    ForAll([x], Implies(Major_Manufacturer(x), Or(Made_By(x, Boeing), Made_By(x, Airbus))))
    # All American Airlines planes are from the world's major large passenger aircraft manufacturers.
    ForAll([x], Implies(American_Airlines(x), Major_Manufacturer(x)))
    # Airbus made more revenue than Boeing last year.
    More_Revenue(Airbus, Boeing)

    # There does not exist a United Airlines plane produced by Boeing.
    return Not(Exists([x], And(United_Airlines(x), Made_By(x, Boeing))))
""".strip()


def prompt_examples_satdev_v0():
    return """
You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. 

Example 1
Premises:
The Mona Lisa is a world's best-known painting.
The Mona Lisa is a portrait painted by Leonardo da Vinci.
Leonardo da Vinci was a scientist and painter.
Painting genres can be history, portrait, animal, landscape, and still life.

Conclusion:
Leonardo da Vinci has artworks in the landscape genre.

# solution in code:
def solution():
    # The Mona Lisa is a world's best-known painting.
    And(Is_Painting(Mona_Lisa), Best_Known_Painting(Mona_Lisa))
    # The Mona Lisa is a portrait painted by Leonardo da Vinci.
    And(Is_Portrait(Mona_Lisa), Painted_By(Mona_Lisa, Leonardo_da_Vinci))
    # Leonardo da Vinci was a scientist and painter.
    And(Is_Scientist(Leonardo_da_Vinci), Is_Painter(Leonardo_da_Vinci))
    # Painting genres can be history, portrait, animal, landscape, and still life.
    ForAll([x], Implies(Is_Painting(x), Or(Is_History(x), Is_Portrait(x), Is_Animal(x), Is_Landscape(x), Is_Still_Life(x))))

    # Leonardo da Vinci has artworks in the landscape genre.
    return Exists([x], And(Is_Artwork(x), Is_Landscape(x), Painted_By(x, Leonardo_da_Vinci)))


Example 2
Premises:
Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.

Conclusion:
A Czech published a book in 1946.

# solution in code:
def solution():
    # Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    And(Czech(Miroslav_Venhoda), Choral_Conductor(Miroslav_Venhoda), Specialized_In(Miroslav_Venhoda, Renaissance_and_Baroque_Music))
    # Any choral conductor is a musician.
    ForAll([x], Implies(Choral_Conductor(x), Is_Musician(x)))
    # Some musicians love music.
    Exists([x], And(Is_Musician(x), Loves(x, Music)))
    # Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    And(Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda), Published_In(Method_of_Studying_Gregorian_Chant, 1946))

    # A Czech published a book in 1946.
    return Exists([x, y], And(Czech(x), Published_By(y, x), Published_In(y, 1946)))

Example 3
Premises:
The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
Airbus made more revenue than Boeing last year.

Conclusion:
There does not exist a United Airlines plane produced by Boeing.

# solution in code:
def solution():
    # The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
    ForAll([x], Implies(Is_Major_Manufacturer(x), Or(x == Boeing, x == Airbus)))
    # All American Airlines planes are from the world's major large passenger aircraft manufacturers.
    ForAll([x], Implies(Is_American_Airlines(x), Exists([y], And(Is_Major_Manufacturer(y), Made_By(x, y))))
    # Airbus made more revenue than Boeing last year.
    Make_More_Revenue(Airbus, Boeing)

    # There does not exist a United Airlines plane produced by Boeing.
    return Not(Exists([x], And(Is_United_Airlines(x), Made_By(x, Boeing))))

Example 4
Premises:
Some professional basketball players are not American nationals.
All professional basketball players can do jump shots.
If someone can jump shots, they leap straight into the air.
If someone leaps straight into the air, they activate their leg muscles.
Yuri does not activate his leg muscles.

Conclusion:
If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.

# solution in code:
def solution():
    # Some professional basketball players are not American nationals.
    Exists([x], And(Is_Professional_Basketball_Player(x), Not(Is_American_National(x))))
    # All professional basketball players can do jump shots.
    ForAll([x], Implies(Is_Professional_Basketball_Player(x), Can_Do_Jump_Shots(x)))
    # If someone can jump shots, they leap straight into the air.
    ForAll([x], Implies(Can_Do_Jump_Shots(x), Leaps_Straight_Into_Air(x)))
    # If someone leaps straight into the air, they activate their leg muscles.
    ForAll([x], Implies(Leaps_Straight_Into_Air(x), Activates_Leg_Muscles(x)))
    # Yuri does not activate his leg muscles.
    Not(Activates_Leg_Muscles(Yuri))

    # If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.
    return Implies(Not(Leaps_Straight_Into_Air(Yuri)), And(Is_American_National(Yuri), Is_Professional_Basketball_Player(Yuri)))

You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the above examples. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. 
""".strip()


def prompt_examples_satdev_v0_cot_solver():
    return """

You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. After writing the code, please solve the value of the result and provide the final answer in the comment in the format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best. 

Example 1
Premises:
The Mona Lisa is a world's best-known painting.
The Mona Lisa is a portrait painted by Leonardo da Vinci.
Leonardo da Vinci was a scientist and painter.
Painting genres can be history, portrait, animal, landscape, and still life.

Conclusion:
Leonardo da Vinci has artworks in the landscape genre.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # The Mona Lisa is a world's best-known painting.
    And(Is_Painting(Mona_Lisa), Best_Known_Painting(Mona_Lisa))
    # The Mona Lisa is a portrait painted by Leonardo da Vinci.
    And(Is_Portrait(Mona_Lisa), Painted_By(Mona_Lisa, Leonardo_da_Vinci))
    # Leonardo da Vinci was a scientist and painter.
    And(Is_Scientist(Leonardo_da_Vinci), Is_Painter(Leonardo_da_Vinci))
    # Painting genres can be history, portrait, animal, landscape, and still life.
    ForAll([x], Implies(Is_Painting(x), Or(Is_History(x), Is_Portrait(x), Is_Animal(x), Is_Landscape(x), Is_Still_Life(x))))

    # Leonardo da Vinci has artworks in the landscape genre.
    return Exists([x], And(Is_Artwork(x), Is_Landscape(x), Painted_By(x, Leonardo_da_Vinci)))

    # Solve the truth value of the returned statement
    # The Mona Lisa is a portrait painted by Leonardo da Vinci.
    # And(Is_Portrait(Mona_Lisa), Painted_By(Mona_Lisa, Leonardo_da_Vinci))
    # Is_Portrait(Mona_Lisa) = True
    # Painted_By(Mona_Lisa, Leonardo_da_Vinci) = True
    # The Mona Lisa is a world's best-known painting.
    # And(Is_Painting(Mona_Lisa), Best_Known_Painting(Mona_Lisa))
    # Is_Painting(Mona_Lisa) = True
    # Exists([x], And(Is_Artwork(x), Painted_By(x, Leonardo_da_Vinci))) = True
    # Painting genres can be history, portrait, animal, landscape, and still life.
    # ForAll([x], Implies(Is_Painting(x), Or(Is_History(x), Is_Portrait(x), Is_Animal(x), Is_Landscape(x), Is_Still_Life(x))))
    # Or(Is_History(Mona_Lisa), Is_Portrait(Mona_Lisa), Is_Animal(Mona_Lisa), Is_Landscape(Mona_Lisa), Is_Still_Life(Mona_Lisa)) = True
    # However, there is no information that mentions that Mona Lisa is in the landscape genre.
    # So we cannot determine the truth value of Is_Landscape(Mona_Lisa)
    # Therefore, the returned statement is Unknown
    # Answer: C



Example 2
Premises:
Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.

Conclusion:
A Czech published a book in 1946.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    And(Czech(Miroslav_Venhoda), Choral_Conductor(Miroslav_Venhoda), Specialized_In(Miroslav_Venhoda, Renaissance_and_Baroque_Music))
    # Any choral conductor is a musician.
    ForAll([x], Implies(Choral_Conductor(x), Is_Musician(x)))
    # Some musicians love music.
    Exists([x], And(Is_Musician(x), Loves(x, Music)))
    # Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    And(Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda), Published_In(Method_of_Studying_Gregorian_Chant, 1946))

    # A Czech published a book in 1946.
    return Exists([x, y], And(Czech(x), Published_By(y, x), Published_In(y, 1946)))

    # Solve the truth value of the returned statement
    # Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    # And(Czech(Miroslav_Venhoda), Choral_Conductor(Miroslav_Venhoda), Specialized_In(Miroslav_Venhoda, Renaissance_and_Baroque_Music))
    # Czech(Miroslav_Venhoda) = True
    # Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    # And(Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda), Published_In(Method_of_Studying_Gregorian_Chant, 1946))
    # Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda) = True
    # Published_In(Method_of_Studying_Gregorian_Chant, 1946) = True
    # And(Czech(Miroslav_Venhoda), And(Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda), Published_In(Method_of_Studying_Gregorian_Chant, 1946)) = True
    # Exists([x, y], And(Czech(x), Published_By(y, x), Published_In(y, 1946))) = True
    # Therefore, the returned statement is True
    # Answer: A

Example 3
Premises:
The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
Airbus made more revenue than Boeing last year.

Conclusion:
There does not exist a United Airlines plane produced by Boeing.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
    ForAll([x], Implies(Is_Major_Manufacturer(x), Or(x == Boeing, x == Airbus)))
    # All American Airlines planes are from the world's major large passenger aircraft manufacturers.
    ForAll([x], Implies(Is_American_Airlines(x), Exists([y], And(Is_Major_Manufacturer(y), Made_By(x, y))))
    # Airbus made more revenue than Boeing last year.
    Make_More_Revenue(Airbus, Boeing)

    # There does not exist a United Airlines plane produced by Boeing.
    return Not(Exists([x], And(Is_United_Airlines(x), Made_By(x, Boeing))))

    # Solve the truth value of the returned statement
    # The returned statement involves the following relation: Is_United_Airlines(x)
    # This relation is not defined in the premises
    # Therefore, the truth value of the returned statement is unknown
    # Answer: C

Example 4
Premises:
Some professional basketball players are not American nationals.
All professional basketball players can do jump shots.
If someone can jump shots, they leap straight into the air.
If someone leaps straight into the air, they activate their leg muscles.
Yuri does not activate his leg muscles.

Conclusion:
If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # Some professional basketball players are not American nationals.
    Exists([x], And(Is_Professional_Basketball_Player(x), Not(Is_American_National(x))))
    # All professional basketball players can do jump shots.
    ForAll([x], Implies(Is_Professional_Basketball_Player(x), Can_Do_Jump_Shots(x)))
    # If someone can jump shots, they leap straight into the air.
    ForAll([x], Implies(Can_Do_Jump_Shots(x), Leaps_Straight_Into_Air(x)))
    # If someone leaps straight into the air, they activate their leg muscles.
    ForAll([x], Implies(Leaps_Straight_Into_Air(x), Activates_Leg_Muscles(x)))
    # Yuri does not activate his leg muscles.
    Not(Activates_Leg_Muscles(Yuri))

    # If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.
    return Implies(Not(Leaps_Straight_Into_Air(Yuri)), And(Is_American_National(Yuri), Is_Professional_Basketball_Player(Yuri)))

    # Solve the truth value of the returned statement
    # If someone can jump shots, they leap straight into the air.
    # ForAll([x], Implies(Can_Do_Jump_Shots(x), Leaps_Straight_Into_Air(x)))
    # Implies(Not(Leaps_Straight_Into_Air(Yuri)), Not(Can_Do_Jump_Shots(Yuri))) = False
    # All professional basketball players can do jump shots.
    # ForAll([x], Implies(Is_Professional_Basketball_Player(x), Can_Do_Jump_Shots(x)))
    # Implies(Not(Can_Do_Jump_Shots(Yuri)), (Is_Professional_Basketball_Player(Yuri))) = False
    # Implies(Not(Leaps_Straight_Into_Air(Yuri)), (Is_Professional_Basketball_Player(Yuri))) = False
    # Implies(Not(Leaps_Straight_Into_Air(Yuri)), And(Is_American_National(Yuri), Is_Professional_Basketball_Player(Yuri))) = False
    # Therefore, the returned statement is False
    # Answer: B



You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. After writing the code, please solve the value of the result and provide the final answer in the comment in the format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best. 

Premises: {} \n
Conclusion: {} \n
Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown
""".strip()



def prompt_examples_satdev_v0_cot_solver_no_cot():
    return """

You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. After writing the code, please state the value of the final result and provide the answer in a comment in the format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best. 

Example 1
Premises:
The Mona Lisa is a world's best-known painting.
The Mona Lisa is a portrait painted by Leonardo da Vinci.
Leonardo da Vinci was a scientist and painter.
Painting genres can be history, portrait, animal, landscape, and still life.

Conclusion:
Leonardo da Vinci has artworks in the landscape genre.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # The Mona Lisa is a world's best-known painting.
    And(Is_Painting(Mona_Lisa), Best_Known_Painting(Mona_Lisa))
    # The Mona Lisa is a portrait painted by Leonardo da Vinci.
    And(Is_Portrait(Mona_Lisa), Painted_By(Mona_Lisa, Leonardo_da_Vinci))
    # Leonardo da Vinci was a scientist and painter.
    And(Is_Scientist(Leonardo_da_Vinci), Is_Painter(Leonardo_da_Vinci))
    # Painting genres can be history, portrait, animal, landscape, and still life.
    ForAll([x], Implies(Is_Painting(x), Or(Is_History(x), Is_Portrait(x), Is_Animal(x), Is_Landscape(x), Is_Still_Life(x))))

    # Leonardo da Vinci has artworks in the landscape genre.
    return Exists([x], And(Is_Artwork(x), Is_Landscape(x), Painted_By(x, Leonardo_da_Vinci)))

    # Solve the truth value of the returned statement
    # Answer: C



Example 2
Premises:
Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.

Conclusion:
A Czech published a book in 1946.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    And(Czech(Miroslav_Venhoda), Choral_Conductor(Miroslav_Venhoda), Specialized_In(Miroslav_Venhoda, Renaissance_and_Baroque_Music))
    # Any choral conductor is a musician.
    ForAll([x], Implies(Choral_Conductor(x), Is_Musician(x)))
    # Some musicians love music.
    Exists([x], And(Is_Musician(x), Loves(x, Music)))
    # Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    And(Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda), Published_In(Method_of_Studying_Gregorian_Chant, 1946))

    # A Czech published a book in 1946.
    return Exists([x, y], And(Czech(x), Published_By(y, x), Published_In(y, 1946)))

    # Solve the truth value of the returned statement
    # Answer: A

Example 3
Premises:
The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
Airbus made more revenue than Boeing last year.

Conclusion:
There does not exist a United Airlines plane produced by Boeing.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
    ForAll([x], Implies(Is_Major_Manufacturer(x), Or(x == Boeing, x == Airbus)))
    # All American Airlines planes are from the world's major large passenger aircraft manufacturers.
    ForAll([x], Implies(Is_American_Airlines(x), Exists([y], And(Is_Major_Manufacturer(y), Made_By(x, y))))
    # Airbus made more revenue than Boeing last year.
    Make_More_Revenue(Airbus, Boeing)

    # There does not exist a United Airlines plane produced by Boeing.
    return Not(Exists([x], And(Is_United_Airlines(x), Made_By(x, Boeing))))

    # Solve the truth value of the returned statement
    # Answer: C

Example 4
Premises:
Some professional basketball players are not American nationals.
All professional basketball players can do jump shots.
If someone can jump shots, they leap straight into the air.
If someone leaps straight into the air, they activate their leg muscles.
Yuri does not activate his leg muscles.

Conclusion:
If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.

Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown

# solution in code:
def solution():
    # Some professional basketball players are not American nationals.
    Exists([x], And(Is_Professional_Basketball_Player(x), Not(Is_American_National(x))))
    # All professional basketball players can do jump shots.
    ForAll([x], Implies(Is_Professional_Basketball_Player(x), Can_Do_Jump_Shots(x)))
    # If someone can jump shots, they leap straight into the air.
    ForAll([x], Implies(Can_Do_Jump_Shots(x), Leaps_Straight_Into_Air(x)))
    # If someone leaps straight into the air, they activate their leg muscles.
    ForAll([x], Implies(Leaps_Straight_Into_Air(x), Activates_Leg_Muscles(x)))
    # Yuri does not activate his leg muscles.
    Not(Activates_Leg_Muscles(Yuri))

    # If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.
    return Implies(Not(Leaps_Straight_Into_Air(Yuri)), And(Is_American_National(Yuri), Is_Professional_Basketball_Player(Yuri)))

    # Solve the truth value of the returned statement
    # Answer: B



You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. After writing the code, please state the value of the final result and provide the answer in a comment in the format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best. 

Premises: {} \n
Conclusion: {} \n
Question: Based on the premises, is the conclusion true, false, or unknown? If you cannot determine whether the conclusion is true or false, choose unknown.
A. True
B. False
C. Unknown
""".strip()

def prompt_examples_satdev_v0_no_comment():
    return """
You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. 

Example 1
Premises:
Premise 1. The Mona Lisa is a world's best-known painting.
Premise 2. The Mona Lisa is a portrait painted by Leonardo da Vinci.
Premise 3. Leonardo da Vinci was a scientist and painter.
Premise 4. Painting genres can be history, portrait, animal, landscape, and still life.

Conclusion:
Leonardo da Vinci has artworks in the landscape genre.

# solution in code:
def solution():
    # Premise 1
    And(Is_Painting(Mona_Lisa), Best_Known_Painting(Mona_Lisa))
    # Premise 2
    And(Is_Portrait(Mona_Lisa), Painted_By(Mona_Lisa, Leonardo_da_Vinci))
    # Premise 3
    And(Is_Scientist(Leonardo_da_Vinci), Is_Painter(Leonardo_da_Vinci))
    # Premise 4
    ForAll([x], Implies(Is_Painting(x), Or(Is_History(x), Is_Portrait(x), Is_Animal(x), Is_Landscape(x), Is_Still_Life(x))))

    # Conclusion
    return Exists([x], And(Is_Artwork(x), Is_Landscape(x), Painted_By(x, Leonardo_da_Vinci)))

Example 2
Premises:
Premise 1. Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Premise 2. Any choral conductor is a musician.
Premise 3. Some musicians love music.
Premise 4. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.

Conclusion:
A Czech published a book in 1946.

# solution in code:
def solution():
    # Premise 1
    And(Czech(Miroslav_Venhoda), Choral_Conductor(Miroslav_Venhoda), Specialized_In(Miroslav_Venhoda, Renaissance_and_Baroque_Music))
    # Premise 2
    ForAll([x], Implies(Choral_Conductor(x), Is_Musician(x)))
    # Premise 3
    Exists([x], And(Is_Musician(x), Loves(x, Music)))
    # Premise 4
    And(Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda), Published_In(Method_of_Studying_Gregorian_Chant, 1946))

    # Conclusion
    return Exists([x, y], And(Czech(x), Published_By(y, x), Published_In(y, 1946)))

Example 3
Premises:
Premise 1. The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
Premise 2. All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
Premise 3. Airbus made more revenue than Boeing last year.

Conclusion:
There does not exist a United Airlines plane produced by Boeing.

# solution in code:
def solution():
    # Premise 1
    ForAll([x], Implies(Is_Major_Manufacturer(x), Or(x == Boeing, x == Airbus)))
    # Premise 2
    ForAll([x], Implies(Is_American_Airlines(x), Exists([y], And(Is_Major_Manufacturer(y), Made_By(x, y))))
    # Premise 3
    Make_More_Revenue(Airbus, Boeing)

    # Conclusion
    return Not(Exists([x], And(Is_United_Airlines(x), Made_By(x, Boeing))))

Example 4
Premises:
Premise 1. Some professional basketball players are not American nationals.
Premise 2. All professional basketball players can do jump shots.
Premise 3. If someone can jump shots, they leap straight into the air.
Premise 4. If someone leaps straight into the air, they activate their leg muscles.
Premise 5. Yuri does not activate his leg muscles.

Conclusion:
If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.

# solution in code:
def solution():
    # Premise 1
    Exists([x], And(Is_Professional_Basketball_Player(x), Not(Is_American_National(x))))
    # Premise 2
    ForAll([x], Implies(Is_Professional_Basketball_Player(x), Can_Do_Jump_Shots(x)))
    # Premise 3
    ForAll([x], Implies(Can_Do_Jump_Shots(x), Leaps_Straight_Into_Air(x)))
    # Premise 4
    ForAll([x], Implies(Leaps_Straight_Into_Air(x), Activates_Leg_Muscles(x)))
    # Premise 5
    Not(Activates_Leg_Muscles(Yuri))

    # Conclusion
    return Implies(Not(Leaps_Straight_Into_Air(Yuri)), And(Is_American_National(Yuri), Is_Professional_Basketball_Player(Yuri)))

You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the above examples. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. 
""".strip()


def prompt_examples_satdev_v1():
        return """
Premises:
The Mona Lisa is a world's best-known painting.
The Mona Lisa is a portrait painted by Leonardo da Vinci.
Leonardo da Vinci was a scientist and painter.
Painting genres can be history, portrait, animal, landscape, and still life.

Conclusion:
Leonardo da Vinci has artworks in the landscape genre.

# solution in code:
def solution():
    # The Mona Lisa is a world's best-known painting.
    Is_Painting(Mona_Lisa) and Best_Known(Mona_Lisa)
    # The Mona Lisa is a portrait painted by Leonardo da Vinci.
    Is_Portrait(Mona_Lisa) and Painted_By(Mona_Lisa, Leonardo_da_Vinci)
    # Leonardo da Vinci was a scientist and painter.
    Is_Scientist(Leonardo_da_Vinci) and Is_Painter(Leonardo_da_Vinci)
    # Painting genres can be history, portrait, animal, landscape, and still life.
    ForAll([x], Implies(Is_Painting(x), Is_History(x) or Is_Portrait(x) or Is_Animal(x) or Is_Landscape(x) or Is_Still_Life(x)))

    # Leonardo da Vinci has artworks in the landscape genre.
    return Exists([x], Is_Artwork(x) and Is_Landscape(x) and Painted_By(x, Leonardo_da_Vinci))



Premises:
Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.

Conclusion:
A Czech published a book in 1946.

# solution in code:
def solution():
    # Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    Czech(Miroslav_Venhoda) and Choral_Conductor(Miroslav_Venhoda) and Specialized_In(Miroslav_Venhoda, Renaissance_and_Baroque_Music)
    # Any choral conductor is a musician.
    ForAll([x], Implies(Choral_Conductor(x), Is_Musician(x)))
    # Some musicians love music.
    Exists([x], Is_Musician(x) and Loves(x, Music))
    # Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Published_By(Method_of_Studying_Gregorian_Chant, Miroslav_Venhoda) and Published_In(Method_of_Studying_Gregorian_Chant, 1946)

    # A Czech published a book in 1946.
    return Exists([x, y], Czech(x) and Published_By(y, x) and Published_In(y, 1946))



Premises:
The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
All American Airlines planes are from the world's major large passenger aircraft manufacturers. 
Airbus made more revenue than Boeing last year.

Conclusion:
There does not exist a United Airlines plane produced by Boeing.

# solution in code:
def solution():
    # The world's only major large passenger aircraft manufacturers are Boeing and Airbus.
    ForAll([x], Implies(Is_Major_Manufacturer(x), x == Boeing or x == Airbus))
    # All American Airlines planes are from the world's major large passenger aircraft manufacturers.
    ForAll([x], Implies(Is_American_Airlines(x), Exists([y], Is_Major_Manufacturer(y) or Made_By(x, y))))
    # Airbus made more revenue than Boeing last year.
    Make_More_Revenue(Airbus, Boeing)

    # There does not exist a United Airlines plane produced by Boeing.
    return Not(Exists([x], Is_United_Airlines(x) and Made_By(x, Boeing)))



Premises:
Some professional basketball players are not American nationals.
All professional basketball players can do jump shots.
If someone can jump shots, they leap straight into the air.
If someone leaps straight into the air, they activate their leg muscles.
Yuri does not activate his leg muscles.

Conclusion:
If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.

# solution in code:
def solution():
    # Some professional basketball players are not American nationals.
    Exists([x], Is_Professional_Basketball_Player(x) and Not(Is_American_National(x)))
    # All professional basketball players can do jump shots.
    ForAll([x], Implies(Is_Professional_Basketball_Player(x), Can_Do_Jump_Shots(x)))
    # If someone can jump shots, they leap straight into the air.
    ForAll([x], Implies(Can_Do_Jump_Shots(x), Leaps_Straight_Into_Air(x)))
    # If someone leaps straight into the air, they activate their leg muscles.
    ForAll([x], Implies(Leaps_Straight_Into_Air(x), Activates_Leg_Muscles(x)))
    # Yuri does not activate his leg muscles.
    not Activates_Leg_Muscles(Yuri)

    # If Yuri does not leap straight into the air, then Yuri is an American professional basketball player.
    return Implies(not Leaps_Straight_Into_Air(Yuri), Is_American_National(Yuri) and Is_Professional_Basketball_Player(Yuri))
""".strip()


class FOLIOSymDataset(FOLIODataset):

    def __init__(self, path_or_url='yale-nlp/FOLIO', split='validation', subset='default', generating_paraphrase=False,symbolic=False, sym_solver='satlm', *args, **kwargs):
        assert sym_solver in ['satlm', 'satdev']
        self.sym_solver = sym_solver
        super().__init__(path_or_url, split, subset, generating_paraphrase, symbolic, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.foliosym

        # override to add satlm prompt
    def load_dataset(self, path_or_url):
        exs = super().load_dataset(path_or_url)
        for ex in exs:
            ex['dataset_type'] = self.dataset_types.foliosym
            ex['sym_solver'] = self.sym_solver

            # question = ('Problem: Based on the premises, is the conclusion true, false, or unknown?\n'
            #             'Premises:\n'
            #             f'{ex["premises"]}\n\n'
            #             'Conclusion:\n'
            #             f'{ex["conclusion"]}')
            # premises = ''
            # for idx,p in enumerate(ex['premises'].split('.')):
            #     if len(p.strip()) <=1:
            #         continue
            #     premises += f'Premise {idx+1}. {p.strip()}.\n'
            # print(premises)
            premises = ex['premises']
            question = ('Premises:\n'
                        f'{premises}\n\n'
                        'Conclusion:\n'
                        f'{ex["conclusion"]}')
            # ex['prompt_parts']['sat_system_prompt'] = 'You are a helpful AI assistant that will answer reasoning questions. You will write logical formulas in the form of code to represent a given set of premises and a given conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. '
            ex['prompt_parts']['sat_system_prompt'] = 'You are a helpful AI assistant. You are good at writing code and answering reasoning questions. '
            if self.sym_solver == 'satlm':
                fs_sat_prompt = prompt_examples_satlm() + "\n\n\n\n" + question
            else:
                fs_sat_prompt = prompt_examples_satdev_v0() + "\n\n\n\n" + question
                # fs_sat_prompt = prompt_examples_satdev_v0_no_comment() + "\n\n\n\n" + question
            ex['prompt_parts']['fs_sat_prompt'] = fs_sat_prompt
        return exs

    # override to add satlm messages
    def process_example(self, example, *args, **kwargs):
        example = super().process_example(example, *args, **kwargs)
        def create_msgs(prompt, sys_prompt=None,multi_turn_few_shot=False):
            if prompt is None:
                return None
            if multi_turn_few_shot:
                icl_examples = prompt.split("\n\n\n\n")[:-1]
                question = prompt.split("\n\n\n\n")[-1]
                msg = []
                for i, icl_ex in enumerate(icl_examples):
                    icl_q = icl_ex.split("# solution in code:")[0]
                    icl_ans = "# solution in code:\n" + icl_ex.split("# solution in code:")[1]
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
        example['fs_sat_messages'] = create_msgs(prompt_parts.get('fs_sat_prompt'), prompt_parts.get('sat_system_prompt'), multi_turn_few_shot=False)
        # print(example['fs_sat_messages'])
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
            try:
                ans = resp.split("def solution():")[1].strip()
                if example['sym_solver'] == 'satlm':
                    ans = fol_satlm_exec(ans)
                else:
                    ans = fol_uninterp_satlm_exec(ans)
                ans = str(ans)
                if ans == 'N/A':
                    ans = 'Unknown'
                correct = ans == answer
                print("Answer:", ans, "GT:", answer)
            except Exception as e:
                ans = None
                correct = False
                print("SatLM Eval Exec:", e)
            if ans not in ['True', 'False', 'Unknown']:
                ans = None
                correct = False
            
            if ans is None:
                print("Incorrect Answer")
                question = ('Premises:\n'
                        f'{example["premises"]}\n\n'
                        'Conclusion:\n'
                        f'{example["conclusion"]}')
                print("Question:\n", question)
                print("Response:\n", resp)


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

    dataset = FOLIODataset()

    ex = dataset[0]


    responses = [
        'I think the answer is TRUE, A',
        'ANSWER: A\n\nBecause of..\n\nSo answer TRUE',
        'I think because...\n\nANSWER: True',
        'I think because...\n\nANSWER: A or B'
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
    print(ex['messages'][0]['content'])
