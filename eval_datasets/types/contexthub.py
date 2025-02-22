import random
from copy import deepcopy

from pathlib import Path
import json
from src.utils.paths import ROOT_FOLDER

from eval_datasets import ReasoningDataset, CSQADataset
from eval_datasets.sat_solver.boolean_solver import boolean_satlm_exec

random.seed(0)



def build_cot(curr_ex, icl_exs):
    cot_messages = []
    direct_messages = []

    for ex in icl_exs:
        cot_messages.append({
            'role': 'user',
            'content': ex['cot_prompt']
        })
        cot_messages.append({
            'role': 'assistant',
            'content': ex['cot'].split("Thus, the answer is")[0].strip() + ' The best answer is ' + ex['answerKey'] + '.'
        })

        direct_messages.append({
            'role': 'user',
            'content': ex['direct_prompt']
        })
        direct_messages.append({
            'role': 'assistant',
            'content': 'The best answer is ' + ex['answerKey'] + '.'
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
        'zs_direct': [direct_messages[-1], {'role': 'assistant', 'content': 'The best answer is '}],
        'zs_cot': [cot_messages[-1]],
        'fs_direct': direct_messages + [{'role': 'assistant', 'content': 'The best answer is '}],
        'fs_cot': cot_messages
    }


def different_logic_nl_icl(ex, dataset, num=3):
    selected_icls = []
    icl_idxs = []
    icl_categories = []

    icl_answers = []

    # root_idx = ex['root_idx']
    # max_root_idx = max([x['root_idx'] for x in dataset])
    # root_idx = (root_idx + 1) % max_root_idx
    # find the first example with the same root_idx
    for xidx, x in enumerate(dataset):
        if xidx in icl_idxs:
            continue
        if any([z.lower() in y.lower() for y in icl_categories for z in x['categories']]):
            continue

        if ex['root_idx'] != x['root_idx'] and all(['abstract' not in y.lower() for y in x['categories']]) and (len(icl_answers) == 0 or len(icl_answers) > 1 or x['answerKey'] != icl_answers[-1]):
            selected_icls.append(x)
            icl_idxs.append(xidx)
            icl_categories.append(" ".join(x["categories"]))
            icl_answers.append(x['answerKey'])

            if len(selected_icls) == num:
                break


    return build_cot(ex, selected_icls)

def different_logic_nl_icl_paraphrase(ex, icl_examples):
    different_logic_icls = []
    nl_logic = ex['nl_logic']
    for icl_ex in icl_examples:
        if nl_logic != icl_ex['nl_logic']:
            different_logic_icls.append(icl_ex)
    return build_cot(ex, icl_examples)

def same_logic_nl_icl(ex, dataset, num=1):
    selected_icls = []
    icl_idxs = []
    icl_categories = []

    root_idx = ex['root_idx']
    sub_cat1, sub_cat2 = ex['categories']
    for xidx, x in enumerate(dataset):
        if xidx in icl_idxs:
            continue
        if any([z.lower() in y.lower() for y in icl_categories for z in x['categories']]):
            continue

        if x['root_idx'] == root_idx and sub_cat1 != x["categories"][0] and sub_cat2 != x["categories"][1] and all(['abstract' not in y.lower() for y in x['categories']]):
            selected_icls.append(x)
            icl_idxs.append(xidx)
            icl_categories.append(" ".join(x["categories"]))

            if len(selected_icls) == num:
                break
    return build_cot(ex, selected_icls)


def different_logic_abstract_icl(ex, dataset, num=3):
    selected_icls = []
    icl_idxs = []
    icl_categories = []

    root_idx = ex['root_idx']
    max_root_idx = max([x['root_idx'] for x in dataset])
    root_idx = (root_idx + 1) % max_root_idx
    # find the first example with the same root_idx
    for xidx, x in enumerate(dataset):
        if xidx in icl_idxs:
            continue


        if x['root_idx'] == root_idx and any(['abstract' in y.lower() for y in x['categories']]) and xidx not in icl_idxs:
            # return build_cot(ex, [x])
            selected_icls.append(x)
            icl_idxs.append(xidx)
            icl_categories.append(" ".join(x["categories"]))
            root_idx = root_idx + 1 % max_root_idx
            if len(selected_icls) == num:
                break
    return build_cot(ex, selected_icls)


def same_logic_abstract_icl(ex, dataset, num=1):
    root_idx = ex['root_idx']
    sub_cat1, sub_cat2 = ex['categories']
    selected_icls = []

    for x in dataset:
        if x['root_idx'] == root_idx and any(['abstract' in y.lower() for y in x['categories']]) and " ".join(ex["categories"]) != " ".join(x["categories"]):
            selected_icls.append(x)
            if len(selected_icls) == num:
                break

    return build_cot(ex, selected_icls)

class ContextHubDataset(ReasoningDataset):
    average_token_len = 1000

    def __init__(self, path_or_url=ROOT_FOLDER / 'eval_datasets/thirdparty/contexthub', *args, level: str = 'data_level1', logic_type: str = 'deductive', used_plan_solve_prompt: bool = False,use_llama_3_1_prompts: bool = True, used_cot_solver_prompt: bool = False, **kwargs):
        self.logic_type = logic_type
        self.used_plan_solve_prompt = used_plan_solve_prompt
        self.use_llama_3_1_prompts = use_llama_3_1_prompts
        self.used_cot_solver_prompt = used_cot_solver_prompt
        if isinstance(path_or_url, str):
            path_or_url = Path(path_or_url)

        if not str(path_or_url).endswith('.json'):
            path_or_url = path_or_url / level / (logic_type + '_logic_traincot.json')

        super().__init__(path_or_url, *args, **kwargs)

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.contexthub

    def load_dataset(self, path_or_url):
        data = json.load(open(str(path_or_url), 'r'))
        examples = []
        
        if 'paraphrase' in self.logic_type:
            icl_examples = []
            for raw_ex in data:
                root_ex = {**raw_ex}
                root_ex['nl_logic'] = root_ex['question_format'][0]["<nl>"]
                q_str = root_ex['paraphrase_question']
                choices = root_ex['choices']
                prompt = self.basic_prompt(q_str, self.format_choices(choices))

                root_ex['prompt_parts'] = {'user_context': prompt}
                root_ex["question_format"] = q_str
                
                root_ex['prompt_parts'] = {
                        'user_context': prompt
                    }
                root_ex['question'] = q_str

                root_ex['cot_prompt'] = "Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + q_str + '\nA. True\nB. False\nC. N/A\n\n' + "- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nGive a description of your reasoning before you answer.\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, or C.\n\nLet's think step by step."
                root_ex['direct_prompt'] ="Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + q_str + '\nA. True\nB. False\nC. N/A\n\n' + "Your response should end with \"The best answer is [the_answer_letter].\" where the [the_answer_letter] is one of A, B, or C."

                
                examples.append(root_ex)
                if 'gold_cot' in raw_ex:
                    root_ex['cot'] = raw_ex['gold_cot']
                    icl_examples.append(root_ex)
            random.shuffle(examples)
            if self.use_llama_3_1_prompts:
                for example in examples:
                    prompts = different_logic_nl_icl_paraphrase(example, icl_examples)
                    example['llama_3_1_eval'] = {'prompts': prompts}
            return examples
            

        for ridx, raw_ex in enumerate(data):
            root_ex = {}

            answer = 'True' if raw_ex['answer'] is True else ('False' if raw_ex['answer'] is False else 'N/A')
            choices = {"label": ["A","B","C"], "text": ['True', 'False', "N/A"]}

            root_ex['choices'] = choices
            root_ex['answer'] = answer
            root_ex['answer_index'] = 0 if raw_ex['answer'] is True else (1 if raw_ex['answer'] is False else 2)
            root_ex['answer_choice_tokens'] = ['A', 'B', 'C']
            root_ex['answerKey'] = 'A' if raw_ex['answer'] is True else ('B' if raw_ex['answer'] is False else 'C')
            root_ex["dataset_type"] = self.dataset_types.contexthub
            root_ex["question_format"] = raw_ex["question"]
            root_ex['root_idx'] = ridx

            for k, v in raw_ex.items():
                if k == 'answer' or k == 'question':
                    continue

                for subk, subv in v.items():
                    if not subv.get('<nl>'):
                        continue

                    ex = deepcopy(root_ex)

                    q_str = ' '.join(subv["<nl>"].split('\n'))

                    prompt = self.basic_prompt(q_str, self.format_choices(choices))

                    ex['prompt_parts'] = {
                        'user_context': prompt
                    }
                    ex['question'] = q_str

                    ex['cot_prompt'] = "Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + q_str + '\nA. True\nB. False\nC. N/A\n\n' + "- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nGive a description of your reasoning before you answer.\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, or C.\n\nLet's think step by step."
                    if self.used_plan_solve_prompt:
                        # ex['cot_prompt'] = "Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + q_str + '\nA. True\nB. False\nC. N/A\n\n' + "Let\'s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. Then, let\'s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer. When you are ready to answer write the answer in the format: \"Answer: <your answer>\". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."
                        ex['cot_prompt'] = "Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + q_str + '\nA. True\nB. False\nC. N/A\n\n' + "Let\'s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. You must explicitly list a plan and releant variables. Then, let\'s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer. When you are ready to answer write the answer in the format: \"Answer: <your answer>\". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."
                        
                        prompt = ex['cot_prompt']

                        ex['prompt_parts'] = {
                            'user_context': prompt
                        }
                    if self.used_cot_solver_prompt:
                        ex['cot_prompt'] = prompt_examples_satlm_cot_solver().format(q_str)
                        prompt = ex['cot_prompt']

                        ex['prompt_parts'] = {
                            'user_context': prompt
                        }
                    ex['direct_prompt'] ="Given the following question and three candidate answers (A, B, and C), choose the best answer.\nQuestion: " + q_str + '\nA. True\nB. False\nC. N/A\n\n' + "Your response should end with \"The best answer is [the_answer_letter].\" where the [the_answer_letter] is one of A, B, or C."

                    ex['nl_question'] = q_str
                    ex["categories"] = [k, subk]
                    ex['cot'] = subv['gold_cot']

                    examples.append({
                        **ex,
                    })

        random.shuffle(examples)
        if self.use_llama_3_1_prompts:
            for example in examples:
                prompts = different_logic_nl_icl(example, examples)
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

def prompt_examples_satlm():
    return """
Problem:
If the knight had either a magic sword or a faithful steed, then he was well-equipped for his quest to slay the dragon. However, if the knight was not well-equipped, then he was ill-prepared to face the formidable dragon. Given that it is false that the knight failed in his quest, what can be determined about whether the knight had a magic sword? (True, False, or N/A (undetermined)).

# solution in code:
def solution():
    # If the knight had either a magic sword or a faithful steed, then he was well-equipped for his quest to slay the dragon.
    Implies(Or(knight_had_a_magic_sword, knight_had_a_faithful_steed), knight_was_well_equipped)
    # However, if the knight was not well-equipped, then he was ill-prepared to face the formidable dragon.
    Implies(Not(knight_was_well_equipped), knight_was_ill_prepared)

    # Question premise: Given that it is false that the knight failed in his quest
    # This means the knight should not be ill-prepared
    Not(knight_was_ill_prepared)
    # what can be determined about whether the knight had a magic sword?
    return knight_had_a_magic_sword



Problem:
Emily practiced her multiplication tables but did not complete her math homework. If Emily completed her math homework or practiced her multiplication tables, then she spent time studying math. Emily did not read a book about science, but she did watch an educational video about planets. If Emily read a science book or watched an educational video about planets, then she learned something new about space. If Emily learned something new about space or spent time studying math, then she expanded her knowledge. Did Emily expand her knowledge?

# solution in code:
def solution():
    # Emily practiced her multiplication tables but did not complete her math homework.
    And(Emily_practiced_multiplication_tables, Not(Emily_completed_math_homework))
    # If Emily completed her math homework or practiced her multiplication tables, then she spent time studying math.
    Implies(Or(Emily_completed_math_homework, Emily_practiced_multiplication_tables), Emily_spent_time_studying_math)
    # Emily did not read a book about science, but she did watch an educational video about planets.
    And(Not(Emily_read_science_book), Emily_watched_educational_video_about_planets)
    # If Emily read a science book or watched an educational video about planets, then she learned something new about space.
    Implies(Or(Emily_read_science_book, Emily_watched_educational_video_about_planets), Emily_learned_something_new_about_space)
    # If Emily learned something new about space or spent time studying math, then she expanded her knowledge.
    Implies(Or(Emily_learned_something_new_about_space, Emily_spent_time_studying_math), Emily_expanded_her_knowledge)

    # Final question: Did Emily expand her knowledge?
    return Emily_expanded_her_knowledge



Problem: 
If serotonin deficiency is present and the patient's stress levels are chronically elevated, then their risk of developing depression is high. When a patient is experiencing persistent low mood or reports loss of interest in their usual activities, this suggests they meet diagnostic criteria for clinical depression. If the patient meets criteria for clinical depression or is at high risk for developing depression based on serotonin deficiency and chronic stress, then antidepressant medication is indicated as an appropriate treatment option. Given that the patient's depressive symptoms do not improve with treatment, what can be concluded about whether the patient is experiencing persistent low mood? (True, False, or N/A)

# solution in code:
def solution():
    # If serotonin deficiency is present and the patient's stress levels are chronically elevated, then their risk of developing depression is high.
    Implies(And(serotonin_deficiency_present, patient_stress_levels_chronically_elevated), patient_risk_of_developing_depression_high)
    # When a patient is experiencing persistent low mood or reports loss of interest in their usual activities, this suggests they meet diagnostic criteria for clinical depression.
    Implies(Or(patient_experiencing_persistent_low_mood, patient_reports_loss_of_interest), patient_meets_diagnostic_criteria_for_clinical_depression)
    # If the patient meets criteria for clinical depression or is at high risk for developing depression based on serotonin deficiency and chronic stress, then antidepressant medication is indicated as an appropriate treatment option.
    Implies(Or(patient_meets_diagnostic_criteria_for_clinical_depression, patient_risk_of_developing_depression_high), antidepressant_medication_indicated_as_appropriate_treatment)

    # Question premise: Given that the patient's depressive symptoms do not improve with treatment
    # This means mntidepressant medication should not be indicated as an appropriate treatment option
    Not(antidepressant_medication_indicated_as_appropriate_treatment)
    # Final question: what can be concluded about whether the patient is experiencing persistent low mood?
    return patient_experiencing_persistent_low_mood



Problem:
Alice's employer paid her salary. Alice also deposited her paycheck into her bank account. If Alice's employer paid her salary or she deposited her paycheck, then it means Alice received money. If Alice did not receive money, then it means she has no income.
Does the information imply that Alice has no income?

# solution in code:
def solution():
    # Alice's employer paid her salary.
    Alice_employer_paid_salary
    # Alice also deposited her paycheck into her bank account.
    Alice_deposited_paycheck
    # If Alice's employer paid her salary or she deposited her paycheck, then it means Alice received money.
    Implies(Or(Alice_employer_paid_salary, Alice_deposited_paycheck), Alice_received_money)
    # If Alice did not receive money, then it means she has no income.
    Implies(Not(Alice_received_money), Alice_has_no_income)

    # Final question: Does the information imply that Alice has no income?
    return Alice_has_no_income
""".strip()


def prompt_examples_satlm_cot_solver():
    return """
You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. After writing the code, please solve the value of the result and provide the final answer in the comment in the format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best. 

Problem:
If the knight had either a magic sword or a faithful steed, then he was well-equipped for his quest to slay the dragon. However, if the knight was not well-equipped, then he was ill-prepared to face the formidable dragon. Given that it is false that the knight failed in his quest, what can be determined about whether the knight had a magic sword?
A. True
B. False
C. Undetermined (N/A)

# solution in code:
def solution():
    # If the knight had either a magic sword or a faithful steed, then he was well-equipped for his quest to slay the dragon.
    Implies(Or(knight_had_a_magic_sword, knight_had_a_faithful_steed), knight_was_well_equipped)
    # However, if the knight was not well-equipped, then he was ill-prepared to face the formidable dragon.
    Implies(Not(knight_was_well_equipped), knight_was_ill_prepared)

    # Question premise: Given that it is false that the knight failed in his quest
    # This means the knight should not be ill-prepared
    Not(knight_was_ill_prepared)
    # what can be determined about whether the knight had a magic sword?
    return knight_had_a_magic_sword

    # Solve the truth value of the returned statement
    # Question premise: Given that it is false that the knight failed in his quest
    # Not(knight_was_ill_prepared)
    # However, if the knight was not well-equipped, then he was ill-prepared to face the formidable dragon.
    # Implies(Not(knight_was_well_equipped), knight_was_ill_prepared)
    # Not(knight_was_well_equipped) = False
    # knight_was_well_equipped = True
    # If the knight had either a magic sword or a faithful steed, then he was well-equipped for his quest to slay the dragon.
    # Implies(Or(knight_had_a_magic_sword, knight_had_a_faithful_steed), knight_was_well_equipped)
    # Or(knight_had_a_magic_sword, knight_had_a_faithful_steed) = True
    # However, it is possible that knight_had_a_magic_sword = True, or knight_had_a_faithful_steed=True and knight_had_a_magic_sword=False
    # Therefore, the returned statement is N/A.
    # Answer: C




Problem:
Emily practiced her multiplication tables but did not complete her math homework. If Emily completed her math homework or practiced her multiplication tables, then she spent time studying math. Emily did not read a book about science, but she did watch an educational video about planets. If Emily read a science book or watched an educational video about planets, then she learned something new about space. If Emily learned something new about space or spent time studying math, then she expanded her knowledge. Did Emily expand her knowledge?
A. True
B. False
C. Undetermined (N/A)

# solution in code:
def solution():
    # Emily practiced her multiplication tables but did not complete her math homework.
    And(Emily_practiced_multiplication_tables, Not(Emily_completed_math_homework))
    # If Emily completed her math homework or practiced her multiplication tables, then she spent time studying math.
    Implies(Or(Emily_completed_math_homework, Emily_practiced_multiplication_tables), Emily_spent_time_studying_math)
    # Emily did not read a book about science, but she did watch an educational video about planets.
    And(Not(Emily_read_science_book), Emily_watched_educational_video_about_planets)
    # If Emily read a science book or watched an educational video about planets, then she learned something new about space.
    Implies(Or(Emily_read_science_book, Emily_watched_educational_video_about_planets), Emily_learned_something_new_about_space)
    # If Emily learned something new about space or spent time studying math, then she expanded her knowledge.
    Implies(Or(Emily_learned_something_new_about_space, Emily_spent_time_studying_math), Emily_expanded_her_knowledge)

    # Final question: Did Emily expand her knowledge?
    return Emily_expanded_her_knowledge

    # Solve the truth value of the returned statement
    # Emily did not read a book about science, but she did watch an educational video about planets.
    # And(Not(Emily_read_science_book), Emily_watched_educational_video_about_planets)
    # Emily_watched_educational_video_about_planets = True
    # Or(Emily_read_science_book, Emily_watched_educational_video_about_planets) = True
    # If Emily read a science book or watched an educational video about planets, then she learned something new about space.
    # Implies(Or(Emily_read_science_book, Emily_watched_educational_video_about_planets), # Emily_learned_something_new_about_space)
    # Emily_learned_something_new_about_space = True
    # Or(Emily_learned_something_new_about_space, Emily_spent_time_studying_math) = True
    # If Emily learned something new about space or spent time studying math, then she expanded her knowledge.
    # Implies(Or(Emily_learned_something_new_about_space, Emily_spent_time_studying_math), Emily_expanded_her_knowledge)
    # Emily_expanded_her_knowledge = True
    # Therefore, the returned statement is True
    # Answer: A






Problem: 
If serotonin deficiency is present and the patient's stress levels are chronically elevated, then their risk of developing depression is high. When a patient is experiencing persistent low mood or reports loss of interest in their usual activities, this suggests they meet diagnostic criteria for clinical depression. If the patient meets criteria for clinical depression or is at high risk for developing depression based on serotonin deficiency and chronic stress, then antidepressant medication is indicated as an appropriate treatment option. Given that the patient's depressive symptoms do not improve with treatment, what can be concluded about whether the patient is experiencing persistent low mood?

A. True
B. False
C. Undetermined (N/A)

# solution in code:
def solution():
    # If serotonin deficiency is present and the patient's stress levels are chronically elevated, then their risk of developing depression is high.
    Implies(And(serotonin_deficiency_present, patient_stress_levels_chronically_elevated), patient_risk_of_developing_depression_high)
    # When a patient is experiencing persistent low mood or reports loss of interest in their usual activities, this suggests they meet diagnostic criteria for clinical depression.
    Implies(Or(patient_experiencing_persistent_low_mood, patient_reports_loss_of_interest), patient_meets_diagnostic_criteria_for_clinical_depression)
    # If the patient meets criteria for clinical depression or is at high risk for developing depression based on serotonin deficiency and chronic stress, then antidepressant medication is indicated as an appropriate treatment option.
    Implies(Or(patient_meets_diagnostic_criteria_for_clinical_depression, patient_risk_of_developing_depression_high), antidepressant_medication_indicated_as_appropriate_treatment)

    # Question premise: Given that the patient's depressive symptoms do not improve with treatment
    # This means mntidepressant medication should not be indicated as an appropriate treatment option
    Not(antidepressant_medication_indicated_as_appropriate_treatment)
    # Final question: what can be concluded about whether the patient is experiencing persistent low mood?
    return patient_experiencing_persistent_low_mood

    # Solve the truth value of the returned statement
    # Question premise: Given that the patient's depressive symptoms do not improve with treatment
    # This means mntidepressant medication should not be indicated as an appropriate treatment option
    # Not(antidepressant_medication_indicated_as_appropriate_treatment)
    # If the patient meets criteria for clinical depression or is at high risk for developing depression based on serotonin deficiency and chronic stress, then antidepressant medication is indicated as an appropriate treatment option.
    # Implies(Or(patient_meets_diagnostic_criteria_for_clinical_depression, patient_risk_of_developing_depression_high), antidepressant_medication_indicated_as_appropriate_treatment)
    # Or(patient_meets_diagnostic_criteria_for_clinical_depression, patient_risk_of_developing_depression_high) = False
    # patient_meets_diagnostic_criteria_for_clinical_depression = False
    # When a patient is experiencing persistent low mood or reports loss of interest in their usual activities, this suggests they meet diagnostic criteria for clinical depression.
    # Implies(Or(patient_experiencing_persistent_low_mood, patient_reports_loss_of_interest), patient_meets_diagnostic_criteria_for_clinical_depression)
    # Or(patient_experiencing_persistent_low_mood, patient_reports_loss_of_interest) = False
    # patient_experiencing_persistent_low_mood = False
    # Therefore, the returned statement is False
    # Answer: B


    




Problem:
Alice's employer paid her salary. Alice also deposited her paycheck into her bank account. If Alice's employer paid her salary or she deposited her paycheck, then it means Alice received money. If Alice did not receive money, then it means she has no income.
Does the information imply that Alice has no income?
A. True
B. False
C. Undetermined (N/A)

# solution in code:
def solution():
    # Alice's employer paid her salary.
    Alice_employer_paid_salary
    # Alice also deposited her paycheck into her bank account.
    Alice_deposited_paycheck
    # If Alice's employer paid her salary or she deposited her paycheck, then it means Alice received money.
    Implies(Or(Alice_employer_paid_salary, Alice_deposited_paycheck), Alice_received_money)
    # If Alice did not receive money, then it means she has no income.
    Implies(Not(Alice_received_money), Alice_has_no_income)

    # Final question: Does the information imply that Alice has no income?
    return Alice_has_no_income

    # Solve the truth value of the returned statement
    # Alice's employer paid her salary.
    # Alice_employer_paid_salary = True
    # Or(Alice_employer_paid_salary, Alice_deposited_paycheck) = True
    # If Alice's employer paid her salary or she deposited her paycheck, then it means Alice received money.
    # Implies(Or(Alice_employer_paid_salary, Alice_deposited_paycheck), Alice_received_money)
    # Alice_received_money = True
    # Not(Alice_received_money) = False
    # If Alice did not receive money, then it means she has no income.
    # Implies(Not(Alice_received_money), Alice_has_no_income)
    # Because Not(Alice_received_money) = False, Alice_has_no_income can be either True or False
    # Therefore, the returned statement is N/A
    # Answer: C

You will read a reasoning question including a set of premises and a conclusion. You will write logical formulas in the form of code to represent the given set of premises and the given conclusion to judge the truth value of the conclusion. You will only write code and nothing else. Your code should closely follow the format in the examples below. You should define every variable name clearly before you use any variable. You should use "_" to connect words together if your variable name has multiple words in it. After writing the code, please solve the value of the result and provide the final answer in the comment in the format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best. 

Problem:
{}
A. True
B. False
C. Undetermined (N/A)


    
""".strip()


class ContextHubSymDataset(ContextHubDataset):
    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.contexthubsym

    # override to add satlm prompt
    def load_dataset(self, path_or_url):
        exs = super().load_dataset(path_or_url)
        for ex in exs:
            ex['dataset_type'] = self.dataset_types.contexthubsym
            if "nl_question" in ex:
                question = ex["nl_question"]
            else:
                question = ex["paraphrase_question"]
            ex['prompt_parts']['sat_system_prompt'] = 'You are a helpful AI assistant that will answer reasoning questions. You will write logical formulas in the form of code to solve the problems. DO NOT return an answer like "True" or "False" or "N/A". Instead, always return the logical formula that represents the target answer.'
            fs_sat_prompt = prompt_examples_satlm() + "\n\n\n\n" + f"Problem:\n{question}"
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
        example['fs_sat_messages'] = create_msgs(prompt_parts.get('fs_sat_prompt'), prompt_parts.get('sat_system_prompt'))
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
                ans = boolean_satlm_exec(ans)
                ans = str(ans)
                correct = ans == answer
                print("Answer:", ans, "GT:", answer)
            except Exception as e:
                ans = None
                correct = False
                print("SatLM Eval Exec:", e)
                print("Response:", resp)
            if ans not in ['True', 'False', 'N/A']:
                ans = None
                correct = False
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

    dataset = ContextHubDataset()

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
