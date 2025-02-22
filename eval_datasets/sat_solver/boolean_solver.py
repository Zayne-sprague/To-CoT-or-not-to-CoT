import re

from z3 import *
from func_timeout import func_timeout

VAR_REGEX = r"[_a-zA-Z0-9]+[,)]"

PREDEFIND_FUNCS = ["ForAll", "Exist", "And", "Or", "Not", "Implies"]

def extract_vars(line):
    all_vars = re.findall(VAR_REGEX, line)
    all_vars = [all_vars.rstrip(",)") for all_vars in all_vars]
    return all_vars

def adhoc_syntax_fix(line):
    line = line.replace(" = True", " == True")
    line = line.replace(" = False", " == False")
    if " and " in line:
        line = re.sub(r"([_a-zA-Z]+) and ([_a-zA-Z]+)", r"And(\1, \2)", line)
        # handle Not(x) and y
        line = re.sub(r"Not\(([_a-zA-Z]+)\) and ([_a-zA-Z]+)", r"And(Not(\1), \2)", line)
        # handle x and Not(y)
        line = re.sub(r"([_a-zA-Z]+) and Not\(([_a-zA-Z]+)\)", r"And(\1, Not(\2))", line) 
    if " or " in line:
        line = re.sub(r"([_a-zA-Z]+) or ([_a-zA-Z]+)", r"Or(\1, \2)", line)
        # handle Not(x) or y
        line = re.sub(r"Not\(([_a-zA-Z]+)\) or ([_a-zA-Z]+)", r"Or(Not(\1), \2)", line)
        # handle x or Not(y)
        line = re.sub(r"([_a-zA-Z]+) or Not\(([_a-zA-Z]+)\)", r"Or(\1, Not(\2))", line)
    return line

def boolean_satlm_exec(code, return_code=False):
    lines = code.splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]
    lines = [l.replace("'", "") for l in lines]
    lines = [adhoc_syntax_fix(l) for l in lines]

    # assert lines[-1].startswith("return")
    # result_line = lines[-1]
    # lines = lines[:-1]
    if lines[-1].startswith("return"):
        result_line = lines[-1]
        lines = lines[:-1]
    else:
        for i,line in enumerate(lines):
            if line.startswith("return"):
                result_line = line
                lines = lines[:i]
                break

    vars = set()

    for line in lines:
        line_vars = extract_vars(line)
        vars.update(line_vars)

    vars = [v for v in vars]


    translated_lines = []
    translated_lines.append('z3._main_ctx = Context()')
    for var in vars:
        translated_lines.append(f"{var} = Bool('{var}')")
    translated_lines.append("precond = []")

    for line in lines:
        translated_lines.append("precond.append({})".format(line))

    translated_lines.append("s = Solver()")
    translated_lines.append("s.add(precond)")

    return_clause = result_line.split("return")[1].strip()
    translated_lines.append(f"question = {return_clause}")

    # translated_lines.append("s.add(Not({}))".format(return_clause))
    # translated_lines.extend([
    #     "if s.check() == unsat:",
    #     "    print('True')",
    #     "else:",
    #     "    print('False')",
    # ])
    translated_lines.extend([
        f"pos_side = s.check(question)",
        f"neg_side = s.check(Not(question))",
        "if pos_side == sat and neg_side == unsat:",
        "    return 'True'",
        "elif pos_side == unsat and neg_side == sat:",
        "    return 'False'",
        "elif pos_side == sat and neg_side == sat:",
        "    return 'N/A'",
        "else:",
        "    return 'UNSAT'"
    ])

    code = "\n".join(translated_lines)
    function_wrap = "def solution():\n" + "\n".join(["    " + line for line in translated_lines])
        # translated_lines = ["from z3 import *"] + translated_lines
    def func():
        try:
            exec(function_wrap)
            return eval('solution()')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(function_wrap)
            return str(e)


    result = func_timeout(1, func)
    if return_code:
        return function_wrap, result
    else:
        return result


contexthub_prompt = """
Problem:
If the knight had either a magic sword or a faithful steed, then he was well-equipped for his quest to slay the dragon. However, if the knight was not well-equipped, then he was ill-prepared to face the formidable dragon. Given that it is false that the knight failed in his quest, what can be determined about whether the knight had a magic sword? (True, False, or N/A (undetermined)).

# solution in code:
def solution():
    # If the knight had either a magic sword or a faithful steed, then he was well-equipped for his quest to slay the dragon.
    Implies(Or(knight_had_a_magic_sword, knight_had_a_faithful_steed), knight_was_well_equipped)
    # However, if the knight was not well-equipped, then he was ill-prepared to face the formidable dragon.
    Implies(Not(knight_was_well_equipped), knight_was_ill_prepared)
    # Given that it is false that the knight failed in his quest
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
    # Did Emily expand her knowledge?
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

    # Given that the patient's depressive symptoms do not improve with treatment
    # This means mntidepressant medication should not be indicated as an appropriate treatment option
    Not(antidepressant_medication_indicated_as_appropriate_treatment)
    # what can be concluded about whether the patient is experiencing persistent low mood?
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
    # Does the information imply that Alice has no income?
    return Alice_has_no_income
""".strip()


def test_sat():
    gts = ["True", "False", "True", "False"]

    output_code = contexthub_prompt
    examples = output_code.split('\n\n\n\n')
    gts = ["N/A", "True", "False", "N/A"]

    for i, ex in enumerate(examples):
        ex = ex.split("def solution():")[1].strip()
        code, result = boolean_satlm_exec(ex, return_code=True)
        print(result, gts[i])
        if result != gts[i]:
            print(code)
            print(result, gts[i])
            print("ERROR")
            break

def single_test():
    # ouput_code = ["def solution():\n    # The patient does not have high blood pressure, is not overweight, and does not have diabetes.\n    Not(patient_has_high_blood_pressure)\n    Not(patient_is_overweight)\n    Not(patient_has_diabetes)\n    # However, the patient does have a sedentary lifestyle.\n    patient_has_sedentary_lifestyle\n    # If the patient is overweight and has a sedentary lifestyle, then the patient is at risk for heart disease.\n    Implies(And(patient_is_overweight, patient_has_sedentary_lifestyle), patient_at_risk_for_heart_disease)\n    # If the patient has diabetes or high blood pressure, then the patient needs to make lifestyle changes.\n    Implies(Or(patient_has_diabetes, patient_has_high_blood_pressure), patient_needs_to_make_lifestyle_changes)\n    # If the patient is at risk for heart disease and needs to make lifestyle changes, then the patient should be prescribed medication and undergo regular monitoring.\n    Implies(And(patient_at_risk_for_heart_disease, patient_needs_to_make_lifestyle_changes), patient_should_be_prescribed_medication_and_undergo_regular_monitoring)\n    # Based on the patient's health profile, should the patient be prescribed medication and undergo regular monitoring?\n    return patient_at_risk_for_heart_disease and patient_needs_to_make_lifestyle_changes"]23
    # ouput_code = ['def solution():\n    # jj is True\n    jj = True\n    # jmf is False\n    jmf = False\n    # wawm is True\n    wawm = True\n    # If jmf or wawm, then ak\n    Implies(Or(jmf, wawm), ak)\n    # ntr is True\n    ntr = True\n    # If jj or ntr, then jcqy\n    Implies(Or(jj, ntr), jcqy)\n    # If ak or jcqy, then kewef\n    Implies(Or(ak, jcqy), kewef)\n    # Deduce the result of kewef\n    return kewef\n\n# Since jj is True, then jcqy is True\n# Since wawm is True, then ak is True\n# Since ak is True, then kewef is True\n# So, the result of kewef is True']
    # ouput_code = ['def solution():\n    # In a statistical study, the sample size is greater than 30, but the population standard deviation is not known.\n    sample_size_greater_than_30 = True\n    population_std_deviation_unknown = True\n    # If the sample size is greater than 30 and the population standard deviation is known, then the z-test can be used.\n    Implies(And(sample_size_greater_than_30, population_std_deviation_known), z_test_can_be_used)\n    # The data in the study follows a normal distribution, but the sample is not a simple random sample.\n    data_follows_normal_distribution = True\n    sample_not_simple_random = True\n    # If the data follows a normal distribution and the sample is a simple random sample, then the t-test can be used.\n    Implies(And(data_follows_normal_distribution, sample_is_simple_random), t_test_can_be_used)\n    # If either the t-test can be used or the z-test can be used, then a parametric hypothesis test can be conducted.\n    Implies(Or(t_test_can_be_used, z_test_can_be_used), parametric_hypothesis_test_can_be_conducted)\n    # Can a parametric hypothesis test be conducted in this study?\n    return parametric_hypothesis_test_can_be_conducted']
    ouput_code = ['def solution():\n    # A survey was conducted with the data categorized based on age groups.\n    Survey_data_categorized_by_age_groups\n    # However, the survey data does not include an "Other" category.\n    Not(Survey_data_includes_an_Other_category)\n    # If the survey data is categorized by age and includes an "Other" category, then it would have both age group categories and an "Other" category.\n    Implies(And(Survey_data_categorized_by_age, Survey_data_includes_an_Other_category), Survey_data_has_age_group_categories_and_Other_category)\n    # The age groups captured in the survey are: Under 18, 18-24, 25-34, 35-44, 45-54, 55-64, and 65+.\n    Age_groups = [Under_18, 18_24, 25_34, 35_44, 45_54, 55_64, 65_plus]\n    # Also, the survey received more than 1000 responses.\n    Survey_received_more_than_1000_responses\n    # If the survey has granular age group categories or a large number of responses, then it has sufficient information to analyze the age demographics.\n    Implies(Or(Survey_data_has_age_group_categories, Survey_received_more_than_1000_responses), Survey_has_sufficient_age_demographic_info)\n    # If the survey data has both age and "Other" categories or sufficient age demographic information, then it allows categorizing the responses to gain insights into different age segments.\n    Implies(Or(Survey_data_has_age_group_categories_and_Other_category, Survey_has_sufficient_age_demographic_info), Survey_allows_categorizing_responses)\n    # Based on the information provided, can the survey responses be categorized to understand different age segments?\n    return Survey_allows_categorizing_responses']

    ouput_code = ouput_code[0]
    ex = ouput_code.split("def solution():")[1].strip()
    code, result = boolean_satlm_exec(ex, return_code=True)
    print(code)
    print(result)

if __name__=="__main__":
    single_test()
    # test_sat()
