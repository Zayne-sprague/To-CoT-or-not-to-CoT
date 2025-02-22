import re

from z3 import *
from func_timeout import func_timeout

# """
# Problem: Based on the premises, is the conclusion true, false, or unknown?

# Premises:
# All mind-reading is either brain reading or brain decoding. 
# All brain decoding that is mind-reading is extracting information from BOLD signals.
# No studies that are mind-reading and extract information from BOLD signals are without statistical pattern analysis. 
# Writing a novel is without statistical pattern analysis.
# If multivoxel (pattern) analysis is without statistical pattern analysis and a brain reading, then multivoxel (pattern) analysis is without statistical pattern analysis and brain decoding.
# Multivoxel (pattern) analysis is a type of mind-reading.

# Conclusion:
# Multivoxel (pattern) analysis is without statistical pattern analysis and writing a novel.

# # solution in code:
# def solution():
#     # All mind-reading is either brain reading or brain decoding.
#     ForAll([x], Implies(Mind_Reading(x), Or(Brain_Reading(x), Brain_Decoding(x))))
#     # All brain decoding that is mind-reading is exatracting information from BOLD signals.
#     ForAll([x], Implies(And(Brain_Decoding(x), Mind_Reading(x)), Extracting_Information_From_BOLD_Singnals(x)))
#     # No studies that are mind-reading and extract information from BOLD signals are without statistical pattern analysis.
#     Not(Exists([x], And(Mind_Reading(x), Extracting_Information_From_BOLD_Singnals(x), Without_Statistical_Pattern_Analysis(x))))
#     # Writing a novel is without statistical pattern analysis.
#     Without_Statistical_Pattern_Analysis(Writing_a_novel)
#     # If multivoxel (pattern) analysis is without statistical pattern analysis and a brain reading, then multivoxel (pattern) analysis is without statistical pattern analysis and brain decoding.
#     Implies(And(Without_Statistical_Pattern_Analysis(Multivoxel_Analysis), Brain_Reading(Multivoxel_Analysis)), And(Without_Statistical_Pattern_Analysis(Multivoxel_Analysis), Brain_Decoding(Multivoxel_Analysis)))
#     # Multivoxel (pattern) analysis is a type of mind-reading.
#     Mind_Reading(Multivoxel_Analysis)

#     # Multivoxel (pattern) analysis is without statistical pattern analysis and writing a novel.
#     return And(Without_Statistical_Pattern_Analysis(Multivoxel_Analysis), Writing_a_novel(Multivoxel_Analysis))
# """
folio_example_prompt = """
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


VAR_REGEX = r"[_a-zA-Z0-9]+[,)]"
FUNC_REGEX = r"[_a-zA-Z0-9]+[(]"

PREDEFIND_FUNCS = ["ForAll", "Exists", "And", "Or", "Not", "Implies"]
PREDEFIND_QUNT_VARS = ["x"]

def make_z3_enum_line(sort_name, members):
    return "{}, ({}) = EnumSort('{}', [{}])".format(
        sort_name,
        ", ".join([f"{n}" for n in members]),
        sort_name,
        ", ".join([f"'{n}'" for n in members])
    )

def extract_var_and_func(line):
    all_vars = re.findall(VAR_REGEX, line)
    all_funcs = re.findall(FUNC_REGEX, line)
    all_vars = [all_vars.rstrip(",)") for all_vars in all_vars]
    all_funcs = [all_funcs.rstrip("(") for all_funcs in all_funcs]
    return all_vars, all_funcs

def determine_func_n_args(code, func):
    start_pos = code.find(func + "(")
    end_pos = code.find(")", start_pos)
    num_args = code[start_pos+len(func)+1:end_pos].count(",") + 1
    return num_args
    

def fol_satlm_exec(code, return_code=False):
    lines = code.splitlines()
    lines = [l.lstrip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]

    assert lines[-1].startswith("return")
    result_line = lines[-1]
    lines = lines[:-1]

    vars = set()
    functions = set()

    for line in lines + [result_line[7:]]:
        line_vars, line_funcs = extract_var_and_func(line)
        vars.update(line_vars)
        functions.update(line_funcs)

    vars = [v for v in vars if v not in PREDEFIND_QUNT_VARS]
    functions = [f for f in functions if f not in PREDEFIND_FUNCS]

    func_n_args = {}
    for func in functions:
        func_n_args[func] = determine_func_n_args(code, func)
    functions = sorted(functions, key=lambda x: func_n_args[x])

    translated_lines = []
    translated_lines.append('z3._main_ctx = Context()')
    translated_lines.append(make_z3_enum_line("ThingsSort", vars))

    for func in functions:
        num_args = func_n_args[func]
        translated_lines.append("{} = Function('{}', {}, BoolSort())".format(func, func, ", ".join(["ThingsSort"]*num_args)))
    translated_lines.append("x = Const('x', ThingsSort)")
    translated_lines.append("precond = []")

    for line in lines:
        translated_lines.append("precond.append({})".format(line))

    translated_lines.append("s = Solver()")
    translated_lines.append("s.add(precond)")

    return_clause = result_line.split("return")[1].strip()
    translated_lines.append(f"question = {return_clause}")

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
            return str(e)

    result = func_timeout(1, func)
    if return_code:
        return function_wrap, result
    else:
        return result


def test_sat():
    output_code = folio_example_prompt
    examples = output_code.split('\n\n\n\n')
    # gts = ["N/A", "False", "True", "N/A"]
    gts = ["N/A", "True", "N/A"]

    for i, ex in enumerate(examples):
        ex = ex.split("def solution():")[1].strip()
        code, result = fol_satlm_exec(ex, return_code=True)
        print(result, gts[i])
        if result != gts[i]:
            print(code)
            print(result, gts[i])
            print("ERROR")
            break

def single_test():
    ouput_code = ["def solution():\n    # All people who eat salads regularly are very conscious about their health and eating habits.\n    ForAll([x], Implies(Eats_Salads_Regularly(x), Very_Conscious_About_Health_And_Eating_Habits(x)))\n    # All people who grew up in health-conscious childhood homes eat salads regularly.\n    ForAll([x], Implies(Grew_Up_In_Health-Conscious_Childhood_Homes(x), Eats_Salads_Regularly(x)))\n    # All people who fulfill their daily nutritional intakes grew up in health-conscious childhood homes.\n    ForAll([x], Implies(Fulfills_Daily_Nutritional_Intakes(x), Grew_Up_In_Health-Conscious_Childhood_Homes(x)))\n    # All people who disregard their physical well-being are not very conscious about their health and eating habits.\n    ForAll([x], Implies(Disregards_Physical_Well-being(x), Not(Very_Conscious_About_Health_And_Eating_Habits(x))))\n    # If people visit the gym at least once a day, then they always fulfill their daily nutritional intakes.\n    ForAll([x], Implies(Visits_Gym_At_Least_Once_A_Day(x), Fulfills_Daily_Nutritional_Intakes(x)))\n    # Taylor either grew up in a health-conscious childhood home and disregard her physical well-being, or she did neither.\n    Or(Grew_Up_In_Health-Conscious_Childhood_Homes(Taylor) And Disregards_Physical_Well-being(Taylor), Not(And(Grew_Up_In_Health-Conscious_Childhood_Homes(Taylor), Disregards_Physical_Well-being(Taylor))))\n\n    # Taylor visits the gym at least once a day.\n    return Visits_Gym_At_Least_Once_A_Day(Taylor)\n\nThis solution is unknown because it is not possible to determine whether Taylor visits the gym at least once a day based on the given premises. The premises only provide information about Taylor's health and eating habits, but do not mention her gym habits."]

    ouput_code = ouput_code[0]
    ex = ouput_code.split("def solution():")[1].strip()
    if "    return" in ex:
        # find the end of the return statement
        before_return, after_return = ex.split("    return", 1)
        ex = before_return + "    return" + after_return.split("\n")[0]

    print(ouput_code)
    code, result = fol_satlm_exec(ex, return_code=True)
    print(code)
    print(result)

if __name__=="__main__":
    single_test()
    # test_sat()
