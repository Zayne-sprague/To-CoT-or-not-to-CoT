import re

from z3 import *
from func_timeout import func_timeout


VAR_REGEX = r"[_a-zA-Z0-9]+[,)]"
FUNC_REGEX = r"[_a-zA-Z0-9]+[(]"

PREDEFIND_FUNCS = ["ForAll", "Exists", "And", "Or", "Not", "Implies"]
PREDEFIND_QUNT_VARS = ["x", "y"]

def make_z3_declare_line(sort_name, members):
    if len(members) == 1:
        return "{} = Const('{}', {})".format(
            members[0],
            members[0],
            sort_name
        )
    return "{} = Consts('{}', {})".format(
        ", ".join([f"{n}" for n in members]),
        " ".join([f"{n}" for n in members]),
        sort_name
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


# small LLMs cannot produce well formed specs just by learning from examples
# fix some common syntax errors
def adhoc_syntax_fix(line):
    line =  ''.join(char for char in line if ord(char) < 128)
    # fix some problemmatic literal in var
    line = line.replace("-", "_")
    line = line.replace("'", "")
    line = line.replace('"', "")
    line = line.replace(".", "")

    line = line.replace(" And ", " & ")
    line = line.replace(" and ", " & ")
    line = line.replace(" Or ", " | ")
    line = line.replace(" or ", " | ")

    # completing unclosed brackets
    if line.count("(") > line.count(")"):
        line += ")" * (line.count("(") - line.count(")"))

    line = line.replace(" = ", " == ")

    vars = set(extract_var_and_func(line)[0])
    for var in vars:
        # if var is a number
        if var[0].isdigit():
            line = line.replace(var, f"num{var}")

    return line

def fol_uninterp_satlm_exec(code, return_code=False):
    lines = code.splitlines()
    lines = [l.lstrip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]
    lines = [l.split("#")[0].rstrip() for l in lines]
    # merge multi-line statements
    _lines = []
    ending = True
    for line in lines:
        if not ending:
            _lines[-1] += line.strip()
        else:
            _lines.append(line)
        if line.endswith(","):
            ending = False
        else:
            ending = True
    lines = _lines
    lines = [adhoc_syntax_fix(l) for l in lines]

    # cut everything after return
    return_line = next((i for i, l in enumerate(lines) if "return " in l), None)
    if return_line is not None:
        lines = lines[:return_line+1]

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
    translated_lines.append("ThingsSort = DeclareSort('ThingsSort')")
    if vars:
        translated_lines.append(make_z3_declare_line("ThingsSort", vars))

    for func in functions:
        num_args = func_n_args[func]
        translated_lines.append("{} = Function('{}', {}, BoolSort())".format(func, func, ", ".join(["ThingsSort"]*num_args)))
    translated_lines.append("x, y = Consts('x y', ThingsSort)")
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
    with open("folio_prompt.py", "r") as f:
        output_code = f.read().strip()
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
    # ouput_code = ["def solution():\n    # It costs $205 to take the GRE test, which is cheaper than $300.\n    Less_Than(300, 205)\n    # ETS provides financial aid to those GRE applicants who prove economic hardship.\n    ForAll([x], Implies(Proves_Economic_Hardship(x), Provides_Financial_Aid(x, ETS)))\n    # Those living in single-parent families or having few resources available to them can prove economic hardship.\n    ForAll([x], Implies(Or(Lives_In_Single_Parent_Family(x), Has_Few_Resources(x)), Proves_Economic_Hardship(x)))\n    # Tom lives in a single-parent family.\n    Lives_In_Single_Parent_Family(Tom)\n    # Tom's dad has been out of work, and Tom has few resources available to them.\n    Has_Few_Resources(Tom)\n    # Tom is applying to take the GRE test.\n    Applying_To_Take_GRE(Tom)\n\n    # ETS provides financial aid to Tom.\n    return Provides_Financial_Aid(Tom, ETS)\n\nThe conclusion is TRUE."]
    # ouput_code = ["def solution():\n    # All of Zaha Hadid's design styles that Max adores have interesting geometries.\n    ForAll([x], Implies(Is_Zaha_Hadid(x), And(Is_Design_Style(x), Adores(Max, x), Has_Interesting_Geometries(x))))\n    # No brutalist buildings that Max adores have interesting geometries.\n    ForAll([x], Implies(Is_Brutalist(x) And Adores(Max, x), Not(Has_Interesting_Geometries(x))))\n    # Every style that Max adores is either Zaha Hadid's design style or Kelly Wearstler's design style.\n    ForAll([x], Implies(Adores(Max, x), Or(Is_Zaha_Hadid(x), Is_Kelly_Wearstler(x))))\n    # All of Kelly Wearstler's design styles that Max adores are evocative.\n    ForAll([x], Implies(Is_Kelly_Wearstler(x) And Adores(Max, x), Is_Evocative(x)))\n    # All of Kelly Wearstler's design styles that Max adores are dreamy.\n    ForAll([x], Implies(Is_Kelly_Wearstler(x) And Adores(Max, x), Is_Dreamy(x)))\n    # If a design by Max that he adores has interesting geometries, then the design is a brutalist building and evocative.\n    ForAll([x], Implies(And(Is_Design_By_Max(x), Adores(Max, x), Has_Interesting_Geometries(x)), And(Is_Brutalist(x), Is_Evocative(x))))\n\n    # A design by Max is a brutalist building.\n    return Exists([x], And(Is_Design_By_Max(x), Is_Brutalist(x)))\n\nThe conclusion is true."]
    # ouput_code = ['def solution():\n    # A Japanese game company created the game the Legend of Zelda.\n    Created_By(The_Legend_of_Zelda, Japanese_Game_Company)\n    # All games on the Top 10 list are made by Japanese game companies.\n    ForAll([x], Implies(Is_On_Top_10_List(x), Made_By(x, Japanese_Game_Company)))\n    # If a game sells more than one million copies, then it will be included in the Top 10 list.\n    ForAll([x], Implies(Sold_More_Than_One_Million_Copies(x), Is_On_Top_10_List(x)))\n    # The Legend of Zelda sold more than one million copies.\n    Sold_More_Than_One_Million_Copies(The_Legend_of_Zelda)\n\n    # The Legend of Zelda is on the Top 10 list.\n    return Exists([x], And(The_Legend_of_Zelda, Is_On_Top_10_List(x)))\n\n# Since the conclusion is the opposite of what the premises suggest, the conclusion is:\nprint("False")']
    # ouput_code = ['```python\ndef solution():\n    # All people who eat salads regularly are very conscious about their health and eating habits.\n    ForAll([x], Implies(Eats_Salads_Regularly(x), Very_Conscious_About_Health(x)))\n    # All people who grew up in health-conscious childhood homes e-p0olokm at salads regularly.\n    ForAll([x], Implies(Grew_Up_In_Health_Conscious_Home(x), Eats_Salads_Regularly(x)))\n    # All people who fulfill their daily nutritional intakes grew up in health-conscious childhood homes.\n    ForAll([x], Implies(Fulfills_Daily_Nutritional_Intakes(x), Grew_Up_In_Health_Conscious_Home(x)))\n    # All people who disregard their physical well-being are not very conscious about their health and eating habits.\n    ForAll([x], Implies(Disregards_Physical_Well_Being(x), Not(Very_Conscious_About_Health(x))))\n    # If people visit the gym at least once a day, then they always fulfill their daily nutritional intakes.\n    ForAll([x], Implies(Visits_Gym_At_Least_Once_A_Day(x), Fulfills_Daily_Nutritional_Intakes(x)))\n    # Taylor either grew up in a health-conscious childhood home and disregard her physical well-being, or she did neither.\n    Or(And(Grew_Up_In_Health_Conscious_Home(Taylor), Disregards_Physical_Well_Being(Taylor)), \n       And(Not(Grew_Up_In_Health_Conscious_Home(Taylor)), Not(Disregards_Physical_Well_Being(Taylor))))\n\n    # Taylor eats salads regularly.\n    return Eats_Salads_Regularly(Taylor)\n```']
    ouput_code = ['```python\ndef solution():\n    # Billings is a city in the state of Montana in U.S.\n    And(Is_City(Billings), In_State(Billings, Montana))\n    # The state of Montana includes the cities of Butte, Helena, and Missoula.\n    And(In_State(Butte, Montana), In_State(Helena, Montana), In_State(Missoula, Montana))\n    # White Sulphur Springs and Butte are cities in the same state in U.S.\n    And(In_State(White_Sulphur_Springs, State_X), In_State(Butte, State_X))\n    # The city of St Pierre is not in the state of Montana.\n    Not(In_State(St_Pierre, Montana))\n    # Any city in Butte is not in St Pierre.\n    ForAll([x], Implies(In_City(x, Butte), Not(In_City(x, St_Pierre))))\n    # A city can only be in one state in U.S. except for Bristol, Texarkana, Texhoma and Union City.\n    ForAll([x], Implies(Is_City(x), Or(Exists([y], In_State(x, y)), x in [Bristol, Texarkana, Texhoma, Union_City]))))\n\n    # Montana is home to the city of Missoula.\n    return In_State(Missoula, Montana)\n```']




    ouput_code = ouput_code[0]
    ex = ouput_code.split("def solution():")[1].strip()
    # if "    return" in ex:
    #     # find the end of the return statement
    #     before_return, after_return = ex.split("    return", 1)
    #     ex = before_return + "    return" + after_return.split("\n")[0]

    # print(ouput_code)
    code, result = fol_uninterp_satlm_exec(ex, return_code=True)
    print('from z3 import *')
    print(code)
    print('print(solution())')
    # print(result)

if __name__=="__main__":
    single_test()
    # test_sat()
