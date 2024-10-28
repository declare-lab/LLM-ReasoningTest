class Template:
    def check_gsm8k_prompt(self, question, generated_answer, gold_answer):
        return f"""Under the context of the #question#, Check if the #generated answer# is correct by comparing it with the #gold answer#. #generated is correct if it contains the #gold answer#. If the expression in #generated answer# and #gold answer# are equivalent mathematically, it is also considered as correct. If correct, output True, otherwise output False. You should only output True or False
#question#:
{question}

#gold answer#:
{gold_answer}

#generated answer#:
{generated_answer}
"""

    def check_human_eval_prompt(
        self, instruction, counterfactual, generated_answer, gold_answer
    ):
        return f"""Under the context of the question, Check whether the #generated answer# is correct, #generated is correct if it contains the #gold answer#. If correct, output True, otherwise output False, You should only output True or False
#question#:
{instruction}
{counterfactual}

#gold answer#:
{gold_answer}

#generated answer#:
{generated_answer}
"""

    def check_gsm8k_answer(self, generated_answer, gold_answer):
        return f"""Check if the #generated answer# is correct by comparing it with the #gold answer#. You only need to compare the final answer. If the expression in #generated answer# and #gold answer# are equivalent mathematically, it is also considered as correct. If correct, output True, otherwise output False. You should only output True or False.

#gold answer#:
{gold_answer}

#generated answer#:
{generated_answer}
"""

    def testcase_input_prompt(self, perturbed_question, reference):
        return f"""Create input parameters for test cases that align with the following #coding requirement#. Focus solely on the input values for these test cases. The input parameters should encompass boundary conditions within the scope defined by the function's requirements specification, and avoid scenarios that fall outside of these requirements. You should generate three to eight test cases.
test_case1: [input parameters 1]
test_case2: [input parameters 2]

#coding requirement#: 
{perturbed_question}
#You can also choose from some references from below#:
{reference}
"""

    def testcase_output_prompt(self, answer, test_input):
        local_scope = {}
        answer = "from typing import *\n" + answer
        exec(answer, local_scope)

        parsed_cases = dict()
        test_input = test_input.encode("ascii", "ignore").decode()
        for line in test_input.strip().split("\n"):
            case_name, case_value = line.split(":", 1)
            try:
                test_case = eval(case_value.strip())
                parsed_cases[case_name] = test_case
            except Exception:
                pass

        output_string = ""
        for case_name, test_input in parsed_cases.items():
            try:
                output = local_scope["gold_solution"](*test_input)
                if isinstance(output, str):
                    output_string += f"{case_name}: '{output}'\n"
                else:
                    output_string += f"{case_name}: {output}\n"
            except Exception:
                pass
        return output_string

    def testcode_prompt(self, perturbed_question, test_input, test_output, generation):
        return f"""#Instruction for Python Code Assertion Generation:
Objective: Create assertion statements in Python to automatically test the extracted code from the provided #answer# using the given #testcase input# against the expected #testcase output#.
Steps:
1. Extract Python Code: Given #coding requirement#, Identify and extract the Python code within the #answer# section.
2. Formulate Assertion Statement: Construct an assertion statement in Python using the extracted code, the provided #testcase input#, and the expected #testcase output#. (If the there is no test_output for test_input, you should ignore that testcase.
Output Format: Your final output should be a Python code snippet formatted as follows:
```python
[extracted python functions]
assert [extracted python functions]([test case input1]) == [test case output1], "testcase 1"
assert [extracted python functions]([test case input2]) == [test case output2], "testcase 2"
...
```

#coding requirement#: {perturbed_question}

#answer#: {generation}

#testcase input#: {test_input}

#testcase output#: {test_output}
"""

    def gsm8k_cot_prompt(self, perturbed_question, model="default"):
        if model in ["gpt4", "chatgpt", "gemini", "o1", "gpt4o", "default"]:
            prompt = (
                """Solve the question step by step before give the final answer. Do not directly give the final answer.
Reasoning Step: [reasoning steps]
Answer: ###[final answer]}
"""
                + "\n"
                + perturbed_question
            )

        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
{perturbed_question}


### Response: Let's think step by step.

"""
        return prompt

    def human_eval_cot_prompt(self, instruction, perturbed_question, model="default"):
        if model in ["gpt4", "chatgpt", "gemini", "default", "o1", "4o"]:
            prompt = f"""Instruction:{instruction}. You should answer the question in a step by step manner and do not directly give final solution.\n
{perturbed_question}
Reasoning Step:[step-by-step solution]
Final Solution: [final solution or code]
"""
        elif model == "llama2":
            prompt = f"""Instruction:{instruction}.
{perturbed_question}
let's answer the question step by step.:
"""
        elif model == "codellama":
            prompt = f"""Source: system

  You should answer the question follow the instruction in a step by step manner and do not directly give final answer. <step> Source: user

  Instruction: {instruction}\n{perturbed_question}  <step> Source: assistant
   
"""
        return prompt
