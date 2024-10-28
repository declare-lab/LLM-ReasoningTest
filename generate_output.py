import pandas as pd
from execution import easy_check
from fire import Fire
from helper import split_test_cases
from template import Template
from tqdm import tqdm

from modeling import select_model


def gen_solution(filename: str, model_name="gpt4"):
    """
    filename: {gsm8k.csv, human_eval.csv}
    model_name: chatgpt gpt4...
    method: dp, cot, pot, consistency
    """
    print(locals())
    dataset_name = "gsm8k" if "gsm8k" in filename else "human_eval"

    df = pd.read_csv(filename)
    model = select_model(model_name)
    templ = Template()

    prompts = []

    for i, o in df.iterrows():
        if dataset_name == "gsm8k":
            counterfactual = o["counterfactual"]
            if counterfactual == "NA" or str(counterfactual) == "nan":
                prompts.append("NA")
                continue
            prompt = templ.gsm8k_cot_prompt(
                perturbed_question=counterfactual, model=model_name
            )

        elif dataset_name == "human_eval":
            counterfactual = o["counterfactual"]
            instruction = o["instruction"]
            if counterfactual == "NA" or str(counterfactual) == "nan":
                prompts.append("NA")
                continue
            if instruction == "Closed Question":
                instruction = (
                    "Generate a python function that fulfills the requirement below."
                )

            prompt = templ.human_eval_cot_prompt(
                instruction=instruction,
                perturbed_question=counterfactual,
                model=model_name,
            )
        prompts.append(prompt)

    outputs = model.generate_batch(prompts)

    df["output"] = outputs
    df.to_csv(filename, index=False)


def check_solution_more(filename: str, eval_model_name="4o"):
    """
    model_name: chatgpt gpt4...
    method: dp, cot, pot, consistency
    """
    print(locals())
    dataset_name = "gsm8k"

    df = pd.read_csv(filename)
    model = select_model(eval_model_name)
    templ = Template()

    prompts = []
    for i, o in df.iterrows():
        counterfactual = o["counterfactual"]
        generated = o["output"]
        gold = o["answer"]
        close_format = o["close_format"]
        if not close_format:
            prompts.append("NA")
            continue
        if str(counterfactual) == "nan":
            prompts.append("NA")
            continue
        prompt = templ.check_gsm8k_prompt(
            question=counterfactual, generated_answer=generated, gold_answer=gold
        )

        prompts.append(prompt)

    auto_labels = model.generate_batch(prompts)
    df["auto_label"] = auto_labels
    df.to_csv(filename, index=False)


def gen_testcode_core(filename: str = "human_eval.csv", model_name="4o"):
    """
    filename: human_eval.csv
    model_name: chatgpt gpt4...
    """
    print(locals())
    dataset_name = "human_eval"

    df = pd.read_csv(filename)
    model = select_model(model_name, temperature=0.0)
    templ = Template()

    prompts = []
    for i, o in df.iterrows():
        counterfactual = o["counterfactual"]
        instruction = o["instruction"]
        test_input = o["test_input"]
        test_output = o["test_output"]
        generation = o["output"]

        if (
            counterfactual == "NA"
            or str(counterfactual) == "nan"
            or instruction != "Closed Question"
            or str(test_output) == "nan"
        ):
            prompts.append("NA")
            continue

        prompt = templ.testcode_prompt(
            perturbed_question=counterfactual,
            test_input=test_input,
            test_output=test_output,
            generation=generation,
        )
        prompts.append(prompt)

    testcodes = model.generate_batch(prompts)
    df["testcode"] = testcodes
    df.to_csv(filename, index=False)


def run_testcode_core(filename: str = "human_eval.csv"):
    """
    filename: human_eval.csv
    """

    df = pd.read_csv(filename)

    labels = []
    passes = []
    for i, o in tqdm(df.iterrows(), total=len(df)):
        counterfactual = o["counterfactual"]
        instruction = o["instruction"]
        testcode = o["testcode"]

        if (
            counterfactual == "NA"
            or str(counterfactual) == "nan"
            or instruction != "Closed Question"
            or str(testcode) == "nan"
        ):
            labels.append("NA")
            passes.append("NA")
            continue

        testcode = testcode.replace("```python", "").replace("```", "")
        testcodes = split_test_cases(testcode)
        label = ""
        passed = True
        for testcode in testcodes:
            return_dict = easy_check(test_program=testcode, timeout=5)
            label += return_dict["result"] + "\n"
            if "fail" in return_dict["result"]:
                passed = False
        labels.append(label)
        passes.append(passed)

    df["label"] = labels
    df["passed"] = passes
    df["auto_label"] = passes
    df.to_csv(filename, index=False)


def check_solution_core(filename: str, eval_model_name="4o"):
    """
    model_name: chatgpt gpt4...
    method: dp, cot, pot, consistency
    """

    df = pd.read_csv(filename)
    model = select_model(eval_model_name)
    templ = Template()

    prompts = []
    for i, o in df.iterrows():
        counterfactual = o["counterfactual"]
        generated = o["output"]
        gold = o["answer"]
        instruction = o["instruction"]
        testcode = o["testcode"]
        if instruction == "Closed Question":
            instruction = (
                "Generate a python function that fulfills the requirement below."
            )

        if str(testcode) != "nan":
            prompts.append("NA")
            continue
        prompt = templ.check_human_eval_prompt(
            instruction=instruction,
            counterfactual=counterfactual,
            generated_answer=generated,
            gold_answer=gold,
        )

        prompts.append(prompt)

    passes = model.generate_batch(prompts)
    for i, item in enumerate(passes):
        if item == "NA":
            passes[i] = df.loc[i, "passed"]
    df["auto_label"] = passes
    df.to_csv(filename, index=False)


def math(filename, model_name):
    print(f"Generating Solution from {model_name}")
    gen_solution(filename, model_name)
    print("Evaluting using GPT4o")
    check_solution_more(filename)

def code(filename, model_name):
    print(f"Generating Solution from {model_name}")
    gen_solution(filename, model_name)
    print("Generating testcases")
    gen_testcode_core(filename)
    print("Running testcases")
    run_testcode_core(filename)
    # check_solution_core(filename)

if __name__ == "__main__":
    Fire()
