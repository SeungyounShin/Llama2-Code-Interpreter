import os, sys
import traceback

HUMAN_EVAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "human-eval",
)

sys.path.append(HUMAN_EVAL_PATH)
from human_eval.data import write_jsonl, read_problems
from finetuning.conversation_template import msg_to_code_result_tok_temp
from code_interpreter.llama_hf import build_model_from_hf_path
from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter
from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from code_interpreter.RetrospectiveGPTCodeInterpreter import (
    RetrospectiveGPTCodeInterpreter,
)

import re

from rich import print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from timeout_decorator import timeout

wrong = 0


def extract_text(prompt, remove_lines=True):
    token = '"""'
    start = token
    end = ">>>"
    # end = '"""'

    start_idx = prompt.find(start) + len(start)
    end_idx = prompt.find(end)

    output = prompt[start_idx:end_idx]
    if remove_lines:
        output = output.replace("\n", " ")
    output = re.sub(r"\s+", " ", output).strip()

    return output


def extract_all_code_block(input_str: str) -> str:
    pattern = r"\[CODE_START_TOK\](.*?)\[/CODE_END_TOK\]"
    matches = re.findall(pattern, input_str, re.DOTALL)
    return "\n".join([match.strip() for match in matches]) if matches else None


def extract_all_code_block_gpt(input_str: str) -> str:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, input_str, re.DOTALL)

    return "\n".join([match.strip() for match in matches]) if matches else None


def delete_print_asser(code_text: str):
    lines = code_text.split("\n")
    new_lines = list()
    for i in lines:
        if i.strip().startswith("print("):
            continue
        new_lines.append(i)

    new_code_text = "\n".join(new_lines)
    return new_code_text


def extract_function_from_code_block(code_block: str) -> str:
    lines = code_block.split("\n")
    function_lines = []

    inside_function = False
    for line in lines:
        # Start extracting from function definition
        if line.startswith("def "):
            inside_function = True

        # If we are inside the function, append lines
        if inside_function:
            function_lines.append(line)

            # If we encounter an unindented line that isn't a comment and isn't the start of another function, stop.
            if (
                not line.startswith("    ")
                and not line.startswith("#")
                and not line.startswith("def ")
            ):
                break

    # Remove trailing comments or blank lines and the last line which caused the exit from the loop
    while function_lines and (
        function_lines[-1].strip() == ""
        or function_lines[-1].strip().startswith("#")
        or not function_lines[-1].startswith("    ")
    ):
        function_lines.pop()

    return "\n".join(function_lines)


def get_last_outermost_function_name(function_str):
    matches = re.findall(r"^def (\w+)", function_str, re.MULTILINE)
    if matches:
        return matches[-1]  # Return the last (outermost) function name
    return ""


def get_last_function_name(function_str):
    # Regular expression to match a function definition
    matches = re.findall(r"def (\w+)", function_str)
    if matches:
        return matches[-1]  # Return the last function name
    return ""


def get_outermost_function_name(function_str):
    matches = re.findall(r"^def (\w+)", function_str, re.MULTILINE)
    if matches:
        return matches[0]  # Return the first (outermost) function name
    return ""


def get_function_name(function_str):
    # Regular expression to match a function definition
    match = re.search(r"def (\w+)", function_str)
    if match:
        return match.group(0)
    return ""


def extract_test_assertion(test_func: str):
    test_cases = list()
    for i in test_func.split("\n"):
        if "assert" in i:
            test_cases.append(i.strip())

    return ("\n".join(test_cases)).strip()


import_str = """
import re
import math
from typing import List, Tuple, Optional
"""


@timeout(100, timeout_exception=TimeoutError)
def exec_with_timeout(import_str, full_test_code):
    env = {**locals()}
    code_to_exec = f"{import_str}\n{full_test_code}"
    try:
        exec(code_to_exec, env)
    except Exception as e:
        print(f"Error Type: {type(e).__name__}, Error Message: {e}")
        return False  # Return False if there's an error during execution
    return True  # Return True if executed without errors


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse

    parser = argparse.ArgumentParser(description="Process path for LLAMA2_FINETUNEED.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the finetuned LLAMA2 model.",
        default='"./output/llama-2-7b-chat-ci"',
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Path to the finetuned LLAMA2 model.",
        default='"./output/llama-2-7b-chat-ci"',
    )
    parser.add_argument(
        "--max-retry",
        type=int,
        required=False,
        help="Maximum number of retries.",
        default=5,  # You can set any default value you want here.
    )
    args = parser.parse_args()
    PROGRAMMING_PUZZLE_Q = True

    problems = read_problems()
    correct_total = 0
    total_problems = len(problems)

    for idx, task_id in enumerate(problems):
        if "gpt" not in args.model.lower():
            LLAMA2_FINETUNEED_PATH = args.path
            interpreter = LlamaCodeInterpreter(
                model_path=LLAMA2_FINETUNEED_PATH,
                # load_in_4bit=True
            )
        else:
            interpreter = RetrospectiveGPTCodeInterpreter(
                model=args.model,
            )

        # dict_keys(['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test'])
        programming_puzzle = problems[task_id]["prompt"].replace("    ", "\t")
        text_only_problem = extract_text(programming_puzzle)

        interpreter.dialog = [
            {
                "role": "system",
                "content": "You are helpful robot that can generate code , excute it and debug then answer",
            }
        ]

        if PROGRAMMING_PUZZLE_Q:
            # programming puzzle
            output_str = interpreter.chat(
                user_message=f"Write a Python script to solve the following problem:\n{programming_puzzle}\nEnsure the solution is verified by printing the expected output.",
                MAX_TRY=args.max_retry,
                VERBOSE=True,
                code_exec_prefix=f"\nfrom typing import List,Tuple\nimport math\n",
                feedback_prompt="Ensure the output matches the expected result, taking into account any corner cases. If discrepancies arise, pinpoint where you went wrong. Then, refine the code to achieve the desired outcome.",
                append_result=True,
            )["content"]

        else:
            output_str = interpreter.chat(
                user_message=f"Write a Python script for this problem:\n{text_only_problem}",
                MAX_TRY=args.max_retry,
                VERBOSE=True,
                code_exec_prefix=f"\nfrom typing import List,Tuple\nimport math\n",
                feedback_prompt="Ensure the output matches the expected result. If not tell where you got wrong, then refine the code to achieve the desired outcome.",
                append_result=True,
            )["content"]

        function_str = ""
        if "gpt" not in args.model.lower():
            code_block = extract_all_code_block(output_str)
        else:
            code_block = extract_all_code_block_gpt(output_str)
        if (code_block is not None) and ("def" in code_block):
            function_str = code_block

        # function_name = get_last_outermost_function_name(function_str)
        function_str = delete_print_asser(function_str)
        function_name = get_last_outermost_function_name(function_str)
        full_test_code = f"{function_str}\n#-----------\n{problems[task_id]['test']}\ncheck({function_name})"

        # Print the full_test_code with syntax highlighting
        syntax = Syntax(
            # f"{programming_puzzle}\n{full_test_code}",
            f"{full_test_code}",
            "python",
            theme="monokai",
            line_numbers=True,
        )
        print(syntax)

        is_correct = False  # default is wrong
        timeout_flag = False
        try:
            is_correct = exec_with_timeout(import_str, full_test_code)
        except TimeoutError as e:
            timeout_flag = True
            print(f"Timeout with error msg : {e}")

        if is_correct:
            correct_total += 1

        acc = (correct_total) / (idx + 1)
        # save dialog
        interpreter.save_dialog(
            path=f"./eval/gpt_humaneval_output/{task_id.replace('/','_')}_{is_correct}.json"
        )
        interpreter.close()
        del interpreter

        # Constructing the output
        accuracy_text = Text(
            f"Accuracy: {correct_total}/{idx+1}[{total_problems}] = {acc:.2%} [{is_correct}]",
            style="bold blue",
        )
        panel = Panel(accuracy_text, title="Results", border_style="green")
        print(panel)
