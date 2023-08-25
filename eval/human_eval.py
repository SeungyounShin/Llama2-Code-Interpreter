import os, sys

HUMAN_EVAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "human-eval",
)

sys.path.append(HUMAN_EVAL_PATH)
from human_eval.data import write_jsonl, read_problems
from finetuning.conversation_template import msg_to_code_result_tok_temp
from code_interpreter.llama_hf import build_model_from_hf_path
from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter

import re

from rich import print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from timeout_decorator import timeout


def extract_text(prompt, remove_lines=True):
    token = '"""'
    start = token
    end = ">>>"

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


def get_function_name(function_str):
    # Regular expression to match a function definition
    match = re.search(r"def (\w+)", function_str)
    if match:
        return match.group(1)
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

example1 = """
from typing import List

def max_subarray(nums: List[int]) -> int:
    \"""
    Find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
    
    >>> max_subarray([-2,1,-3,4,-1,2,1,-5,4])
    6
    # Explanation: The contiguous subarray [4,-1,2,1] has the largest sum = 6.
    
    >>> max_subarray([1])
    1
    
    >>> max_subarray([-1,-2])
    -1
    \"""
"""

few_shot_1 = [
    {
        "role": "user",
        "content": f"Given the function definition:\n{example1}\nPlease provide the best implementation for it.",
    },
    {
        "role": "assistant",
        "content": """
Let me provide you with a solution using Kadane's algorithm, which is an efficient O(n) approach to solve the Maximum Subarray Problem.
Here's the solution in the desired format:

[CODE_START_TOK]
def max_subarray(nums: List[int]) -> int:
    max_current = max_global = nums[0]
    for i in range(1, len(nums)):
        max_current = max(nums[i], max_current + nums[i])
        if max_current > max_global:
            max_global = max_current
    return max_global

assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([-1,-2]) == -1
[/CODE_END_TOK]
[RESULT_TOK]

[/RESULT_TOK]

The solution passed all the test cases without any assertion errors, which means the function works as expected for the provided test cases.
""",
    },
]

example2 = """
def is_three(nums: List[int]) -> bool:
    \"""
    Check sum of given list is 3 if 3 return True else Flase
    >>> is_three([1,2])
    True
    >>> is_three([-1,0,4,2])
    False
    \"""
"""

few_shot_2 = [
    {
        "role": "user",
        "content": f"Given the function definition:\n{example2}\nPlease provide the best implementation for it.",
    },
    {
        "role": "assistant",
        "content": """

[CODE_START_TOK]
def is_three(nums: List[int]) -> bool:
    return sum(nums) == 3

print(is_three([1,2])) # expected to be True
print(is_three([-1,0,4,2])) # expected to be False
[/CODE_END_TOK]
[RESULT_TOK]
True
False
[/RESULT_TOK]

The solution passed all the test cases, which means the function works as expected for the provided test cases.
""",
    },
]

example3 = """
def reverse_string(x : str) -> str:
    \"""
    Print the given string in reverse order
    >>> reverse_string('abc')
    cba
    >>> reverse_string('Hello world!')
    !dlrow olleH
    \"""
"""

few_shot_3 = [
    {
        "role": "user",
        "content": f"Given the function definition:\n{example3}\nPlease provide the best implementation for it.",
    },
    {
        "role": "assistant",
        "content": """

[CODE_START_TOK]
def reverse_string(x : str) -> str:
    return x[::-1]

print(reverse_string('abc')) # expect 'cba'
print(reverse_string('Hello world!')) # expect '!dlrow olleH'
[/CODE_END_TOK]
[RESULT_TOK]
cba
!dlrow olleH
[/RESULT_TOK]

The solution passed all the test cases, which means the function works as expected for the provided test cases.
""",
    },
]

example4 = """
def list2str(x : List[str]) -> str:
    \"""
    Make string from given list
    >>> list2str(['a','b','c'])
    a b c
    >>> list2str(['my name', 'is', 'john'])
    my name is john
    \"""
"""

few_shot_4 = [
    {
        "role": "user",
        "content": f"Given the function definition:\n{example4}\nPlease provide the best implementation for it.",
    },
    {
        "role": "assistant",
        "content": """
The challenge entails ensuring a match between characters within the list, while also inserting spaces in between.

[CODE_START_TOK]
def list2str(x : List[str]) -> str:
    return ' '.join(x)

print(list2str(['my name', 'is', 'john'])) # expect 'a b c'
print(list2str(['a','b','c'])) # expect 'my name is john'
[/CODE_END_TOK]
[RESULT_TOK]
a b c
my name is john
[/RESULT_TOK]

The solution passed all the test cases, which means the function works as expected for the provided test cases.
""",
    },
]

from timeout_decorator import timeout

wrong = 0


@timeout(10, timeout_exception=TimeoutError)
def exec_with_timeout(import_str, full_test_code):
    try:
        exec(f"{import_str}\n{full_test_code}")
    except Exception as e:
        return False  # Return False if there's an error during execution
    return True  # Return True if executed without errors


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter
    import argparse

    parser = argparse.ArgumentParser(description="Process path for LLAMA2_FINETUNEED.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the finetuned LLAMA2 model.",
        default='"./output/llama-2-7b-chat-ci"',
    )
    args = parser.parse_args()
    LLAMA2_FINETUNEED_PATH = args.path
    PROGRAMMING_PUZZLE_Q = False

    interpreter = LlamaCodeInterpreter(
        model_path=LLAMA2_FINETUNEED_PATH,
        # load_in_4bit=True
    )

    problems = read_problems()
    correct = 0
    total_problems = len(problems)

    for idx, task_id in enumerate(problems):
        # dict_keys(['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test'])
        programming_puzzle = problems[task_id]["prompt"]
        text_only_problem = extract_text(programming_puzzle)

        interpreter.dialog = [
            {
                "role": "system",
                "content": "You are helpful robot that can generate code , excute it and debug then answer",
            }
        ]  # this will replaced in template conversion
        # interpreter.dialog += few_shot_1
        # interpreter.dialog += few_shot_2
        # interpreter.dialog += few_shot_3
        # interpreter.dialog += few_shot_4

        if PROGRAMMING_PUZZLE_Q:
            # programming puzzle
            output_str = interpreter.chat(
                user_message=f"Given the function definition:\n{programming_puzzle}\nPlease provide the best implementation for it.",
                MAX_TRY=6,
                VERBOSE=False,
            )["content"]

        else:
            output_str = interpreter.chat(
                user_message=f"Create a Python script for this problem:\n{text_only_problem}",
                MAX_TRY=6,
                VERBOSE=False,
            )["content"]

        function_str = ""
        code_block = extract_all_code_block(output_str)
        if (code_block is not None) and ("def" in code_block):
            if PROGRAMMING_PUZZLE_Q:
                function_str = extract_function_from_code_block(code_block)
            else:
                function_str = code_block

        function_name = get_function_name(function_str)
        full_test_code = (
            f"{function_str}\n{problems[task_id]['test']}\ncheck({function_name})"
        )

        # Print the full_test_code with syntax highlighting
        syntax = Syntax(
            # f"{programming_puzzle}\n{full_test_code}",
            f"{full_test_code}",
            "python",
            theme="monokai",
            line_numbers=True,
        )
        print(syntax)

        wrong_flag = False
        timeout_flag = False
        try:
            wrong_flag = exec_with_timeout(import_str, full_test_code)
        except TimeoutError:
            wrong += 1
            timeout_flag = True
            print("Error: Code execution timed out after 10 seconds.")

        if wrong_flag or timeout_flag:
            # not timeout but got wrong answer
            wrong += 1

        acc = ((idx + 1) - wrong) / (idx + 1)

        # Constructing the output
        accuracy_text = Text(
            f"Accuracy: {(idx+1)-wrong}/{idx+1}[{total_problems}] = {acc:.2%}",
            style="bold blue",
        )
        panel = Panel(accuracy_text, title="Results", border_style="green")
        print(panel)
