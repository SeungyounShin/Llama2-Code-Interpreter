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
from rich.text import Text


def extract_last_code_block(input_str: str) -> str:
    pattern = r"\[CODE_START_TOK\](.*?)\[/CODE_END_TOK\]"
    matches = re.findall(pattern, input_str, re.DOTALL)
    return matches[-1].strip() if matches else None


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


def extract_test_assertion(test_func: str):
    test_cases = list()
    for i in test_func.split("\n"):
        if "assert" in i:
            test_cases.append(i.strip())

    return ("\n".join(test_cases)).strip()


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

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter

    LLAMA2_FINETUNEED_PATH = "./output/llama-2-7b-chat-ci"

    interpreter = LlamaCodeInterpreter(
        model_path=LLAMA2_FINETUNEED_PATH,
        # load_in_4bit=True
    )

    problems = read_problems()
    wrong = 0
    correct = 0

    for idx, task_id in enumerate(problems):
        # dict_keys(['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test'])
        programming_puzzle = problems[task_id]["prompt"]

        interpreter.dialog = [
            {
                "role": "system",
                "content": "You are helpful robot that can generate code , excute it and debug then answer",
            }
        ]  # this will replaced in template conversion
        interpreter.dialog += few_shot_1
        # interpreter.dialog += few_shot_2
        # interpreter.dialog += few_shot_3

        output_str = interpreter.chat(
            user_message=f"Given the function definition:\n{programming_puzzle}\nPlease provide the best implementation for it.",
            MAX_TRY=6,
            VERBOSE=True,
        )["content"]

        function_str = ""
        code_block = extract_last_code_block(output_str)
        if (code_block is not None) and ("def" in code_block):
            function_str = extract_function_from_code_block(code_block)
        test_cases_str = extract_test_assertion(problems[task_id]["test"])

        full_test_code = f"{function_str}\n{test_cases_str}"

        try:
            exec(full_test_code)
        except:
            wrong += 1

        acc = ((idx + 1) - wrong) / (idx + 1)

        # Constructing the output
        accuracy_text = Text(
            f"Accuracy: {(idx+1)-wrong}/{idx+1} = {acc:.2%}", style="bold blue"
        )
        panel = Panel(accuracy_text, title="Results", border_style="green")
        print(panel)
