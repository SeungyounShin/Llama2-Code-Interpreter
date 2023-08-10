import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import re
from typing import List, Dict

DATA_DIR = "gpt_data_gen"

B_CODE = "[CODE_START_TOK]"
E_CODE = "[/CODE_END_TOK]"

B_RESULT = "[RESULT_TOK]"
E_RESULT = "[/RESULT_TOK]"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

CODE_SYS_PROMPT_FOR_TRAIN = """
You are 'CodeLLama', an advanced Language Model assistant that can generate, execute, and evaluate code. 
Respond to user queries by providing code-based solutions and insights.
"""


def msg_to_code_result_tok_temp(msg: List[Dict]) -> str:
    full_str = f"{B_SYS}\n{CODE_SYS_PROMPT_FOR_TRAIN}\n{E_SYS}\n\n"

    for chat in msg:
        if chat["role"] == "system":
            continue
        if chat["role"].lower() == "user":
            chat["content"] = chat["content"].replace(
                "/home/seungyoun/llama_code_interpreter/", "./"
            )
            full_str += f"{B_INST}\n###User : {chat['content']}\n"
        elif chat["role"].lower() == "assistant":
            chat["content"] = chat["content"].replace(
                "/home/seungyoun/llama_code_interpreter/", "./"
            )

            # Replace the code block start and end markers using regex
            code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            chat["content"] = code_pattern.sub(
                r"[CODE_START_TOK]\n\1[/CODE_END_TOK]", chat["content"]
            )

            # Replace the result block start and end markers using regex
            result_pattern = re.compile(r"```RESULTS?\n(.*?)```", re.DOTALL)
            chat["content"] = result_pattern.sub(
                r"[RESULT_TOK]\n\1[/RESULT_TOK]", chat["content"]
            )

            full_str += f"\n###Assistant : {chat['content']}\n{E_INST}\n"

    full_str = full_str.replace("')()", "')")

    return full_str


def json_to_code_result_tok_temp(json_file_name: str = "425.json") -> str:
    file_rel_path = os.path.join(DATA_DIR, json_file_name)

    with open(file_rel_path, "r") as json_file:
        msg = json.load(json_file)

    full_str = msg_to_code_result_tok_temp(msg)

    return full_str


if __name__ == "__main__":
    print(json_to_code_result_tok_temp())
