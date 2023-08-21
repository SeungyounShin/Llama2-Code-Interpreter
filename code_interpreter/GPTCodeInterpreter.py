import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Dict

# Get the path from environment variable
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)
from code_interpreter.JuypyterClient import JupyterNotebook
from code_interpreter.BaseCodeInterpreter import BaseCodeInterpreter
from utils.const import *
from colorama import init, Fore, Style
from rich.markdown import Markdown
import base64

import openai
from retrying import retry
import logging
from termcolor import colored

# load from key file
with open("./openai_api_key.txt") as f:
    OPENAI_API_KEY = key = f.read()
openai.api_key = OPENAI_API_KEY
from utils.cleaner import clean_error_msg


def save_dialog(dialog, base_path: str = f"{prj_root_path}/gpt_data_gen"):
    file_number = 0
    while True:
        # Construct the path
        file_name = f"{file_number}.json"
        full_path = os.path.join(base_path, file_name)

        # Check if the file already exists
        if not os.path.exists(full_path):
            # If not, save the file
            with open(full_path, "w") as f:
                json.dump(dialog, f)
            print(f"Dialog saved to {full_path}")
            break
        else:
            # If the file does exist, increment the file number and try again
            file_number += 1


class GPTCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, model="gpt-4"):
        self.model = model
        self.dialog = [
            {"role": "system", "content": CODE_INTERPRETER_SYSTEM_PROMPT},
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

        self.response = None

        assert os.path.isfile(
            "./openai_api_key.txt"
        ), "The openai_api_key.txt file could not be found. Please make sure it is in the same directory as this script, and that it contains your OpenAI API key."

        # load from key file
        with open("./openai_api_key.txt") as f:
            OPENAI_API_KEY = f.read()
        openai.api_key = OPENAI_API_KEY

        self.nb = JupyterNotebook()

    def get_response_content(self):
        if self.response:
            return self.response["choices"][0]["message"]["content"]
        else:
            self.logger.warning("Response is empty.")
            return None

    @retry(
        stop_max_attempt_number=7,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def ChatCompletion(self):
        try:
            self.response = openai.ChatCompletion.create(
                model=self.model, messages=self.dialog
            )
        except Exception as e:
            print(f"error while OPENAI api call {e}")

    def chat(self, user_message: str, VERBOSE: bool = False):
        self.dialog.append({"role": "user", "content": user_message})

        code_block_output = ""
        attempt = 0
        img_data = None

        if VERBOSE:
            print(
                "###User : " + Fore.BLUE + Style.BRIGHT + user_message + Style.RESET_ALL
            )
            print("\n###Assistant : ")
        while True:
            if attempt > 3:
                break

            self.ChatCompletion()

            generated_text = self.get_response_content()
            generated_code_blocks = self.extract_code_blocks(generated_text)

            if len(generated_code_blocks) > 0:
                # Find the position of the first code block in the last answer
                first_code_block_pos = (
                    generated_text.find(generated_code_blocks[0])
                    if generated_code_blocks
                    else -1
                )
                text_before_first_code_block = (
                    generated_text
                    if first_code_block_pos == -1
                    else generated_text[:first_code_block_pos]
                )
                if VERBOSE:
                    print(Fore.GREEN + text_before_first_code_block + Style.RESET_ALL)
                if VERBOSE:
                    print(
                        Fore.YELLOW
                        + generated_code_blocks[0]
                        + "\n```\n"
                        + Style.RESET_ALL
                    )
                code_block_output, error_flag = self.execute_code_and_return_output(
                    generated_code_blocks[0]
                )

                code_block_output = f"{code_block_output}"

                if code_block_output is not None:
                    code_block_output = code_block_output.strip()

                code_block_output_str = f"\n```RESULTS\n{code_block_output}\n```\n"
                if VERBOSE:
                    print(Fore.LIGHTBLACK_EX + code_block_output_str + Style.RESET_ALL)
                    # markdown = Markdown(code_block_output_str)print(markdown)

                gen_final = f"{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}"

                if self.dialog[-1]["role"] == "user":
                    self.dialog.append({"role": "assistant", "content": gen_final})
                elif self.dialog[-1]["role"] == "assistant":
                    self.dialog[-1]["content"] += gen_final

                if not error_flag:
                    # attempt = max(attempt,2)
                    break
                    # return self.dialog[-1]
            else:
                if self.dialog[-1]["role"] == "user":
                    self.dialog.append({"role": "assistant", "content": generated_text})
                else:
                    self.dialog[-1]["content"] += generated_text
                # no code found break
                if VERBOSE:
                    print(Fore.GREEN + generated_text + Style.RESET_ALL)
                break

            # early stop
            if DEFAULT_EOS_TOKEN in self.dialog[-1]["content"]:
                if img_data is not None:
                    return (
                        f"{self.dialog[-1]}\n![plot](data:image/png;base64,{img_data})"
                    )
                return self.dialog[-1]

            attempt += 1
            # print(f"====Attempt[{attempt}]====\n{self.dialog[-1]['content']}")

        # print(self.dialog)
        if img_data is not None:
            return f"{self.dialog[-1]}\n![plot](data:image/png;base64,{img_data})"
        return self.dialog[-1]


if __name__ == "__main__":
    import random

    # output = interpreter.chat(user_message='How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?',
    #                          VERBOSE=True)
    # $print('--OUT--')
    # print(output['content'])
    question = "I'm thinking of a number between 1 and 100. Write a program that tries to guess it based on my hints (higher/lower)."

    interpreter = GPTCodeInterpreter()

    output = interpreter.chat(user_message=question, VERBOSE=True)

    print("=" * 10)
    # print(interpreter.dialog)

    # Save the dialog when 's' is pressed
    while True:
        input_char = input("Press 'q' to exit the dialog: ")
        if input_char.lower() == "q":
            save_dialog(interpreter.dialog)
            break

        else:
            output = interpreter.chat(user_message=input_char, VERBOSE=True)
