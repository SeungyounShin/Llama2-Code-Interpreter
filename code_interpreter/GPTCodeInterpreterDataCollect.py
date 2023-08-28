import json
import os, sys
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
from prompt.gpt4_prompt import *


def remove_string(s):
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:.*LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64\n"
    return re.sub(pattern, "", s)


def gen_questions(prefix="What is 55th fibonacci number?"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are teacherGPT, You need to generate only questions(to student not the explanation and solution) based on student history. \n\nGive him only one question.\n\nAlso remember that student can use code. ",
            },
            {
                "role": "user",
                "content": f"{prefix}\nmore harder one but not the similar domain of above.",
            },
        ],
        temperature=0.1,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["message"]["content"]


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


def clean_the_dialog(dialog, question):
    question_idx = 0
    for idx, item in enumerate(dialog):
        if item["content"] == question:
            question_idx = idx

    filtered_dialog = dialog[question_idx:]

    user_qinit_dict = filtered_dialog[0]
    answer_fuse_str = "\n".join([i["content"].strip() for i in filtered_dialog[1::2]])

    final_dialog_dict = [
        {"role": "user", "content": user_qinit_dict["content"]},
        {"role": "assistant", "content": answer_fuse_str},
    ]

    return final_dialog_dict


class GPTCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, model="gpt-4"):
        self.model = model
        self.dialog = [
            # {"role": "system", "content":  CODE_INTERPRETER_SYSTEM_PROMPT },
            {
                "role": "system",
                "content": CODE_INTERPRETER_SYSTEM_PROMPT + "\n" + extra_prompt,
            },
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

        self.dialog += few_shot_1
        # self.dialog += few_shot_4
        self.response = None

        assert os.path.isfile(
            "./openai_api_key.txt"
        ), "The openai_api_key.txt file could not be found. Please make sure it is in the same directory as this script, and that it contains your OpenAI API key."

        # load from key file
        with open("./openai_api_key.txt") as f:
            OPENAI_API_KEY = f.read()
        openai.api_key = OPENAI_API_KEY

        self.nb = JupyterNotebook()
        out = self.nb.add_and_run(TOOLS_CODE)  # tool import

    def get_response_content(self):
        if self.response:
            return self.response["choices"][0]["message"]["content"]
        else:
            return None

    @retry(
        stop_max_attempt_number=7,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def ChatCompletion(self):
        try:
            self.response = openai.ChatCompletion.create(
                model=self.model, messages=self.dialog, temperature=0.1, top_p=1.0
            )
        except Exception as e:
            print(f"error while OPENAI api call {e}")

    def chat(self, user_message: str, VERBOSE: bool = False, MAX_RETRY: int = 6):
        self.dialog.append({"role": "user", "content": user_message})

        code_block_output = ""
        attempt = 0
        img_data = None

        if VERBOSE:
            print(
                "###User : " + Fore.BLUE + Style.BRIGHT + user_message + Style.RESET_ALL
            )
            print("\n###Assistant : ")

        for i in range(MAX_RETRY):
            # GPT response
            self.ChatCompletion()

            # Get code block
            generated_text = self.get_response_content()
            generated_code_blocks = self.extract_code_blocks(generated_text)
            # execute code
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

                code_block_output = remove_string(code_block_output)
                if len(code_block_output) > 500:
                    code_block_output = (
                        code_block_output[:200] + "⋯(skip)⋯" + code_block_output[-200:]
                    )
                code_block_output_str = f"\n```RESULT\n{code_block_output}\n```\n"
                if VERBOSE:
                    print(Fore.LIGHTBLACK_EX + code_block_output_str + Style.RESET_ALL)
                    # markdown = Markdown(code_block_output_str)print(markdown)

                gen_final = f"{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}"

                self.dialog.append(
                    {
                        "role": "assistant",
                        "content": f"{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}",
                    }
                )

                self.dialog.append(
                    {
                        "role": "user",
                        "content": "Keep going. if you think debugging generate code. need conclusion to question only text (Do not leave result part alone). Doesn't need to generated anything then just say <done>",
                    }
                )

            else:
                if "<done>" in generated_text:
                    generated_text = generated_text.split("<done>")[0].strip()

                if len(generated_text) <= 0:
                    break

                if VERBOSE:
                    print(Fore.GREEN + generated_text + Style.RESET_ALL)

                self.dialog.append(
                    {
                        "role": "assistant",
                        "content": f"{generated_text}",
                    }
                )
                break

        return self.dialog[-1]


if __name__ == "__main__":
    import random

    SEED_TASK = [
        # "Resize this image to 512x512\nUser Uploaded File : './tmp/img.png'",
        "Write a Python script that retrieves Google Trends data for a given keyword and stock price data for a specific company over the same timeframe, normalizes both datasets to the same scale, and then plots them on the same graph to analyze potential correlations.",
        "Could you conduct a frequency analysis on Apple's stock price to determine any cyclic patterns that occur on a weekly, monthly, or quarterly basis?",
    ]

    questions = SEED_TASK

    from tqdm import tqdm

    for i in tqdm(range(150000)):
        interpreter = GPTCodeInterpreter()

        question = questions[i]
        output = interpreter.chat(user_message=question, VERBOSE=True, MAX_RETRY=5)

        sample = clean_the_dialog(interpreter.dialog, question)

        save_dialog(sample)

        # q1,q2,q3 = random.sample(questions, k=3)
        # question = gen_questions(prefix = f'{q1}\n{q2}\n{q3}')
        # questions.append(question)

        del interpreter

        print(f"new question :: {question}")
