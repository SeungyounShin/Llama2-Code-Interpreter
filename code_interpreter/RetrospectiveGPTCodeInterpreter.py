import json
import os
import sys
import time
import copy
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Dict
import numpy as np
from tqdm import tqdm

# Get the path from environment variable
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)
from code_interpreter.JuypyterClient import JupyterNotebook
from code_interpreter.BaseCodeInterpreter import BaseCodeInterpreter
from utils.const import *
from prompt.gpt4_prompt import CODE_INTERPRETER_SYSTEM_PROMPT

# from prompt.gpt4_prompt import CODE_INTERPRETER_SYSTEM_PROMPT
from colorama import init, Fore, Style
from rich.markdown import Markdown
import base64

import openai
from retrying import retry
import requests
import logging
from termcolor import colored

# load from key file
with open("./openai_api_key.txt") as f:
    OPENAI_API_KEY = key = f.read()
openai.api_key = OPENAI_API_KEY
from utils.cleaner import clean_error_msg


def remove_string(s):
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:.*LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64\n"
    return re.sub(pattern, "", s)


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


def get_embedding(text, model="text-embedding-ada-002"):
    global counter
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",  # Make sure to replace with your OpenAI API key
        "Content-Type": "application/json",
    }
    payload = {"input": text, "model": model}

    response = requests.post(
        "https://api.openai.com/v1/embeddings", headers=headers, json=payload
    )

    if response.status_code == 429:  # Rate limit error
        retry_after = int(response.headers.get("retry-after", 60))
        print(f"Rate limited. Retrying in {retry_after} seconds...")
        time.sleep(retry_after)
        return get_embedding(text, model)
    elif response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}")

    return np.array(response.json()["data"][0]["embedding"])


class QueryRetrospect:
    def __init__(
        self,
        data_directory="./gpt_data_gen_retrospect/",
        embeddings_path="./gpt_data_gen_retrospect/embeddings.npy",
    ):
        self.data_directory = data_directory
        self.embeddings_path = embeddings_path
        self.data = []
        self.embeddings = []

        if os.path.exists(embeddings_path):
            print("++ Embedding Exists!")
            self.embeddings = np.load(embeddings_path)
            for fname in [i for i in os.listdir(data_directory) if i.endswith(".json")]:
                with open(
                    os.path.join(data_directory, fname),
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    self.data.append(json.load(f))
        else:
            only_files = [
                f
                for f in os.listdir(data_directory)
                if os.path.isfile(os.path.join(data_directory, f))
                and f.endswith(".json")
            ]

            for fname in tqdm(only_files):
                with open(
                    os.path.join(data_directory, fname), "r", encoding="cp1252"
                ) as f:
                    data_point = json.load(f)
                    self.data.append(data_point)
                    self.embeddings.append(
                        get_embedding(data_point["execution_result"])
                    )
            self.embeddings = np.array(self.embeddings)
            self.save_embeddings()
            print(f"++ Embedding Saved! {self.embeddings.shape}")

    def save_embeddings(self):
        np.save(self.embeddings_path, self.embeddings)

    def __call__(self, query, top_k=3, VERBOSE: bool = False):
        query_embedding = get_embedding(query)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.data[i]["retrospection"] for i in top_indices]


class QueryRetrospectAgent:
    def __init__(
        self,
        model="gpt-4",
        data_directory="./gpt_data_gen_retrospect/",
        embeddings_path="./gpt_data_gen_retrospect/embeddings.npy",
    ):
        self.data_directory = data_directory
        self.embeddings_path = embeddings_path
        self.data = []
        self.embeddings = []

        if os.path.exists(embeddings_path):
            print("++ Embedding Exists!")
            self.embeddings = np.load(embeddings_path)
            for fname in [i for i in os.listdir(data_directory) if i.endswith(".json")]:
                with open(
                    os.path.join(data_directory, fname),
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    self.data.append(json.load(f))
        else:
            only_files = [
                f
                for f in os.listdir(data_directory)
                if os.path.isfile(os.path.join(data_directory, f))
                and f.endswith(".json")
            ]

            for fname in tqdm(only_files):
                with open(
                    os.path.join(data_directory, fname), "r", encoding="cp1252"
                ) as f:
                    data_point = json.load(f)
                    self.data.append(data_point)
                    self.embeddings.append(
                        get_embedding(data_point["execution_result"])
                    )
            self.embeddings = np.array(self.embeddings)
            self.save_embeddings()
            print(f"++ Embedding Saved! {self.embeddings.shape}")

        self.model = model
        self.SYS = "You are RetrospectiveGPT"
        self.dialog = [
            # {"role": "system", "content":  CODE_INTERPRETER_SYSTEM_PROMPT },
            {"role": "system", "content": self.SYS},
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]
        self.response = ""

    def ChatCompletion(self):
        try:
            self.response = openai.ChatCompletion.create(
                model=self.model, messages=self.dialog, temperature=0.1, top_p=1.0
            )
        except Exception as e:
            print(f"error while OPENAI api call {e}")

    def save_embeddings(self):
        np.save(self.embeddings_path, self.embeddings)

    def __call__(
        self, query, top_k=3, current_context: str = "", VERBOSE: bool = False
    ):
        self.dialog = [
            {"role": "system", "content": self.SYS},
        ]
        query_embedding = get_embedding(query)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_i = top_indices[0]
        retrospection_list = [self.data[i]["retrospection"] for i in top_indices]

        retrospection_adaption_prompt = f"From prior experience\nYou first provide the code :\n{self.data[top_i]['first_round_code']}\
            Result : \n{self.data[top_i]['execution_result']}\n\
            So You tried agian to get : \n{self.data[top_i]['second_round_code']}\n\
            Here is current error :\n{current_context}"

        print(
            f"From prior experience\nYou first provide the code :\n{self.data[top_i]['first_round_code']}\
            Result : \n{self.data[top_i]['execution_result']}\n\
            So You tried agian to get : \n{self.data[top_i]['second_round_code']}\n\
            Here is current error :\n{current_context}"
        )
        self.dialog.append({"role": "user", "content": retrospection_adaption_prompt})
        self.ChatCompletion()

        return [self.response["choices"][0]["message"]["content"]]


class RetrospectiveGPTCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, model="gpt-4"):
        self.model = model
        self.dialog = [
            # {"role": "system", "content":  CODE_INTERPRETER_SYSTEM_PROMPT },
            {
                "role": "system",
                "content": CODE_INTERPRETER_SYSTEM_PROMPT,
            },
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

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

        # retrospections
        self.retrospector = QueryRetrospect()

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
                model=self.model, messages=self.dialog, temperature=0.2, top_p=0.9
            )
        except Exception as e:
            print(f"error while OPENAI api call {e}")

    def save_dialog(self, path: str = "./output/dialog.json"):
        with open(path, "w") as f:
            json.dump(self.dialog, f)
            print(f" ++Dialog saved to [{path}]")

    def close(self):
        """
        close jupyter notebook, and this class instance
        """
        self.nb.close()

    def chat(
        self,
        user_message: str,
        VERBOSE: bool = False,
        MAX_TRY: int = 6,
        code_exec_prefix: str = "",
        feedback_prompt: str = "",
        append_result: bool = True,
        use_retrospect: bool = True,
    ):
        self.dialog.append({"role": "user", "content": user_message})
        init_feedback = copy.deepcopy(feedback_prompt)

        code_block_output = ""
        attempt = 0
        img_data = None

        if VERBOSE:
            print(
                "###User : " + Fore.BLUE + Style.BRIGHT + user_message + Style.RESET_ALL
            )
            print("\n###Assistant : ")

        for i in range(MAX_TRY):
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
                if append_result:
                    gen_final = f"{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}"
                    if VERBOSE:
                        print(
                            Fore.LIGHTBLACK_EX + code_block_output_str + Style.RESET_ALL
                        )
                else:
                    gen_final = (
                        f"{text_before_first_code_block}{generated_code_blocks[0]}\n```"
                    )

                self.dialog.append(
                    {
                        "role": "assistant",
                        "content": gen_final,
                    }
                )

                if use_retrospect and error_flag:
                    retrospection = self.retrospector(
                        query=code_block_output,
                        # current_context=f"{generated_code_blocks[0]}\n{code_block_output}",
                    )[0]
                    feedback_prompt = f"From previous similar error retrosepct\n{retrospection}\nImprove the code."
                    if VERBOSE:
                        print(Fore.MAGENTA + feedback_prompt + Style.RESET_ALL)
                else:
                    if len(init_feedback) > 1:
                        feedback_prompt = f"{init_feedback}\nif you accomplish the instruction just say <done>\nIf not keep going."
                        if VERBOSE:
                            print(Fore.MAGENTA + feedback_prompt + Style.RESET_ALL)

                feedback_dict = {
                    "role": "user",
                    "content": feedback_prompt,
                }

                self.dialog.append(feedback_dict)

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

        self.dialog = [self.dialog[0]] + clean_the_dialog(
            self.dialog, question=user_message
        )  # delete retrospections after generation step

        return self.dialog[-1]


if __name__ == "__main__":
    import pickle
    import random
    from tqdm import tqdm

    # python3 -m code_interpreter.RetrospectiveGPTCodeInterpreter

    retro_interpreter = RetrospectiveGPTCodeInterpreter(model="gpt-4")

    instruction = "Plot the MACD indicator in tesla.\n ref code : tqqq['MACD'], tqqq['MACD Signal'], _ = ta.trend.macd(tqqq['Close'])"
    # instruction = "Can you make a image of astraunaut in the garden?"

    # example
    retro_interpreter.chat(
        user_message=instruction,
        MAX_TRY=5,
        use_retrospect=True,
        feedback_prompt="Verfiy the result is expected.",
        VERBOSE=True,
    )
