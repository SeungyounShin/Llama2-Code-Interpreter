import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Dict

prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

import nbformat

# from nbconvert.preprocessors import ExecutePreprocessor
# from nbconvert.preprocessors.execute import CellExecutionError

from utils.const import *
from utils.cleaner import clean_error_msg
from colorama import init, Fore, Style
from rich.markdown import Markdown
import base64

import openai
from retrying import retry
import logging
from termcolor import colored
from code_interpreter.JuypyterClient import JupyterNotebook


class BaseCodeInterpreter:
    def __init__(self):
        self.dialog = [
            {
                "role": "system",
                "content": CODE_INTERPRETER_SYSTEM_PROMPT,
            },
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

        self.nb = JupyterNotebook()

    @staticmethod
    def extract_code_blocks(text: str):
        pattern = r"```(?:python\n)?(.*?)```"  # Match optional 'python\n' but don't capture it
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return [block.strip() for block in code_blocks]

    @staticmethod
    def parse_last_answer(text: str) -> str:
        return text.split(E_INST)[-1]

    def execute_code_and_return_output(self, code_str: str):
        outputs, error_flag = self.nb.add_and_run(code_str)
        return outputs, error_flag
