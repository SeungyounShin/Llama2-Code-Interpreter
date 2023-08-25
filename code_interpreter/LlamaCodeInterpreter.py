import sys
import os

prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)

from code_interpreter.JuypyterClient import JupyterNotebook
from code_interpreter.BaseCodeInterpreter import BaseCodeInterpreter
from utils.const import *

from typing import List, Literal, Optional, Tuple, TypedDict, Dict
from colorama import init, Fore, Style
import copy
import re

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from finetuning.conversation_template import msg_to_code_result_tok_temp
from utils.special_tok_llama2 import (
    B_CODE,
    E_CODE,
    B_RESULT,
    E_RESULT,
    B_INST,
    E_INST,
    B_SYS,
    E_SYS,
    DEFAULT_PAD_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    IGNORE_INDEX,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class LlamaCodeInterpreter(BaseCodeInterpreter):
    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        peft_model: Optional[str] = None,
    ):
        # build tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=False,
        )

        # Handle special tokens
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN  # 32000
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN  # 2
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN  # 1
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.add_tokens(
            [B_CODE, E_CODE, B_RESULT, E_RESULT, B_INST, E_INST, B_SYS, E_SYS],
            special_tokens=True,
        )

        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        if peft_model is not None:
            peft_model = PeftModel.from_pretrained(self.model, peft_model)

        self.model = self.model.eval()

        self.dialog = [
            {
                "role": "system",
                "content": CODE_INTERPRETER_SYSTEM_PROMPT + "\nUse code to answer",
            },
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

        self.nb = JupyterNotebook()
        self.MAX_CODE_OUTPUT_LENGTH = 3000
        out = self.nb.add_and_run(TOOLS_CODE)  # tool import

    def dialog_to_prompt(self, dialog: List[Dict]) -> str:
        full_str = msg_to_code_result_tok_temp(dialog)

        return full_str

    @torch.inference_mode()
    def generate(
        self,
        prompt: str = "[INST]\n###User : hi\n###Assistant :",
        max_new_tokens=512,
        do_sample: bool = True,
        use_cache: bool = True,
        top_p: float = 0.95,
        temperature: float = 0.1,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
    ) -> str:
        # Get the model and tokenizer, and tokenize the user text.

        input_prompt = copy.deepcopy(prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        input_tokens_shape = inputs["input_ids"].shape[-1]

        eos_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_EOS_TOKEN)
        e_code_token_id = self.tokenizer.convert_tokens_to_ids(E_CODE)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=[
                eos_token_id,
                e_code_token_id,
            ],  # Stop generation at either EOS or E_CODE token
        )[0]

        generated_tokens = output[input_tokens_shape:]
        generated_text = self.tokenizer.decode(generated_tokens)

        return generated_text

    def extract_code_blocks(self, prompt: str) -> Tuple[bool, str]:
        pattern = re.escape(B_CODE) + r"(.*?)" + re.escape(E_CODE)
        matches = re.findall(pattern, prompt, re.DOTALL)

        if matches:
            # Return the last matched code block
            return True, matches[-1].strip()
        else:
            return False, ""

    def clean_code_output(self, output: str) -> str:
        if self.MAX_CODE_OUTPUT_LENGTH < len(output):
            return (
                output[: self.MAX_CODE_OUTPUT_LENGTH // 5]
                + "...(skip)..."
                + output[-self.MAX_CODE_OUTPUT_LENGTH // 5 :]
            )

        return output

    def chat(self, user_message: str, VERBOSE: bool = False, MAX_TRY=5):
        self.dialog.append({"role": "user", "content": user_message})
        if VERBOSE:
            print(
                "###User : " + Fore.BLUE + Style.BRIGHT + user_message + Style.RESET_ALL
            )
            print("\n###Assistant : ")

        # setup
        HAS_CODE = False  # For now
        INST_END_TOK_FLAG = False
        full_generated_text = ""
        prompt = self.dialog_to_prompt(dialog=self.dialog)
        start_prompt = copy.deepcopy(prompt)
        prompt = f"{prompt} {E_INST}"

        generated_text = self.generate(prompt)
        full_generated_text += generated_text
        HAS_CODE, generated_code_block = self.extract_code_blocks(generated_text)

        attempt = 1
        while HAS_CODE:
            if attempt > MAX_TRY:
                break
            # if no code then doesn't have to execute it

            # replace unknown thing to none
            generated_code_block = generated_code_block.replace("<unk>_", "").replace(
                "<unk>", ""
            )

            code_block_output, error_flag = self.execute_code_and_return_output(
                f"{generated_code_block}"
            )
            code_block_output = self.clean_code_output(code_block_output)
            generated_text = (
                f"{generated_text}\n{B_RESULT}\n{code_block_output}\n{E_RESULT}\n"
            )
            full_generated_text += f"\n{B_RESULT}\n{code_block_output}\n{E_RESULT}\n"

            first_code_block_pos = (
                generated_text.find(generated_code_block)
                if generated_code_block
                else -1
            )
            text_before_first_code_block = (
                generated_text
                if first_code_block_pos == -1
                else generated_text[:first_code_block_pos]
            )
            if VERBOSE:
                print(Fore.GREEN + text_before_first_code_block + Style.RESET_ALL)
                print(Fore.GREEN + generated_code_block + Style.RESET_ALL)
                print(
                    Fore.YELLOW
                    + f"\n{B_RESULT}\n{code_block_output}\n{E_RESULT}\n"
                    + Style.RESET_ALL
                )

            prompt = f"{prompt} {E_INST}{generated_text}"
            generated_text = self.generate(prompt)
            HAS_CODE, generated_code_block = self.extract_code_blocks(generated_text)

            full_generated_text += generated_text

            attempt += 1

        if VERBOSE:
            print(Fore.GREEN + generated_text + Style.RESET_ALL)

        self.dialog.append(
            {
                "role": "assistant",
                "content": full_generated_text.replace("<unk>_", "")
                .replace("<unk>", "")
                .replace("</s>", ""),
            }
        )

        return self.dialog[-1]


if __name__ == "__main__":
    import random

    LLAMA2_MODEL_PATH = "./ckpt/llama-2-13b-chat"
    LLAMA2_MODEL_PATH = "meta-llama/Llama-2-70b-chat-hf"
    LLAMA2_FINETUNEED_PATH = "./output/llama-2-7b-chat-ci"

    interpreter = LlamaCodeInterpreter(
        model_path=LLAMA2_FINETUNEED_PATH, load_in_4bit=True
    )
    output = interpreter.chat(
        user_message=random.choice(
            [
                # "In a circle with center \( O \), \( AB \) is a chord such that the midpoint of \( AB \) is \( M \). A tangent at \( A \) intersects the extended segment \( OB \) at \( P \). If \( AM = 12 \) cm and \( MB = 12 \) cm, find the length of \( AP \)."
                # "A triangle \( ABC \) is inscribed in a circle (circumscribed). The sides \( AB \), \( BC \), and \( AC \) are tangent to the circle at points \( P \), \( Q \), and \( R \) respectively. If \( AP = 10 \) cm, \( BQ = 15 \) cm, and \( CR = 20 \) cm, find the radius of the circle.",
                # "Given an integer array nums, return the total number of contiguous subarrays that have a sum equal to 0.",
                "what is second largest city in japan?",
                # "Can you show me 120days chart of tesla from today to before 120?"
            ]
        ),
        VERBOSE=True,
    )

    while True:
        input_char = input("Press 'q' to quit the dialog: ")
        if input_char.lower() == "q":
            break

        else:
            output = interpreter.chat(user_message=input_char, VERBOSE=True)
