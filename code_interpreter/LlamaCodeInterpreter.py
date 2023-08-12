import sys
import os

prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)

from code_interpreter.JuypyterClient import JupyterNotebook
from code_interpreter.BaseCodeInterpreter import BaseCodeInterpreter
from utils.const import *

from typing import List, Literal, Optional, Tuple, TypedDict, Dict
from colorama import init, Fore, Style

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class LlamaCodeInterpreter(BaseCodeInterpreter):
    def __init__(
        self, model_path: str, load_in_8bit: bool = False, load_in_4bit: bool = False
    ):
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        # Add special token
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=self.tokenizer,
            model=self.model,
        )

        self.dialog = [
            {
                "role": "system",
                "content": CODE_INTERPRETER_SYSTEM_PROMPT + "\nUse code to answer",
            },
            # {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            # {"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

        self.nb = JupyterNotebook()

    def dialog_to_prompt(
        self, dialog: List[Dialog], SYS_PROMPT: str = ""
    ) -> torch.Tensor:
        """
        code borrowed from : https://github.com/facebookresearch/llama/blob/main/llama/generation.py
        """
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": SYS_PROMPT,
                }
            ] + dialog
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]

        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )

        # print(dialog[::2], dialog[1::2],)

        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        # assert (
        #    dialog[-1]["role"] == "user"
        # ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )

        return torch.tensor(dialog_tokens).unsqueeze(0)

    def hard_coded_eos_splitter(self):
        self.dialog[-1]["content"] = self.dialog[-1]["content"].split(
            DEFAULT_EOS_TOKEN
        )[0]

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
            dialog_tokens = self.dialog_to_prompt(dialog=self.dialog)

            gen_tokens = self.model.generate(
                dialog_tokens.cuda(),
                max_new_tokens=512,
                top_p=1.0,
                do_sample=True,
                use_cache=True,
            )

            generated_text_all = self.tokenizer.batch_decode(gen_tokens)[0]
            generated_text = self.tokenizer.batch_decode(
                gen_tokens[:, dialog_tokens.shape[1] :]
            )[0]

            last_answer = self.parse_last_answer(generated_text_all)

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
                self.hard_coded_eos_splitter()
                if img_data is not None:
                    return (
                        f"{self.dialog[-1]}\n![plot](data:image/png;base64,{img_data})"
                    )
                return self.dialog[-1]

            self.hard_coded_eos_splitter()

            attempt += 1
            # print(f"====Attempt[{attempt}]====\n{self.dialog[-1]['content']}")

        # print(self.dialog)
        if img_data is not None:
            return f"{self.dialog[-1]}\n![plot](data:image/png;base64,{img_data})"
        return self.dialog[-1]


if __name__ == "__main__":
    import random

    LLAMA2_MODEL_PATH = "./ckpt/llama-2-13b-chat"
    LLAMA2_MODEL_PATH = "meta-llama/Llama-2-70b-chat-hf"

    interpreter = LlamaCodeInterpreter(model_path=LLAMA2_MODEL_PATH, load_in_4bit=True)
    output = interpreter.chat(
        user_message=random.choice(
            ["who is current korean president?", "what is sin(20)?"]
        ),
        VERBOSE=True,
    )

    while True:
        input_char = input("Press 'q' to quit the dialog: ")
        if input_char.lower() == "q":
            break

        else:
            output = interpreter.chat(user_message=input_char, VERBOSE=True)
