import gradio as gr
import random
import time, os
import copy
import re

import torch
from rich.console import Console
from rich.table import Table
from datetime import datetime

from threading import Thread
from typing import Optional
from transformers import TextIteratorStreamer

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

from finetuning.conversation_template import (
    json_to_code_result_tok_temp,
    msg_to_code_result_tok_temp,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter


class StreamingLlamaCodeInterpreter(LlamaCodeInterpreter):
    streamer: Optional[TextIteratorStreamer] = None

    # overwirte generate function
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

        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, Timeout=5
        )

        input_prompt = copy.deepcopy(prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        input_tokens_shape = inputs["input_ids"].shape[-1]

        eos_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_EOS_TOKEN)
        e_code_token_id = self.tokenizer.convert_tokens_to_ids(E_CODE)

        kwargs = dict(
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
            streamer=self.streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        return ""


def change_markdown_image(text: str):
    modified_text = re.sub(r"!\[(.*?)\]\(\'(.*?)\'\)", r"![\1](/file=\2)", text)
    return modified_text


def gradio_launch(model_path: str, load_in_4bit: bool = True, MAX_TRY: int = 5):
    with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
        chatbot = gr.Chatbot(height=820, avatar_images="./assets/logo2.png")
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        interpreter = StreamingLlamaCodeInterpreter(
            model_path=model_path, load_in_4bit=load_in_4bit
        )

        def bot(history):
            user_message = history[-1][0]

            interpreter.dialog.append({"role": "user", "content": user_message})

            print(f"###User : [bold]{user_message}[bold]")
            # print(f"###Assistant : ")

            # setup
            HAS_CODE = False  # For now
            INST_END_TOK_FLAG = False
            full_generated_text = ""
            prompt = interpreter.dialog_to_prompt(dialog=interpreter.dialog)
            start_prompt = copy.deepcopy(prompt)
            prompt = f"{prompt} {E_INST}"

            _ = interpreter.generate(prompt)
            history[-1][1] = ""
            generated_text = ""
            for character in interpreter.streamer:
                history[-1][1] += character
                generated_text += character
                yield history

            full_generated_text += generated_text
            HAS_CODE, generated_code_block = interpreter.extract_code_blocks(
                generated_text
            )

            attempt = 1
            while HAS_CODE:
                if attempt > MAX_TRY:
                    break
                # if no code then doesn't have to execute it

                # refine code block for history
                history[-1][1] = (
                    history[-1][1]
                    .replace(f"{B_CODE}", "\n```python\n")
                    .replace(f"{E_CODE}", "\n```\n")
                )
                history[-1][1] = change_markdown_image(history[-1][1])
                yield history

                # replace unknown thing to none ''
                generated_code_block = generated_code_block.replace(
                    "<unk>_", ""
                ).replace("<unk>", "")

                (
                    code_block_output,
                    error_flag,
                ) = interpreter.execute_code_and_return_output(
                    f"{generated_code_block}"
                )
                code_block_output = interpreter.clean_code_output(code_block_output)
                generated_text = (
                    f"{generated_text}\n{B_RESULT}\n{code_block_output}\n{E_RESULT}\n"
                )
                full_generated_text += (
                    f"\n{B_RESULT}\n{code_block_output}\n{E_RESULT}\n"
                )

                # append code output
                history[-1][1] += f"\n```RESULT\n{code_block_output}\n```\n"
                history[-1][1] = change_markdown_image(history[-1][1])
                yield history

                prompt = f"{prompt} {generated_text}"

                _ = interpreter.generate(prompt)
                for character in interpreter.streamer:
                    history[-1][1] += character
                    generated_text += character
                    history[-1][1] = change_markdown_image(history[-1][1])
                    yield history

                HAS_CODE, generated_code_block = interpreter.extract_code_blocks(
                    generated_text
                )

                if generated_text.endswith("</s>"):
                    break

                attempt += 1

            interpreter.dialog.append(
                {
                    "role": "assistant",
                    "content": generated_text.replace("<unk>_", "")
                    .replace("<unk>", "")
                    .replace("</s>", ""),
                }
            )

            print("----------\n" * 2)
            print(interpreter.dialog)
            print("----------\n" * 2)

            return history[-1][1]

        def user(user_message, history):
            return "", history + [[user_message, None]]

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process path for LLAMA2_FINETUNEED.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the finetuned LLAMA2 model.",
        default="./output/llama-2-7b-codellama-ci",
    )
    args = parser.parse_args()

    gradio_launch(model_path=args.path, load_in_4bit=True)
