from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import logging
import os, sys
import copy

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

from torch.utils.data import Dataset
from transformers import Trainer

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./output/llama-2-7b-chat-ci")
    load_peft: Optional[bool] = field(default=False)
    peft_model_name_or_path: Optional[str] = field(
        default="./output/llama-2-7b-chat-ci"
    )


def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def build_model_from_hf_path(
    hf_base_model_path: str = "./ckpt/llama-2-13b-chat",
    load_peft: Optional[bool] = False,
    peft_model_path: Optional[str] = None,
):
    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        hf_base_model_path,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN  # 32000
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN  # 2
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN  # 1
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.add_tokens(
        [
            B_CODE,  # 32001
            E_CODE,  # 32002
            B_RESULT,  # 32003
            E_RESULT,  # 32004
            B_INST,
            E_INST,
            B_SYS,
            E_SYS,  # 32008
        ],
        special_tokens=True,
    )

    # build model
    model = LlamaForCausalLM.from_pretrained(
        hf_base_model_path,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.resize_token_embeddings(len(tokenizer))

    if load_peft and (peft_model_path is not None):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, peft_model_path)
        print(f"Peft Model Loaded")

    return {"tokenizer": tokenizer, "model": model}


@torch.inference_mode()
def inference(
    user_input="What is 55th fibonacci?",
    max_new_tokens=512,
    do_sample: bool = True,
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    VERBOSE: bool = True,
):
    parser = transformers.HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]

    model_dict = build_model_from_hf_path(
        hf_base_model_path=model_args.model_name_or_path,
        load_peft=model_args.load_peft,
        peft_model_path=model_args.peft_model_name_or_path,
    )

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    # peft
    # create peft config
    model.eval()

    user_input = msg_to_code_result_tok_temp(
        [{"role": "user", "content": f"{user_input}"}]
    )
    prompt = f"{user_input}\n### Assistant :"
    # prompt = f"{user_input}\n### Assistant : Here is python code to get the 55th fibonacci number {B_CODE}\n"

    batch = tokenizer(prompt, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}

    outputs = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        use_cache=use_cache,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

    if VERBOSE:
        print(generated_text)

    return outputs


if __name__ == "__main__":
    inference()
