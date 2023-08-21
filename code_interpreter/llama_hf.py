from typing import Optional
import os, sys

from transformers import LlamaForCausalLM, LlamaTokenizer

import torch
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
    load_in_4bit: bool = True,
):
    start_time = datetime.now()

    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        hf_base_model_path,
        padding_side="right",
        use_fast=False,
    )

    # Handle special tokens
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
        [B_CODE, E_CODE, B_RESULT, E_RESULT, B_INST, E_INST, B_SYS, E_SYS],
        special_tokens=True,
    )

    # build model
    model = LlamaForCausalLM.from_pretrained(
        hf_base_model_path,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))

    if load_peft and (peft_model_path is not None):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, peft_model_path)

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    return {"tokenizer": tokenizer, "model": model}
