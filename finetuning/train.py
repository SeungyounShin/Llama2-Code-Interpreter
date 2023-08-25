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

from conversation_template import json_to_code_result_tok_temp


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./ckpt/llama-2-13b-chat")
    peft: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
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
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(f"Using Peft")
    return model, peft_config


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def find_all_sublist_end(main_list, sublist):
    """Find all the ending indices of a sublist in a main list."""
    sublist_len = len(sublist)
    main_list = main_list.tolist()
    indices = []
    for index in (i for i, e in enumerate(main_list) if e == sublist[0]):
        if main_list[index : index + sublist_len] == sublist:
            indices.append(index + sublist_len)
    return indices


def find_all_sublist_start(main_list, sublist):
    """Find all the starting indices of a sublist in a main list."""
    sublist_len = len(sublist)
    main_list = main_list.tolist()
    indices = []
    for index in (i for i, e in enumerate(main_list) if e == sublist[0]):
        if main_list[index : index + sublist_len] == sublist:
            indices.append(index)
    return indices


def preprocess(
    trajs: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    INST_START_INDEX = tokenizer.encode(f"{B_INST}")[-1]
    INST_END_INDEX = tokenizer.encode(f"{E_INST}")[-1]
    RESULT_START_INDEX = tokenizer.encode(f"{B_RESULT}")[-1]
    RESULT_END_INDEX = tokenizer.encode(f"{E_RESULT}")[-1]

    """Preprocess the data by tokenizing."""
    examples_tokenized = _tokenize_fn(trajs, tokenizer)

    input_ids_lens = examples_tokenized["input_ids_lens"]
    input_ids = examples_tokenized["input_ids"]  # [torch.tensor , torch.tensor , ...]
    labels = copy.deepcopy(input_ids)

    # IGNORE INDEX SET
    for i, label in enumerate(labels):
        user_start_inds = find_all_sublist_start(label, [INST_START_INDEX])
        assistant_start_inds = find_all_sublist_end(label, [INST_END_INDEX])

        result_start_inds = find_all_sublist_start(label, [RESULT_START_INDEX])
        result_end_inds = find_all_sublist_end(label, [RESULT_END_INDEX])

        # for debug
        # for len_i, ind in enumerate(label):
        #    print(f'{len_i}|{ind} -> "{tokenizer.decode(ind)}"')

        assert len(user_start_inds) == len(
            assistant_start_inds
        ), f"User and Assistant pair should be equal :: \n\tUser [{user_start_inds}]/\n\tAssistant [{assistant_start_inds}]\n\n Text : \n{trajs[i]}"

        assert len(result_start_inds) == len(
            result_end_inds
        ), f"Start and End indices pairs do not match.: : \nText : \n{trajs[i]}"

        for user_start_ind, assistant_start_ind in zip(
            user_start_inds, assistant_start_inds
        ):
            label[user_start_ind + 1 : assistant_start_ind - 1] = IGNORE_INDEX

        for start, end in zip(result_start_inds, result_end_inds):
            label[start + 1 : end - 1] = IGNORE_INDEX

    # cut max length
    input_ids = [i[:1500] for i in input_ids]
    labels = [i[:1500] for i in labels]

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning(f"Loading data from data path : {data_path}")
        all_json = os.listdir(data_path)

        trajs = list()
        for json_file_name in all_json:
            traj = json_to_code_result_tok_temp(json_file_name=json_file_name)
            trajs.append(traj)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(trajs, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def build_model_from_hf_path(
    hf_model_path: str = "./ckpt/llama-2-13b-chat", peft: bool = False
):
    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        hf_model_path,
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
    if peft:
        model = LlamaForCausalLM.from_pretrained(
            hf_model_path,
            load_in_8bit=True,
            device_map="auto",
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16,
        )
    else:
        # for llama
        # model = LlamaForCausalLM.from_pretrained(
        #    hf_model_path, ignore_mismatched_sizes=True
        # )

        # for codellama
        from codellama_wrapper import CodeLlamaForCausalLM

        model = CodeLlamaForCausalLM.from_pretrained(hf_model_path)

    model.resize_token_embeddings(len(tokenizer))

    return {"tokenizer": tokenizer, "model": model}


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_dict = build_model_from_hf_path(
        hf_model_path=model_args.model_name_or_path, peft=model_args.peft
    )

    model, tokenizer = model_dict["model"], model_dict["tokenizer"]
    # peft setting
    model.train()
    if model_args.peft:
        model, lora_config = create_peft_config(model)

    # make dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # train
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
