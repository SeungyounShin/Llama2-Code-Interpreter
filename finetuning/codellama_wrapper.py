from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaModel, LlamaConfig
import torch.nn as nn

CODELLAMA_VOCAB_SIZE = 32016


class CodeLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, CODELLAMA_VOCAB_SIZE, bias=False)
        self.model.embed_tokens = nn.Embedding(
            CODELLAMA_VOCAB_SIZE, config.hidden_size, config.pad_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()
