import json
import os
from pathlib import Path
from typing import Union

import config
from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
)

KB = 1024


class SupportedModel:
    def __init__(
        self,
        nameOfModel: str,
        pathToModel: str,
        quantization: int,
        max_seq_len: int,
        prompt_limit: int,
        response_limit: int,
        max_context_theory: int = (
            128 * KB
        ),  # Debugging purposes: max_seq_len < max_context_theory
    ):
        self.name = nameOfModel
        self.path = pathToModel
        self.quant = quantization
        self.max_seq_len = max_seq_len
        self.prompt_limit = prompt_limit
        self.response_limit = response_limit
        self.max_context_theory = max_context_theory
        if self.max_seq_len > max_context_theory:
            raise ValueError(f"max_seq_len > max_context_theory ({max_context_theory})")
        if (self.prompt_limit + self.response_limit) > max_seq_len:
            raise ValueError(
                f"Worst-case one-shot does not fit in max sequence context ({max_seq_len})"
            )

    def to_dict(self):
        return {
            "name": self.name,
            "path": str(self.path),
            "quantization": self.quant,
            "max_seq_len": self.max_seq_len,
            "prompt_limit": self.prompt_limit,
            "response_limit": self.response_limit,
        }

    def get_cache(self):
        if self.quant == 4:
            return ExLlamaV2Cache_Q4
        elif self.quant == 6:
            return ExLlamaV2Cache_Q6
        elif self.quant == 8:
            return ExLlamaV2Cache_Q8
        elif self.quant == 16:
            return ExLlamaV2Cache
        else:
            raise ValueError(f"Quantization {self.quant} is not supported")

    def set_model(self):
        with open(config.ACTIVE_MODEL_FILE, "w") as f:
            json.dump(self.to_dict(), f)

    def get_supported_models():
        res = []
        for m in SUPPORTED_MODELS:
            res.append(m.name)
        return res


SUPPORTED_MODELS = [
    SupportedModel(
        "gemma3-27b-Q6",  # TODO: quantize and test
        config.MODEL_DIR + "/gemma-3-27b-it-Q6",
        8,
        48 * KB,
        40 * KB,
        8 * KB,
    ),
    SupportedModel(
        "gemma3-27b-Q4",  # TODO: quantize and test
        config.MODEL_DIR + "/gemma-3-27b-it-Q4",
        4,
        64 * KB,
        32 * KB,
        8 * KB,
    ),
    SupportedModel(
        "Qwen-72b-Q4",  # APPROVED
        config.MODEL_DIR + "/Qwen2.5-72B-Instruct-Q4",
        4,
        24 * KB,
        2 * KB,
        8 * KB,
    ),
    SupportedModel(
        "Qwen-Coder-32B-Q4",  # APPROVED
        config.MODEL_DIR + "/Qwen2.5-Coder-32B-Instruct-Q4",
        4,
        120 * KB,
        8 * KB,
        8 * KB,
    ),
    SupportedModel(
        "Qwen-Coder-32B-Q8",  # TODO: quantize and test
        config.MODEL_DIR + "/Qwen2.5-Coder-32B-Instruct-Q8",
        8,
        24 * KB,
        16 * KB,
        8 * KB,
    ),
    SupportedModel(
        "Qwen-Coder-14B",  # APPROVED
        config.MODEL_DIR + "/Qwen2.5-Coder-14B-Instruct",
        16,
        128 * KB,
        8 * KB,
        8 * KB,
    ),
    SupportedModel(
        "Qwen-Coder-14B-Q8",  # APPROVED
        config.MODEL_DIR + "/Qwen2.5-Coder-14B-Instruct-Q8",
        8,
        128 * KB,
        8 * KB,
        8 * KB,
    ),
    SupportedModel(
        "QwQ-32B-Q4",  # APPROVED
        config.MODEL_DIR + "/QwQ-32B-Q4",
        4,
        57 * KB,
        16 * KB,
        8 * KB,
    ),
    SupportedModel(
        "QwQ-32B-Q8",  # APPROVED
        config.MODEL_DIR + "/QwQ-32B-Q8",
        8,
        16 * KB,
        8 * KB,
        8 * KB,
    ),
]


def get_model_by_name(name: str) -> SupportedModel | None:
    return next((m for m in SUPPORTED_MODELS if m.name == name), None)


def get_active_model() -> SupportedModel | None:
    print(config.ACTIVE_MODEL_FILE)
    if not os.path.exists(config.ACTIVE_MODEL_FILE):
        return None
    with open(config.ACTIVE_MODEL_FILE, "r") as f:
        data = json.load(f)
        return SupportedModel(**data[0])
