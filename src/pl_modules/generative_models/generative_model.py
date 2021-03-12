from dataclasses import dataclass
from typing import Dict, Any, Union

import torch
from torch import nn


@dataclass
class TAGenerativeModelOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    predictions: torch.Tensor


@dataclass
class GenGenerativeModelOutput:
    generation: torch.Tensor
    raw: Any


class GenerativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generation_mode = False
        self.generation_params = None

    def enable_generation_mode(self):
        self.generation_mode = True

    def disable_generation_mode(self):
        self.generation_mode = False

    def load_generation_params(self, generation_params: Dict[str, Any]):
        self.generation_params = generation_params

    def forward(self, *args, **kwargs) -> Union[TAGenerativeModelOutput, GenGenerativeModelOutput]:
        raise NotImplementedError
