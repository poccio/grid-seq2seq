from dataclasses import dataclass
from typing import Dict, Any, Union

import torch
from torch import nn


@dataclass
class TAGenerativeModelOutput:
    """Dataclass storing the returned fields of a teacher-forced forward.

    Args:
        loss (torch.Tensor)
        logits (torch.Tensor)
        predictions (torch.Tensor)
    """

    loss: torch.Tensor
    logits: torch.Tensor
    predictions: torch.Tensor


@dataclass
class GenGenerativeModelOutput:
    """Dataclass storing the returned fields of a generative forward.

    Args:
        generation (torch.Tensor): tensor of shape (batch_size, num_sequences, sequence_length) storing the generated sequences
        raw (torch.Tensor): raw object storing details of the decoding processes. For example, when using HuggingFace Transformers'
            models, it can be used to store attentions, generation scores, ...
    """

    generation: torch.Tensor
    raw: Any


class GenerativeModel(nn.Module):
    """Abstract class denoting the interface of a generative model.

    This class essentially alternates between two states, teacher-forced or generative, and the current state regulates
    the forward method. The teacher-forced state is the one expected to be activate at training time: given the input, the
    model will compute and return a teacher-forced loss. Conversely, the generative state is for generation: the model will
    generate sequences conditioned on the provided input.

    Besides the forward method, GenerativeModel offers hooks to switch between the two modes and an additional one to
    load generation-time parameters.
    """

    def __init__(self):
        super().__init__()
        self.generation_mode = False
        self.generation_params = None

    def enable_generation_mode(self):
        """Enables generation mode for the model."""
        self.generation_mode = True

    def disable_generation_mode(self):
        """Disables generation mode for the model."""
        self.generation_mode = False

    def load_generation_params(self, generation_params: Dict[str, Any]):
        """Loads and stores the given generation params, that will be used in the next generation forwards"""
        self.generation_params = generation_params

    def forward(self, *args, **kwargs) -> Union[TAGenerativeModelOutput, GenGenerativeModelOutput]:
        raise NotImplementedError
