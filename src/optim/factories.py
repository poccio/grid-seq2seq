from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig

from src.optim.optimizers.radam import RAdam


class Factory:
    def __call__(self, module: torch.nn.Module):
        raise NotImplementedError


class TorchFactory(Factory):

    # todo add scheduler support as well

    def __init__(self, optimizer: DictConfig):
        self.optimizer = optimizer

    def __call__(self, module: torch.nn.Module):
        return hydra.utils.instantiate(self.optimizer, params=module.parameters())


class RadamWithDecayFactory(Factory):
    def __init__(self, lr: float, weight_decay: float, no_decay_params: Optional[List[str]]):
        self.lr = lr
        self.weight_decay = weight_decay
        self.no_decay_params = no_decay_params

    def __call__(self, module: torch.nn.Module):

        if self.no_decay_params is not None:

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in module.named_parameters()
                        if not any(nd in n for nd in self.no_decay_params)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in module.named_parameters()
                        if any(nd in n for nd in self.no_decay_params)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        else:

            optimizer_grouped_parameters = [
                {"params": module.parameters(), "weight_decay": self.weight_decay}
            ]

        optimizer = RAdam(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
