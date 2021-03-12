from typing import Dict, Any

import hydra
import pytorch_lightning as pl
import torch


class GenerativePLModule(pl.LightningModule):
    def __init__(self, generative_model: Dict, optim_conf: Dict):

        super().__init__()
        self.save_hyperparameters()

        # generative model & optim
        self.generative_model = hydra.utils.instantiate(generative_model)
        self._optim_conf = optim_conf

        # metrics
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    @property
    def tokenizer(self):
        return self.generative_model.tokenizer

    def enable_generation_mode(self):
        self.generative_model.enable_generation_mode()

    def disable_generation_mode(self):
        self.generative_model.disable_generation_mode()

    def load_generation_params(self, generation_params: Dict):
        self.generative_model.load_generation_params(generation_params)

    def forward(self, *args, **kwargs):
        return self.generative_model(*args, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:

        forward_output = self.forward(**batch)

        # loss
        self.log("train_loss", forward_output.loss, prog_bar=True, on_step=False, on_epoch=True)

        # perplexity
        self.log(
            "train_ppl", torch.exp(forward_output.loss), prog_bar=True, on_step=False, on_epoch=True
        )

        # accuracy
        padding_mask = batch["target_padding_mask"][:, 1:].reshape(-1)
        train_acc = self.train_acc(
            forward_output.predictions.view(-1)[padding_mask],
            batch["target"][:, 1:].reshape(-1)[padding_mask],
        )
        self.log("train_accuracy", train_acc, prog_bar=True, on_step=False, on_epoch=True)

        # return
        return forward_output.loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):

        forward_output = self.forward(**batch)

        # loss
        self.log("val_loss", forward_output.loss, prog_bar=True, on_step=False, on_epoch=True)

        # perplexity
        self.log(
            "val_ppl", torch.exp(forward_output.loss), prog_bar=True, on_step=False, on_epoch=True
        )

        # accuracy
        padding_mask = batch["target_padding_mask"][:, 1:].reshape(-1)
        val_acc = self.val_acc(
            forward_output.predictions.view(-1)[padding_mask],
            batch["target"][:, 1:].reshape(-1)[padding_mask],
        )
        self.log("val_accuracy", val_acc, prog_bar=True, on_step=False, on_epoch=True)

        # return
        return forward_output.loss

    def configure_optimizers(self):
        return hydra.utils.instantiate(self._optim_conf, _recursive_=False)(module=self)
