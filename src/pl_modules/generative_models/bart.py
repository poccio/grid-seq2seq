from typing import Union

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, AutoConfig

from src.pl_modules.generative_models.generative_model import (
    GenerativeModel,
    TAGenerativeModelOutput,
    GenGenerativeModelOutput,
)
from src.pl_modules.utils import label_smoothed_nll_loss


class BartGenerativeModel(GenerativeModel):
    def __init__(self, bart_model: str, dropout: float, label_smoothing: float):

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(bart_model)
        self.config = AutoConfig.from_pretrained(bart_model, dropout=dropout)
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            bart_model, config=self.config
        )
        self._dropout = dropout
        self._label_smoothing = label_smoothing

        # metrics
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(
        self,
        source: torch.Tensor,
        source_padding_mask: torch.Tensor,
        target: torch.Tensor,
        target_padding_mask: torch.Tensor,
        num_sequences: int = 1,
        **kwargs
    ) -> Union[TAGenerativeModelOutput, GenGenerativeModelOutput]:

        if (
            target.shape[1] > 1
        ):  # training-phase: "target" is provided and we can use the "teacher-forcing" strategy.

            assert (
                not self.generation_mode
            ), 'The "target" is not empty but the GenerativeModel is in "generation mode"'

            # build target&labels
            decoder_input_ids = target[:, :-1].contiguous()
            decoder_padding_mask = target_padding_mask[:, :-1].contiguous()
            labels = target[:, 1:].contiguous()
            labels_padding_mask = target_padding_mask[:, 1:].contiguous()
            labels[~labels_padding_mask] = -100

            # actual forward
            result = self.bart_model(
                input_ids=source,
                attention_mask=source_padding_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_padding_mask,
                labels=labels,
            )
            logits = result[1]

            # compute loss with label smoothing
            labels[~labels_padding_mask] = self.tokenizer.pad_token_id
            log_probs = torch.log_softmax(logits, dim=-1)
            smoothed_loss, nll_loss = label_smoothed_nll_loss(
                log_probs.view(-1, log_probs.shape[2]),
                labels.view(-1),
                self._label_smoothing,
                padding_mask=labels_padding_mask.view(-1),
            )
            loss = smoothed_loss

            # return
            return TAGenerativeModelOutput(loss=loss, logits=logits, predictions=logits.argmax(-1))

        else:  # autoregressive-phase: the only token in target is "begin of sequence" (<s> for bart).

            assert self.generation_mode

            assert target.shape[1] == 1
            assert len(set(target[:, 0].tolist())) == 1

            generation = self.bart_model.generate(
                input_ids=source,
                attention_mask=source_padding_mask,
                num_return_sequences=num_sequences,
                decoder_start_token_id=target[0][0].item(),
                return_dict_in_generate=True,
                **self.generation_params
            )

            return GenGenerativeModelOutput(
                generation=generation.sequences[:, 1:].reshape(source.shape[0], num_sequences, -1),
                raw=generation,
            )
