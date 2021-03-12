import glob
import itertools
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Callable, Optional

import pytorch_lightning as pl
import torch
import wandb
from datasets import load_metric

from src.scripts.model.translate import translate

from src.utils.logging import get_project_logger

logger = get_project_logger(__name__)


class GenerationCallback:
    def __call__(self, translations: List[Tuple[str, str]], module: pl.LightningModule):
        raise NotImplementedError


class RougeGenerationCallback(GenerationCallback):
    def __init__(self):
        self.rouge = load_metric("rouge")

    def __call__(
        self, translations: List[Tuple[str, List[str], Optional[str]]], module: pl.LightningModule
    ):
        assert all(t[2] is not None for t in translations)
        results = self.rouge.compute(
            predictions=[t[1][0] for t in translations], references=[t[2] for t in translations]
        )
        for k, v in results.items():
            module.log(f"val_{k}", v.mid.fmeasure, prog_bar=True, on_step=False, on_epoch=True)


class TextGenerationCallback(pl.Callback):
    def __init__(
        self, generation_callbacks: Dict[str, GenerationCallback], generations: List[Dict[str, Any]]
    ):
        self._epoch = 0
        self.generation_callbacks = generation_callbacks
        self.generations_confs = []
        for g in generations:
            self.generations_confs.append(
                (
                    g["name"],
                    g["glob_translate_path"],
                    g["generation_param_conf_path"],
                    g["num_sequences"],
                    g["token_batch_size"],
                    g["limit"],
                    g["enabled_generation_callbacks"],
                )
            )

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        pl_module.freeze()
        pl_module.eval()

        wandb_table = wandb.Table(columns=["Configuration", "Source", "Input", "Pred", "Gold"])
        logger.info("Executing translation callback")

        spec_name2translation_pairs = {}

        for (
            name,
            glob_translate_path,
            generation_param_conf_path,
            num_sequences,
            token_batch_size,
            limit,
            enabled_generation_callbacks,
        ) in self.generations_confs:

            spec_name2translation_pairs[name] = []

            # translate

            for translation_path in glob.iglob(glob_translate_path):

                logger.info(
                    f"Translating translation path {translation_path} for configuration {name}"
                )
                source_type = translation_path.split("/")[-1][:-4]

                with open(translation_path) as f:

                    # read sources
                    iterator = map(lambda l: l.strip(), f)

                    # do only a dry run on first epoch (correspond to sanity check run)
                    if self._epoch == 0:
                        iterator = itertools.islice(iterator, 5)

                    # apply limit
                    if limit != -1:
                        iterator = itertools.islice(iterator, limit)

                    for i, (source, sample_translations, gold_output) in enumerate(
                        translate(
                            pl_module,
                            iterator,
                            num_sequences=num_sequences,
                            generation_param_conf_path=generation_param_conf_path,
                            token_batch_size=token_batch_size,
                        )
                    ):
                        if i % 100 == 0:
                            logger.debug(
                                f"Translating translation path {translation_path} for configuration {name}: {i} lines translated"
                            )

                        for translation in sample_translations:
                            wandb_table.add_data(
                                name, source_type, source, translation, gold_output
                            )

                        spec_name2translation_pairs[name].append(
                            (source, sample_translations, gold_output)
                        )

                    if self._epoch == 0:
                        # do only a dry run on first epoch (correspond to sanity check run)
                        break

            # run callbacks

            for callback in enabled_generation_callbacks:
                self.generation_callbacks[callback](spec_name2translation_pairs[name], pl_module)

        pl_module.spec_name2translation_pairs = spec_name2translation_pairs
        if self._epoch > 0:
            trainer.logger.experiment.log({"translations": wandb_table})

        logger.info("Translation callback completed")
        self._epoch += 1

        pl_module.unfreeze()
        pl_module.train()
