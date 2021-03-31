import os

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.callbacks.best_checkpoint import ModelCheckpointWithBest


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.training.reprod.seed)

    # main module declaration
    pl_module = hydra.utils.instantiate(
        conf.model, optim_conf=conf.training.trainer.optim, _recursive_=False
    )

    # data_module declaration
    pl_data_module = hydra.utils.instantiate(
        conf.data.datamodule,
        tokenizer=pl_module.tokenizer,  # todo bad coupling towards huggingface
        _recursive_=False,
    )

    # callbacks

    callbacks = []

    # callbacks: checkpoint and early stopping

    monitor = conf.training.trainer.checkpoint.monitor
    assert monitor[0] in ["-", "+"]
    mode = "min" if monitor[0] == "-" else "max"
    monitor = monitor[1:]

    callbacks.append(
        ModelCheckpointWithBest(
            monitor=monitor,
            mode=mode,
            dirpath=f"checkpoints",
            filename=conf.training.trainer.checkpoint.filename,
            save_top_k=conf.training.trainer.checkpoint.save_top_k,
            save_last=conf.training.trainer.checkpoint.save_last,
            verbose=True,
        )
    )

    patience = conf.training.trainer.patience
    if patience is not None:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=conf.training.trainer.patience)
        )

    # custom callbacks

    for callback in conf.callbacks.callbacks:
        callbacks.append(hydra.utils.instantiate(callback))

    # instantiate trainer logger
    logger = hydra.utils.instantiate(conf.training.logger)

    # trainer
    trainer = pl.Trainer(
        **conf.device,
        accumulate_grad_batches=conf.training.trainer.gradient_acc_steps,
        gradient_clip_val=conf.training.trainer.gradient_clip_value,
        max_steps=conf.training.trainer.max_steps,
        val_check_interval=conf.training.trainer.val_check_interval,
        logger=logger,
        callbacks=callbacks,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../../configurations/hydra-train", config_name="root")
def main(conf: omegaconf.DictConfig):

    # fix paths

    def fix(conf):
        if type(conf) == list or type(conf) == omegaconf.listconfig.ListConfig:
            for i in range(len(conf)):
                conf[i] = fix(conf[i])
            return conf
        elif type(conf) == dict or type(conf) == omegaconf.dictconfig.DictConfig:
            for k, v in conf.items():
                conf[k] = fix(v)
            return conf
        elif type(conf) == str:
            if "/" in conf and os.path.exists(
                hydra.utils.to_absolute_path(conf[: conf.rindex("/")])
            ):
                return hydra.utils.to_absolute_path(conf)
            else:
                return conf
        elif type(conf) in [float, int, bool]:
            return conf
        else:
            raise ValueError(f"Unexpected type {type(conf)}: {conf}")

    fix(conf)

    # actual train
    train(conf)


if __name__ == "__main__":
    main()
