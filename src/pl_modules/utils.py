import pytorch_lightning as pl
import hydra
import torch
from omegaconf import OmegaConf


def label_smoothed_nll_loss(lprobs, target, epsilon, padding_mask: torch.Tensor):
    """
    Inspired by https://github.com/huggingface/transformers/blob/5148f433097915f30864bf0ca6090656fecefbb8/examples/seq2seq/utils.py

    With a change however: using mean rather than sum ( nll_loss = nll...)

    """

    assert target.dim() == padding_mask.dim()
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        padding_mask = padding_mask.unsqueeze(-1)

    # compute nll loss
    nll_loss = -lprobs.gather(dim=-1, index=target)
    nll_loss.masked_fill_(~padding_mask, 0.0)

    # compute smooth loss
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    smooth_loss.masked_fill_(~padding_mask, 0.0)

    nll_loss = nll_loss.sum() / padding_mask.sum()
    smooth_loss = smooth_loss.sum() / padding_mask.sum()

    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


def load_pl_module_from_checkpoint(checkpoint_path: str):
    """
    Load a PL module from a checkpoint path only. Infer the model to load from the dumped hydra conf

    Args:
        checkpoint_path (str):

    Returns:
        pl.LightningModule

    """

    # find hydra config path
    hydra_config_path = "/".join(checkpoint_path.split("/")[:-2])

    # load hydra config
    conf = OmegaConf.load(f"{hydra_config_path}/.hydra/config.yaml")

    # instantiate and return
    return hydra.utils.instantiate(
        {"_target_": f'{conf["model"]["_target_"]}.load_from_checkpoint'},
        checkpoint_path=checkpoint_path,
    )
