from typing import Union, List, Optional, Dict

import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.utils.logging import get_project_logger

logger = get_project_logger(__name__)


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Optional[Dict] = None,
        validation_dataset: Optional[Dict] = None,
        test_dataset: Optional[Dict] = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_dataset_conf = train_dataset
        self.validation_dataset_conf = validation_dataset
        self.test_dataset_conf = test_dataset
        self.num_workers = num_workers
        self.train_dataset, self.validation_dataset, self.test_dataset = None, None, None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            assert self.train_dataset_conf is not None and self.validation_dataset_conf is not None
            self.train_dataset = hydra.utils.instantiate(
                self.train_dataset_conf, tokenizer=self.tokenizer, _recursive_=False
            )
            self.validation_dataset = hydra.utils.instantiate(
                self.validation_dataset_conf, tokenizer=self.tokenizer, _recursive_=False
            )
        else:
            assert self.test_dataset_conf is not None
            self.test_dataset = hydra.utils.instantiate(
                self.test_dataset_conf, tokenizer=self.tokenizer, _recursive_=False
            )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=None, num_workers=self.num_workers)
