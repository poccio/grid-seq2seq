import os
from typing import Union, List, Optional, Dict

import hydra
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlp_gen.utils.logging import get_project_logger

logger = get_project_logger(__name__)


class CNNDMDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, dataset: Dict, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.dataset = dataset
        self.train, self.val, self.test = None, None, None

    def prepare_data(self, *args, **kwargs):

        # check if download is needed
        if os.path.exists(self.data_dir):
            if os.path.exists(f"{self.data_dir}/LOCK"):
                return
            else:
                logger.warning(
                    "Dataset folder already exists, but written dataset seems incomplete. Overwriting it"
                )
        else:
            os.mkdir(self.data_dir)

        # download and save cnn-datamodule to disk
        logger.info("Downloading CNN-DM corpus")
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        for split in ["train", "validation", "test"]:
            with open(f"{self.data_dir}/{split}.tsv", "w") as f:
                for sample in tqdm(dataset[split], desc=f"Writing CNN-DM {split} split to disk"):
                    article, summary = (
                        sample["article"].replace("\n", ". "),
                        sample["highlights"].replace("\n", ". "),
                    )
                    f.write(f"{article}\t{summary}\n")

        # dump lock to skip future re-downloads
        with open(f"{self.data_dir}/LOCK", "w") as _:
            pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train = hydra.utils.instantiate(self.dataset, path=f"{self.data_dir}/train.tsv")
            self.val = hydra.utils.instantiate(self.dataset, path=f"{self.data_dir}/validation.tsv")
        else:
            self.test = hydra.utils.instantiate(self.dataset, path=f"{self.data_dir}/test.tsv")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val, batch_size=None, num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test, batch_size=None, num_workers=self.num_workers)
