from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, random_split

from src.datamodules.components.lung_dataset import LungDataset
from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule


class LungDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )

    @property
    def num_classes(self) -> int:
        return 27

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            # During testing on MNIST, transformations should be the same
            # because we split initial train + test into train + valid + test
            transforms_train = TransformsWrapper(self.transforms.get("train"))
            transforms_test = TransformsWrapper(
                self.transforms.get("valid_test_predict")
            )
            train_set = LungDataset(
                self.cfg_datasets.get("data_dir"),
                self.cfg_datasets.get("data_dir"),
                transform=transforms_train,
            )
            dataset = ConcatDataset(datasets=[train_set])
            seed = self.cfg_datasets.get("seed")
            self.train_set, self.valid_set, self.test_set = random_split(
                dataset=dataset,
                lengths=self.cfg_datasets.get("train_val_test_split"),
                generator=torch.Generator().manual_seed(seed),
            )
        # load predict dataset only if test set existed already
        if (stage == "predict") and self.test_set:
            self.predict_set = {"PredictDataset": self.test_set}

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    pass