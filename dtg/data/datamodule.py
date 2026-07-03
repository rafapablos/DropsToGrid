from typing import Any, Literal

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from dtg.data.fake_dataset import FakeDtGDataset
from dtg.data.source import Source
from dtg.transforms.factory import Transforms


class DtGDataModule(L.LightningDataModule):
    def __init__(
        self,
        sources: list[Source],
        batch_size: int = 1,
        eval_batch_size: int | None = None,
        num_workers: int = 4,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None else batch_size
        )
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.sources = sources
        self.save_hyperparameters()

    def set_transforms(self, transforms: Transforms):
        self.transforms = transforms

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.create_dataset("train")
            self.val_dataset = self.create_dataset("val")
        elif stage == "validate":
            self.val_dataset = self.create_dataset("val")
        elif stage == "test":
            self.test_dataset = self.create_dataset("test")
        elif stage == "predict":
            self.predict_dataset = self.create_dataset("predict")
        else:
            raise ValueError(f"Unknown stage {stage}")

    def create_dataset(
        self,
        split: Literal["train", "val", "test", "predict"],
    ) -> Dataset:
        transforms = self.transforms.train if split == "train" else self.transforms.eval
        return FakeDtGDataset(self.sources, data_transforms=transforms.data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            generator=g,
        )

    def test_dataloader(self, shuffle=True):
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            generator=g,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"transforms": self.transforms}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.transforms = state_dict["transforms"]
        if self.trainer is not None and self.trainer.state.fn is not None:
            # Trainer call DataModule.setup(stage) before loading the checkpoint
            # This leads to the datasets initially using incorrect transforms.
            # To fix this, we call setup again here to re-initialize datasets
            # with the correct transforms.
            self.setup(self.trainer.state.fn)
