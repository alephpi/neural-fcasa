from typing import Any
from pathlib import Path
from torch.utils.data import DataLoader
import lightning as lt
# from datasets import Dataset as HFDataset, load_from_disk
from neural_fcasa.datasets.hf_wavact_dataset import WavActDataset, WavActTransform


class DataModule(lt.LightningDataModule):
    def __init__(
        self,
        train_dataset_path: str | Path,
        val_dataset_path: str | Path,
        batch_size: int,
        duration: int | None,
        sr: int,
        hop_length: int,
        randperm_mic: bool = True,
        randperm_spk: bool = True,
        num_workers: int = 10,
        keep_in_memory: bool | None = None,
    ):
        super().__init__()

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path

        self.dataset_kwargs: dict[str, Any] = dict(
            duration=duration,
            sr=sr,
            hop_length=hop_length,
            randperm_mic=randperm_mic,
            randperm_spk=randperm_spk,
            keep_in_memory=keep_in_memory,
        )

        self.dataloader_configs: dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def setup(self, stage: str | None):
        if stage == "fit":
            self.train_dataset = WavActDataset(self.train_dataset_path, **self.dataset_kwargs)
            self.val_dataset = WavActDataset(self.val_dataset_path, **self.dataset_kwargs)

            print(f"Dataset size: {len(self.train_dataset)=},  {len(self.val_dataset)=}")
        else:
            raise ValueError("`stage` is not 'fit'.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            drop_last=True,
            **self.dataloader_configs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            drop_last=False,
            **self.dataloader_configs,
        )
