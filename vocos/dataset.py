from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig, train_with_mel):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params
        self.train_with_mel = train_with_mel

    def _get_dataloder(self, cfg: DataConfig, train: bool, train_with_mel):
        dataset = VocosDataset(cfg, train, train_with_mel)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, True, self.train_with_mel)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, False, self.train_with_mel)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool, train_with_mel: bool):
        with open(cfg.filelist_path, 'r', encoding='utf-8') as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.train_with_mel = train_with_mel

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        # y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        y = torchaudio.functional.gain(y, gain)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)

        if self.train_with_mel:
            return y[0], torch.load(str(Path(audio_path).with_suffix('.pt')))   # audio file and mel file are at the same folder
        else:
            if y.size(-1) < self.num_samples:
                pad_length = self.num_samples - y.size(-1)
                padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
            elif self.train:
                start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
                y = y[:, start: start + self.num_samples]
            else:
                # During validation, take always the first segment for determinism
                y = y[:, : self.num_samples]
            return y[0]
