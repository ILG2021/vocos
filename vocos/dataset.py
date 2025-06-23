from dataclasses import dataclass

import librosa
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


def normalize_db(y: torch.Tensor, target_db: float, eps: float = 1e-8) -> torch.Tensor:
    """
    将音频张量标准化到指定的目标分贝水平 (RMS based)。
    这是 torchaudio.sox_effects 'norm' 的一个纯 PyTorch 等效实现。

    Args:
        y (torch.Tensor): 输入音频张量，形状为 (..., channels, T)。
        target_db (float): 目标分贝值。
        eps (float): 用于防止 log(0) 的小常数。

    Returns:
        torch.Tensor: 标准化后的音频张量。
    """
    # 计算当前音频的 RMS
    rms = torch.sqrt(torch.mean(y ** 2) + eps)

    # 计算将当前 RMS 调整到目标 dB 所需的线性缩放因子
    # 公式: scale = 10^(target_db / 20) / current_rms
    # 为了数值稳定性，我们使用对数运算：
    # gain_db = target_db - 20 * log10(rms)
    # scale = 10^(gain_db / 20)

    current_db = 20 * torch.log10(rms)
    gain_db = target_db - current_db
    scale_factor = 10 ** (gain_db / 20)

    # 应用增益
    normalized_y = y * scale_factor

    # SoX 的 norm 效果会自动处理削波，这里我们也加上，确保值在 [-1, 1] 范围内
    normalized_y = torch.clamp(normalized_y, -1.0, 1.0)

    return normalized_y


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path, 'r', encoding='utf-8') as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        print(audio_path)
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3

        # 替换为:
        y = normalize_db(y, target_db=gain)

        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
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
