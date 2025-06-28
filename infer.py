import torch
import torchaudio
from torch import nn

from vocos import Vocos


def get_vocos_mel_spectrogram(
        waveform,
        n_fft=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        hop_length=256,
        win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24_000,
            mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel

wavform, sr = torchaudio.load("ref.wav")
if sr != 24000:
    resampler = torchaudio.transforms.Resample(sr, 24000)
    wavform = resampler(wavform)
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
data = vocos.decode(MelSpec()(wavform))
torchaudio.save("output.wav", data.cpu(), 24000)

