import math
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import transformers

from vocos import Vocos
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.feature_extractors import FeatureExtractor
from vocos.heads import FourierHead
from vocos.helpers import plot_spectrogram_to_numpy
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from vocos.models import Backbone
from vocos.modules import safe_log


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            backbone: Backbone,
            head: FourierHead,
            sample_rate: int,
            initial_learning_rate: float,
            num_warmup_steps: int = 0,
            mel_loss_coeff: float = 45,
            mrd_loss_coeff: float = 1.0,
            pretrain_mel_steps: int = 0,
            decay_mel_coeff: bool = False,
            evaluate_utmos: bool = False,
            evaluate_pesq: bool = False,
            evaluate_periodicty: bool = False,
            pretrained_hf_model_id: Optional[str] = None,
            train_with_mel=False
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

        if pretrained_hf_model_id:
            print(f"Loading pretrained generator weights from Hugging Face: {pretrained_hf_model_id}")
            # Vocos.from_pretrained() 会返回一个包含预训练的 feature_extractor, backbone, head 的 Vocos 对象
            pretrained_generator = Vocos.from_pretrained(pretrained_hf_model_id)

            # 将预训练的权重加载到我们当前的模块中
            self.feature_extractor.load_state_dict(pretrained_generator.feature_extractor.state_dict())
            self.backbone.load_state_dict(pretrained_generator.backbone.state_dict())
            self.head.load_state_dict(pretrained_generator.head.state_dict())

            print("Successfully loaded pretrained generator weights.")

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff
        self.train_with_mel = train_with_mel

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
        ]
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
        ]

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def forward(self, x, **kwargs):
        if self.train_with_mel:
            features = x
        else:
            features = self.feature_extractor(x, **kwargs)
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        if self.train_with_mel:
            audio_input, mel_input = batch
        else:
            audio_input = batch

        # train discriminator
        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                if self.train_with_mel:
                    audio_hat = self(mel_input, **kwargs)
                else:
                    audio_hat = self(audio_input, **kwargs)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs, )
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs, )
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("updates", int(self.global_step), on_step=True, prog_bar=True, logger=False)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            return loss

        # train generator
        if optimizer_idx == 1:
            if self.train_with_mel:
                audio_hat = self(mel_input, **kwargs)
            else:
                audio_hat = self(audio_input, **kwargs)
            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                    loss_gen_mp
                    + self.hparams.mrd_loss_coeff * loss_gen_mrd
                    + loss_fm_mp
                    + self.hparams.mrd_loss_coeff * loss_fm_mrd
                    + self.mel_loss_coeff * mel_loss
            )

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("updates", int(self.global_step), on_step=True, prog_bar=True, logger=False)
            self.log("mel_loss_coeff", self.mel_loss_coeff)
            self.log("generator/mel_loss", mel_loss)

            if self.global_step % 1000 == 0 and self.global_rank == 0:
                self.logger.experiment.add_audio(
                    "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                self.logger.experiment.add_audio(
                    "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
                self.logger.experiment.add_image(
                    "train/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "train/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )

            return loss

    def on_validation_epoch_start(self):
        if self.hparams.evaluate_utmos:
            from metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx, **kwargs):
        if self.train_with_mel:
            audio_input, mel_input = batch
            audio_hat = self(mel_input, **kwargs)
        else:
            audio_input = batch
            audio_hat = self(audio_input, **kwargs)

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        if self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score)

        return {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            for i in range(min(len(outputs), 5)):
                *_, audio_in, audio_pred = outputs[i].values()
                self.logger.experiment.add_audio(
                    f"val_in{i}", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
                )
                self.logger.experiment.add_audio(
                    f"val_pred{i}", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
                )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


class VocosEncodecExp(VocosExp):
    """
    VocosEncodecExp is a subclass of VocosExp that overrides the parent experiment to function as a conditional GAN.
    It manages an additional `bandwidth_id` attribute, which denotes a learnable embedding corresponding to
    a specific bandwidth value of EnCodec. During training, a random bandwidth_id is generated for each step,
    while during validation, a fixed bandwidth_id is used.
    """

    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            backbone: Backbone,
            head: FourierHead,
            sample_rate: int,
            initial_learning_rate: float,
            num_warmup_steps: int,
            mel_loss_coeff: float = 45,
            mrd_loss_coeff: float = 1.0,
            pretrain_mel_steps: int = 0,
            decay_mel_coeff: bool = False,
            evaluate_utmos: bool = False,
            evaluate_pesq: bool = False,
            evaluate_periodicty: bool = False,
    ):
        super().__init__(
            feature_extractor,
            backbone,
            head,
            sample_rate,
            initial_learning_rate,
            num_warmup_steps,
            mel_loss_coeff,
            mrd_loss_coeff,
            pretrain_mel_steps,
            decay_mel_coeff,
            evaluate_utmos,
            evaluate_pesq,
            evaluate_periodicty,
        )
        # Override with conditional discriminators
        self.multiperioddisc = MultiPeriodDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.multiresddisc = MultiResolutionDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))

    def training_step(self, *args):
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,),
                                     device=self.device, )
        output = super().training_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_step(self, *args):
        bandwidth_id = torch.tensor([0], device=self.device)
        output = super().validation_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, _ = outputs[0].values()
            # Resynthesis with encodec for reference
            self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[0])
            encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
            self.logger.experiment.add_audio(
                "encodec", encodec_audio[0, 0].data.cpu().numpy(), self.global_step, self.hparams.sample_rate,
            )

        super().validation_epoch_end(outputs)
