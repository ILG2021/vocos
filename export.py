import os.path
import shutil

import click
import huggingface_hub
import torch


@click.command()
@click.option("--checkpoint")
def trim(checkpoint):
    pretrain_dir = huggingface_hub.snapshot_download("charactr/vocos-mel-24khz")
    shutil.copy(os.path.join(pretrain_dir, "config.yaml"), ".")
    pretrain_weight = torch.load(os.path.join(pretrain_dir, "pytorch_model.bin"))
    my_weight = torch.load(checkpoint)['state_dict']
    my_weight = {k: v for k, v in my_weight.items() if k in pretrain_weight}
    torch.save(my_weight, "pytorch_model.bin")


if __name__ == '__main__':
    trim()