"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.plugins import DDPPlugin
from torchvision import transforms

from data.transforms import (
    AddGaussianNoise,
    Compose,
    HistogramNormalize,
    RandomGaussianBlur,
    TensorToRGB,
)
from data.xray_module import XrayDataModule
from training.moco_module import MoCoModule


def build_args():
    pl.seed_everything(1234)
    data_config = "configs/data.yaml"
    arg_defaults = {
        "accelerator": "ddp",
        "strategy": DDPPlugin(find_unused_parameters=False),
        "gpus": [0, 1],
        "num_workers": 96,
        "batch_size": 256,
    }

    parser = ArgumentParser()
    parser.add_argument("--im_size", default=224, type=int)
    parser.add_argument("--ckpt", type=str)

    parser = MoCoModule.add_model_specific_args(parser)
    parser = XrayDataModule.add_model_specific_args(parser)

    parser.set_defaults(**arg_defaults)
    args = parser.parse_args()

    if args.default_root_dir is None:
        args.default_root_dir = Path.cwd()

    if args.dataset_dir is None:
        with open(data_config, "r") as f:
            paths = yaml.load(f, Loader=yaml.SafeLoader)["paths"]

        if args.dataset_name == "mimic":
            args.dataset_dir = paths["mimic"]
        elif args.dataset_name == "chexpert":
            args.dataset_dir = paths["chexpert"]
        elif args.dataset_name == "mimic-chexpert":
            args.dataset_dir = [paths["chexpert"], paths["mimic"]]
        elif args.dataset_name == "nih":
            args.dataset_dir = paths["nih"]
        else:
            raise ValueError("Unrecognized path config.")
    return args


def cli_main(args):
    # ------------
    # data
    # ------------
    transform_list = [
        transforms.RandomResizedCrop(args.im_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        RandomGaussianBlur(),
        AddGaussianNoise(snr_range=(4, 8)),
        HistogramNormalize(),
        TensorToRGB(),
    ]
    data_module = XrayDataModule(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_two_images=False,
        train_transform=Compose(transform_list),
        val_transform=Compose(transform_list),
        test_transform=Compose(transform_list),
    )

    # ------------
    # model
    # ------------
    model = MoCoModule.load_from_checkpoint(args.ckpt, arch=args.arch,
                                            feature_dim=args.feature_dim,
                                            queue_size=args.queue_size,
                                            ).model.encoder_q


    for batch in data_module.train_dataloader():
        images, targets = batch['image'].cuda(), batch['labels'].cuda()



if __name__ == "__main__":
    args = build_args()
    cli_main(args)
