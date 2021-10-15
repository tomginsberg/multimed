"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from argparse import ArgumentParser
from os.path import basename
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger
from random_word import RandomWords
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


def build_args(arg_defaults=None):
    pl.seed_everything(1234)
    data_config = "configs/data.yaml"
    tmp = arg_defaults
    arg_defaults = {
        "accelerator": "ddp",
        "max_epochs": 200,
        "gpus": [0, 1],
        "num_workers": 10,
        "batch_size": 256,
        "callbacks": [],
    }
    if tmp is not None:
        arg_defaults.update(tmp)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--im_size", default=224, type=int)
    parser.add_argument("--run_name", default=None, type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = XrayDataModule.add_model_specific_args(parser)
    parser = MoCoModule.add_model_specific_args(parser)
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
        else:
            raise ValueError("Unrecognized path config.")

    # ------------
    # checkpoints
    # ------------
    checkpoint_dir = Path(args.default_root_dir) / "checkpoints"
    sub_dirs = {basename(str(x)) for x in checkpoint_dir.glob('*')}
    if args.run_name is not None:
        name = args.run_name
        assert name not in sub_dirs, f'{name} already in {checkpoint_dir}'
    else:
        while True:
            rw = RandomWords()
            name = f'{rw.get_random_word()}-{rw.get_random_word()}'
            if name not in sub_dirs:
                break

    args.logger = WandbLogger(project='moco', entity='ml4health', name=name)
    checkpoint_dir = checkpoint_dir / name
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    elif args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    args.callbacks.append(
        pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, verbose=True, monitor='train_metrics/loss', mode='min',
                                     save_last=True, auto_insert_metric_name=False)
    )
    args.logger.log_hyperparams(args.__dict__)
    args.log_every_n_steps = 1
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
        use_two_images=True,
        train_transform=Compose(transform_list),
        val_transform=Compose(transform_list),
        test_transform=Compose(transform_list),
    )

    # ------------
    # model
    # ------------
    model = MoCoModule(
        arch=args.arch,
        feature_dim=args.feature_dim,
        queue_size=args.queue_size,
        use_mlp=args.use_mlp,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.max_epochs,
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    args = build_args()
    cli_main(args)
