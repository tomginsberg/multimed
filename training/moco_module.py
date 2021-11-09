"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.models as models

from modelling.moco import MoCo


class MoCoModule(pl.LightningModule):
    def __init__(
            self,
            arch,
            feature_dim,
            queue_size,
            use_mlp=False,
            learning_rate=1.0,
            momentum=0.9,
            weight_decay=1e-4,
            epochs=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs

        # build model
        self.model = MoCo(
            encoder_q=models.__dict__[arch](num_classes=feature_dim),
            encoder_k=models.__dict__[arch](num_classes=feature_dim),
            dim=feature_dim,
            K=queue_size,
            mlp=use_mlp,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, image0, image1):
        return self.model(image0, image1)

    def training_step(self, batch, batch_idx):
        im0, im1, id0, id1, lab0, lab1 = batch["key_image"], batch["query_image"],\
                                         batch["key_id"], batch["query_id"], \
                                         batch["key_labels"], batch["query_labels"]

        meta_info = {'id': (id0, id1), 'disease': (lab0, lab1)}
        output, target = self(im0, im1, meta_info=meta_info)

        # metrics
        loss_val = self.loss_fn(output, target)
        acc = self.train_acc(output, target)
        self.log("train_metrics/loss", loss_val)
        self.log("train_metrics/accuracy", acc, on_step=True, on_epoch=False)

        return loss_val

    def training_epoch_end(self, outputs):
        self.log("train_metrics/epoch_accuracy", self.train_acc.compute())
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        self.log("val_metrics/epoch_accuracy", self.val_acc.compute())
        self.val_acc.reset()

    def validation_step(self, batch, batch_idx):
        image0, image1 = batch["image0"], batch["image1"]

        output, target = self(image0, image1)

        # metrics
        loss_val = self.loss_fn(output, target)
        acc = self.val_acc(output, target)
        self.log("val_metrics/loss", loss_val)
        self.log("val_metrics/accuracy", acc, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--arch", default="densenet121", type=str)
        parser.add_argument("--feature_dim", default=256, type=int)
        parser.add_argument("--queue_size", default=65536, type=int)
        parser.add_argument("--use_mlp", default=False, type=bool)
        parser.add_argument("--learning_rate", default=1.0, type=float)
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight-decay", default=1e-4, type=float)

        return parser
