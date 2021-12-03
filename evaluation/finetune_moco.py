"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# Adapted from
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.models as models

# import adabound
import wandb


def filter_nans(logits, labels):
    logits = logits[~torch.isnan(labels)]
    labels = labels[~torch.isnan(labels)]

    return logits, labels


def validate_pretrained_model(state_dict, pretrained_file):
    # sanity check to make sure we're not altering weights
    pretrained_dict = torch.load(pretrained_file, map_location="cpu")["state_dict"]
    model_dict = dict()
    for k, v in pretrained_dict.items():
        if "model.encoder_q" in k:
            model_dict[k[len("model.encoder_q."):]] = v

    for k in list(model_dict.keys()):
        # only ignore fc layer
        if "classifier.weight" in k or "classifier.bias" in k:
            continue
        if "fc.weight" in k or "fc.bias" in k:
            continue

        assert (
                state_dict[k].cpu() == model_dict[k]
        ).all(), f"{k} changed in linear classifier training."


class FineTuneModule(pl.LightningModule):
    def __init__(
            self,
            arch,
            num_classes,
            label_list,
            val_pathology_list,
            pretrained_file=None,
            learning_rate=1e-3,
            pos_weights=None,
            epochs=5,
            linear=False
    ):
        super().__init__()

        pretrained_file = str(pretrained_file)

        self.label_list = label_list
        self.val_pathology_list = val_pathology_list
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.pretrained_file = pretrained_file
        self.linear = linear

        # load the pretrained model
        if pretrained_file is not None:
            self.pretrained_file = str(self.pretrained_file)

            # # download the model if given a url
            # if "https://" in pretrained_file:
            #     url = self.pretrained_file
            #     self.pretrained_file = Path.cwd() / pretrained_file.split("/")[-1]
            #     download_model(url, self.pretrained_file)

            pretrained_dict = torch.load(self.pretrained_file)["state_dict"]
            if "model.encoder_q.classifier.0.weight" in pretrained_dict.keys():
                pretrained_dict["model.encoder_q.classifier.weight"] = pretrained_dict["model.encoder_q.classifier.0.weight"]
                del pretrained_dict["model.encoder_q.classifier.0.weight"]

                pretrained_dict["model.encoder_q.classifier.bias"] = pretrained_dict["model.encoder_q.classifier.0.bias"]
                del pretrained_dict["model.encoder_q.classifier.0.bias"]

                del pretrained_dict["model.encoder_q.classifier.2.bias"]
                del pretrained_dict["model.encoder_q.classifier.2.weight"]

            state_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("model.encoder_q."):
                    k = k.replace("model.encoder_q.", "")
                    state_dict[k] = v

            if "model.encoder_q.classifier.weight" in pretrained_dict.keys():
                feature_dim = pretrained_dict[
                    "model.encoder_q.classifier.weight"
                ].shape[0]
                in_features = pretrained_dict[
                    "model.encoder_q.classifier.weight"
                ].shape[1]

                self.model = models.__dict__[arch](num_classes=feature_dim)
                # self.model = torchvision.models.densenet121(pretrained=True)
                self.model.load_state_dict(state_dict)
                del self.model.classifier

                # Linear layer fine-tuning if self.linear = True, else end-to-end fine tuning
                if self.linear:
                    self.model.eval()
                    for param in self.model.parameters():
                        param.requires_grad = False

                else:
                    for param in self.model.parameters():
                        param.requires_grad = True

                self.model.add_module(
                    "classifier", torch.nn.Linear(in_features, num_classes)
                )

            elif "model.encoder_q.fc.weight" in pretrained_dict.keys():
                feature_dim = pretrained_dict["model.encoder_q.fc.weight"].shape[0]
                in_features = pretrained_dict["model.encoder_q.fc.weight"].shape[1]

                self.model = models.__dict__[arch](num_classes=feature_dim)
                self.model.load_state_dict(state_dict)
                del self.model.fc
                self.model.add_module("fc", torch.nn.Linear(in_features, num_classes))
            else:
                raise RuntimeError("Unrecognized classifier.")
        else:
            self.model = models.__dict__[arch](num_classes=num_classes)

        # loss function
        if pos_weights is None:
            pos_weights = torch.ones(num_classes)
        self.register_buffer("pos_weights", pos_weights)
        print(self.pos_weights)

        # metrics
        self.train_acc = torch.nn.ModuleList(
            [torchmetrics.Accuracy() for _ in val_pathology_list]
        )
        self.val_acc = torch.nn.ModuleList(
            [torchmetrics.Accuracy() for _ in val_pathology_list]
        )
        self.val_conf = torch.nn.ModuleList(
            [torchmetrics.ConfusionMatrix(num_classes=2) for _ in val_pathology_list]
        )

        self.val_f1 = torch.nn.ModuleList(
            [torchmetrics.F1(num_classes=1, average='macro') for _ in val_pathology_list]
        )

    def on_epoch_start(self):
        if self.pretrained_file is not None:
            self.model.eval()

    def forward(self, image):
        return self.model(image)

    def loss(self, output, target):
        counts = 0
        loss = 0
        for i in range(len(output)):
            pos_weights, _ = filter_nans(self.pos_weights, target[i])
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weights, reduction="sum"
            )
            bind_logits, bind_labels = filter_nans(output[i], target[i])
            loss = loss + loss_fn(bind_logits, bind_labels)
            counts = counts + bind_labels.numel()

        counts = 1 if counts == 0 else counts
        loss = loss / counts

        return loss

    def training_step(self, batch, batch_idx):
        # forward pass
        output = self(batch["image"])
        target = batch["labels"]

        # calculate loss
        loss_val = self.loss(output, target)

        # metrics
        self.log("train_metrics/loss", loss_val)
        for i, path in enumerate(self.val_pathology_list):
            j = self.label_list.index(path)
            logits, labels = filter_nans(output[:, j], target[:, j])
            if len(logits) == 0:
                break
            self.train_acc[i](logits, labels.int())
            self.log(
                f"train_metrics/accuracy_{path}",
                self.train_acc[i],
                on_step=True,
                on_epoch=False,
            )

        return loss_val

    def training_epoch_end(self, outputs) -> None:
        self.train_acc[0].reset()

    def validation_step(self, batch, batch_idx):
        # forward pass
        output = self(batch["image"])
        target = batch["labels"]

        # calculate loss
        loss_val = self.loss(output, target)

        # metrics
        result_logits = {}
        result_labels = {}
        self.log("val_metrics/loss", loss_val)
        for path in self.val_pathology_list:
            j = self.label_list.index(path)
            logits, labels = filter_nans(output[:, j], target[:, j])

            result_logits[path] = logits
            result_labels[path] = labels

        return {"logits": result_logits, "targets": result_labels}

    def validation_epoch_end(self, outputs):
        # make sure we didn't change the pretrained weights
        # if self.pretrained_file is not None:
        #     validate_pretrained_model(self.model.state_dict(), self.pretrained_file)

        auc_vals = []
        pr_auc_vals = []
        for i, path in enumerate(self.val_pathology_list):
            logits = []
            targets = []
            for output in outputs:
                logits.append(output["logits"][path].flatten())
                targets.append(output["targets"][path].flatten())

            logits = torch.cat(logits)
            targets = torch.cat(targets)
            print(f"path: {path}, len: {len(logits)}")

            self.val_acc[i](logits, targets.int())
            self.val_conf[i](logits, targets.int())
            self.val_f1[i].update(logits, targets.int())

            try:
                auc_val = torchmetrics.functional.auroc(torch.sigmoid(logits), targets.int(), pos_label=1)
                auc_vals.append(auc_val)
            except ValueError:
                auc_val = 0

            try:
                pr = torchmetrics.PrecisionRecallCurve(num_classes=1) # Change the number of classes if the num of classes > 1
                precision, recall, threshold = pr(torch.sigmoid(logits), targets.int())
                pr_auc_val = torch.functional.auc(precision, recall, reorder=True)
                pr_auc_vals.append(pr_auc_val)
            except:
                pr_auc_val = 0

            print(f"path: {path}, auc_val: {auc_val}")

            self.log(
                f"val_metrics/accuracy_{path}",
                self.val_acc[i],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val_metrics/f1_{path}",
                self.val_f1[i],
                on_step=False,
                on_epoch=True,
            )
            self.val_acc[i].reset()
            self.val_f1[i].reset()

            self.log(f"val_metrics/auc_{path}", auc_val)
            self.log("val_metrics/pr_auc", pr_auc_val)

            cf = self.val_conf[i].compute()
            self.val_conf[i].reset()
            print(cf)

            wandb.log({'val_metrics/cf': wandb.Table(columns=['0', '1'], data=list(cf.cpu()))})
            # self.log(f"val_metrics/tp_{path}", cf[1, 1].item())
            # self.log(f"val_metrics/tn_{path}", cf[0, 0].item())
            # self.log(f"val_metrics/fp_{path}", cf[0, 1].item())
            # self.log(f"val_metrics/fn_{path}", cf[1, 0].item())

        self.log("val_metrics/auc_mean", sum(auc_vals) / len(auc_vals))

    def configure_optimizers(self):
        # if self.pretrained_file is None:
        #     model = self.model
        # else:
        #     if hasattr(self.model, "classifier"):
        #         model = self.model.classifier
        #     elif hasattr(self.model, "fc"):
        #         model = self.model.fc
        #     else:
        #         raise RuntimeError("Unrecognized classifier.")

        # optimizer = adabound.AdaBound(self.model.parameters(), lr=self.learning_rate, final_lr=0.1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.2)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--arch", default="densenet121", type=str)
        parser.add_argument("--num_classes", default=1, type=int)
        parser.add_argument("--pretrained_file", default=None, type=str)
        parser.add_argument("--val_pathology_list", nargs="+")
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--pos_weights", default=None, type=float)
        parser.add_argument("--linear", default=False, action='store_true')

        return parser
