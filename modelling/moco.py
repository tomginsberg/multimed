# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Code adapted from https://github.com/facebookresearch/moco
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
            self,
            encoder_q: nn.Module,
            encoder_k: nn.Module,
            meta_keys: list,
            dim: int = 128,
            K: int = 65536,
            m: float = 0.999,
            T: float = 0.07,
            mlp: bool = False,
    ):
        """
        meta_keys: list of metadata used in negative/positive strategies, e.g., `study`, `lat`, etc.
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        if mlp:  # hack: brute-force replacement
            if hasattr(self.encoder_q, "fc"):  # ResNet models
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
                )
                self.encoder_k.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
                )
            elif hasattr(self.encoder_q, "classifier"):  # Densenet models
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.classifier
                )
                self.encoder_k.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.classifier
                )

        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer(
            "queue", F.normalize(torch.randn(dim, K), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the meta_info queue
        self.meta_keys = meta_keys
        self.queue_meta = dict()
        for meta in self.meta_keys:
            self.register_buffer("queue_" + meta, -1 * torch.ones(K))
            self.queue_meta[meta] = getattr(self, "queue_" + meta)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor, meta_info):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        # gather meta_info
        key_metas = {}
        for meta in self.meta_keys:
            tmp, _ = meta_info[meta]
            key_metas[meta] = concat_all_gather(tmp)

        batch_size = keys.shape[0]

        assert isinstance(self.queue_ptr, Tensor)
        ptr = int(self.queue_ptr)
        assert (
                self.K % batch_size == 0
        ), f"batch_size={batch_size}, K={self.K}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        for meta in self.meta_keys:
            self.queue_meta[meta][ptr:ptr + batch_size] = key_metas[meta]

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: Tensor, idx_unshuffle: Tensor) -> Tensor:
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q: Tensor, im_k: Tensor, meta_info) -> Tuple[Tensor, Tensor]:
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        l_neg = self.get_negative_samples(l_neg, meta_info)

        # q = [q1, q2, q3]
        # k = [k1, k2, k3]

        # l_pos = [[q1 @ k1], [q2 @ k2]...]
        # l_neg = [[q1 @ queue1, q2 @ queue1, ...], [q1 @ queue2, q2 @ queue2, ...], ...]
        # logits = [[q1 @ k1, q1 @ queue1, q2 @ queue1, ...], [q2 @ k2, q1 @ queue2, q2 @ queue2, ...], ...]

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    @ torch.no_grad()
    def get_negative_samples(self, neg_logits, meta_info):
        """
        Input:
            meta_info (dict)
        Output:
            all the negative samples (Tensor)
        """

        # first strategy: based on disease
        _, query_disease = meta_info['disease']
        _, query_id = meta_info['id']

        # [[q1 @ key1, q1 @ key2, ...], [q2 @ key1, q2 @ key2, ...], ...] (N * K)
        same_disease = query_disease.unsqueeze(1) == self.queue_meta['disease'].unsqueeze(0)
        diff_id = query_id.unsqueeze(1) != self.queue_meta['id'].unsqueeze(0)

        hard_neg = diff_id & same_disease
        easy_neg = diff_id & torch.logical_not(same_disease)

        hard_weight, easy_weight = torch.tensor([1]), torch.tensor([0])
        neg_logits[hard_neg] += torch.log(hard_weight)
        neg_logits[easy_neg] += torch.log(easy_weight)

        return neg_logits


# utils
@torch.no_grad()
def concat_all_gather(tensor: Tensor) -> Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)

    return output