"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from abc import ABC
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Root class for X-ray data sets.

    The base data set logs parameters as attributes, reducing code duplication
    across the various public X-ray data loaders.

    Args:
        dataset_name: Name of the dataset.
        directory: Location of the data.
        split: One of ('train', 'val', 'test', 'all').
        label_list: A list of labels for the data loader to extract.
        subselect: Argument to pass to `pandas` subselect.
        transform: A set of data transforms.
        fraction: One of (1.0, 0.1, 0.001) of the whole training dataset
    """

    def __init__(
            self,
            dataset_name: str,
            directory: Union[str, os.PathLike],
            split: str,
            label_list: Union[str, List[str]],
            subselect: Optional[str],
            transform: Optional[Callable],
            fraction: float = 1.0,
            **kwargs
    ):
        self.dataset_name = dataset_name

        split_list = ["train", "val", "test", "all"]

        if split not in split_list:
            raise ValueError("split {} not a valid split".format(split))

        self.directory = Path(directory)
        self.csv = None
        self.split = split
        self.label_list = label_list
        self.subselect = subselect
        self.transform = transform
        self.metadata_keys: List[str] = []
        self.fraction = fraction

    def filter_csv(self, csv: pd.DataFrame) -> pd.DataFrame:
        if self.split == "train":
            csv = csv.sample(frac=self.fraction)  # Get fraction of database for fine-tuning

        return csv

    @staticmethod
    def open_image(path: Union[str, os.PathLike]) -> Image:
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("F")

    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    def calc_pos_weights(self) -> float:
        if self.csv is None:
            return 0.0

        pos = (self.csv[self.label_list] == 1).sum()
        neg = (self.csv[self.label_list] == 0).sum()

        neg_pos_ratio = (neg / np.maximum(pos, 1)).values.astype(np.float)

        return neg_pos_ratio

    def retrieve_metadata(
            self, idx: int, filename: Union[str, os.PathLike], exam: pd.Series
    ) -> Dict:
        metadata = {"dataset_name": self.dataset_name, "dataloader class": self.__class__.__name__, "idx": idx}
        for key in self.metadata_keys:
            # cast to string due to typing issues with dataloader
            metadata[key] = str(exam[key])
        metadata["filename"] = str(filename)

        metadata["label_list"] = self.label_list  # type: ignore

        return metadata

    def __repr__(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    @property
    def labels(self) -> Union[str, List[str]]:
        return self.label_list
