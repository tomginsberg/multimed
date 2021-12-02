"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset


class CheXpertDataset(BaseDataset):
    """
    Data loader for CheXpert data set.

    Args:
        directory: Base directory for data set with subdirectory
            'CheXpert-v1.0-small'.
        split: String specifying split.
            options include:
                'all': Include all splits.
                'train': Include training split.
                'val': Include validation split.
        label_list: String specifying labels to include. Default is 'all',
            which loads all labels.
        transform: A composable transform list to be applied to the data.


    Irvin, Jeremy, et al. "Chexpert: A large chest radiograph dataset with
    uncertainty labels and expert comparison." Proceedings of the AAAI
    Conference on Artificial Intelligence. Vol. 33. 2019.

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    def __init__(
            self,
            directory: Union[str, os.PathLike],
            split: str = "train",
            label_list: Union[str, List[str]] = "all",
            subselect: Optional[str] = None,
            transform: Optional[Callable] = None,
            fraction: float = 1.0,  # Fraction: 0.01, 0.1, 1.0 of the fine-tuning
            **kwargs
    ):
        super().__init__(
            "chexpert_v1", directory, split, label_list, subselect, transform, fraction
        )

        if label_list == "all":
            self.label_list = self.default_labels()
        else:
            self.label_list = label_list

        self.metadata_keys = [
            "Patient ID",
            "Path",
            "Sex",
            "Age",
            "Frontal/Lateral",
            "AP/PA",
        ]

        if self.split == "train":
            self.csv_path = self.directory / "CheXpert-v1.0-small" / "train.csv"
            self.csv = pd.read_csv(self.directory / self.csv_path)
        elif self.split == "val":
            self.csv_path = self.directory / "CheXpert-v1.0-small" / "valid.csv"
            self.csv = pd.read_csv(self.directory / self.csv_path)
        elif self.split == "all":
            self.csv_path = self.directory / "train.csv"
            self.csv = pd.concat(
                [
                    pd.read_csv(self.directory / "CheXpert-v1.0-small" / "train.csv"),
                    pd.read_csv(self.directory / "CheXpert-v1.0-small" / "valid.csv"),
                ]
            )
        else:
            logging.warning(
                "split {} not recognized for dataset {}, "
                "not returning samples".format(split, self.__class__.__name__)
            )
        self.csv = self.preproc_csv(self.csv, self.subselect)
        self.csv = self.filter_csv(self.csv)

    @staticmethod
    def default_labels() -> List[str]:
        return ["Pneumonia"]
        # [
        #     "No Finding",
        #     "Enlarged Cardiomediastinum",
        #     "Cardiomegaly",
        #     "Lung Opacity",
        #     "Lung Lesion",
        #     "Edema",
        #     "Consolidation",
        #     "Pneumonia",
        #     "Atelectasis",
        #     "Pneumothorax",
        #     "Pleural Effusion",
        #     "Pleural Other",
        #     "Fracture",
        #     "Support Devices",
        # ]

    @staticmethod
    def preproc_csv(csv: pd.DataFrame, subselect: Optional[str]) -> pd.DataFrame:
        if csv is not None:
            # Get the fraction of the database for fine-tuning
            # csv.sample(frac=self.fraction)

            csv["Patient ID"] = csv["Path"].str.extract(pat="(patient\\d+)")
            csv["view"] = csv["Frontal/Lateral"].str.lower()

            csv = csv.dropna(subset=['Pneumonia'])
            csv = csv[csv.Pneumonia != -1]

            if subselect is not None:
                csv = csv.query(subselect)

        return csv

    def __len__(self) -> int:
        length = 0
        if self.csv is not None:
            length = len(self.csv)

        return length

    def __getitem__(self, idx: int) -> Dict:
        assert self.csv is not None
        exam = self.csv.iloc[idx]

        filename = self.directory / exam["Path"]
        image = self.open_image(filename)

        metadata = self.retrieve_metadata(idx, filename, exam)

        # retrieve labels while handling missing ones for combined data loader
        labels = np.array(exam.reindex(self.label_list)[self.label_list]).astype(
            np.float
        )

        sample = {"image": image, "labels": labels, "metadata": metadata}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
