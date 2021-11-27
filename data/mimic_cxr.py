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


class MimicCxrJpgDataset(BaseDataset):
    """
    Data loader for MIMIC CXR data set.
    Args:
        directory: Base directory for data set.
        split: String specifying split.
            options include:
                'all': Include all splits.
                'train': Include training split.
                'val': Include validation split.
                'test': Include testing split.
        label_list: String specifying labels to include. Default is 'all',
            which loads all labels.
        transform: A composable transform list to be applied to the data.
    """

    def __init__(
            self,
            directory: Union[str, os.PathLike],
            split: str = "train",
            label_list: Union[str, List[str]] = "all",
            subselect: Optional[str] = None,
            transform: Optional[Callable] = None,
            fraction: float = 1.0
    ):
        super().__init__(
            "mimic-cxr-jpg", directory, split, label_list, subselect, transform, fraction
        )

        if label_list == "all":
            self.label_list = self.default_labels()
        else:
            self.label_list = label_list

        self.metadata_keys = [
            "dicom_id",
            "subject_id",
            "study_id",
            "PerformedProcedureStepDescription",
            "ViewPosition",
            "Rows",
            "Columns",
            "StudyDate",
            "StudyTime",
            "ProcedureCodeSequence_CodeMeaning",
            "ViewCodeSequence_CodeMeaning",
            "PatientOrientationCodeSequence_CodeMeaning",
            "gender",
            "anchor_age",
            "ethnicity"
        ]

        self.label_csv_path = (
                self.directory / "2.0.0" / "mimic-cxr-2.0.0-chexpert.csv.gz"
        )
        self.meta_csv_path = (
                self.directory / "2.0.0" / "mimic-cxr-2.0.0-metadata-extra.csv.gz"
        )
        self.split_csv_path = self.directory / "2.0.0" / "mimic-cxr-2.0.0-split.csv.gz"
        if self.split in ("train", "val", "test"):
            split_csv = pd.read_csv(self.split_csv_path)["split"].str.contains(
                self.split
            )
            meta_csv = pd.read_csv(self.meta_csv_path)[split_csv].set_index(
                ["subject_id", "study_id"]
            )
            label_csv = pd.read_csv(self.label_csv_path).set_index(
                ["subject_id", "study_id"]
            )

            self.csv = meta_csv.join(label_csv).reset_index()
        elif self.split == "all":
            meta_csv = pd.read_csv(self.meta_csv_path).set_index(
                ["subject_id", "study_id"]
            )
            label_csv = pd.read_csv(self.label_csv_path).set_index(
                ["subject_id", "study_id"]
            )

            self.csv = meta_csv.join(label_csv).reset_index()
        else:
            logging.warning(
                "split {} not recognized for dataset {}, "
                "not returning samples".format(split, self.__class__.__name__)
            )

        self.csv = self.preproc_csv(self.csv, self.subselect)
        self.csv = self.filter_csv(self.csv)

    @staticmethod
    def default_labels() -> List[str]:
        return [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

    def preproc_csv(self, csv: pd.DataFrame, subselect: Optional[str]) -> pd.DataFrame:
        if csv is not None:

            def format_view(s):
                if s in ("AP", "PA", "AP|PA"):
                    return "frontal"
                elif s in ("LATERAL", "LL"):
                    return "lateral"
                else:
                    return None

            csv["view"] = csv.ViewPosition.apply(format_view)

            if subselect is not None:
                csv = csv.query(subselect)

        return csv

    def __len__(self):
        length = 0
        if self.csv is not None:
            length = len(self.csv)

        return length

    def get_filename(self, exam):
        filename = self.directory / "2.0.0" / "files"
        subject_id = str(exam["subject_id"])
        study_id = str(exam["study_id"])
        dicom_id = str(exam["dicom_id"])
        return (
                filename
                / "p{}".format(subject_id[:2])
                / "p{}".format(subject_id)
                / "s{}".format(study_id)
                / "{}.jpg".format(dicom_id)
        )

    def __getitem__(self, idx: int) -> Dict:
        assert self.csv is not None
        exam = self.csv.iloc[idx]

        filename = self.get_filename(exam)
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
