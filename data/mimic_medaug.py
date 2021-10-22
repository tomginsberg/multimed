import os
import random
from typing import Union, Optional, Callable, List

import pandas as pd
from PIL import Image

from data.mimic_cxr import MimicCxrJpgDataset


class MimicPatientPositivePairDataset(MimicCxrJpgDataset):
    def __init__(self,
                 directory: Union[str, os.PathLike],
                 split: str = "train",
                 label_list: Union[str, List[str]] = "all",
                 subselect: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 same_patient=True, same_study=False,
                 diff_study=False,
                 same_laterality=False,
                 diff_laterality=False,
                 same_disease=False,
                 **kwargs):
        """
        Args:
            same_patient (bool): if True, positive pairs come from same patient
            same_study (bool): if True, positive pairs come from same study
            diff_study (bool): if True, positive pairs come from distinct study
            if same_study and diff_study both False: positive pairs come from same patient regardless of study number

            same_laterality (bool): if True, positive pairs come from same laterality
            diff_laterality (bool): if True, positive pairs come from distinct laterality
            if same_laterality and diff_laterality both False: positive pairs come from 
            same patient regardless of laterality

            same_disease (bool): (cheating using underlying labels): if True, images in positive pair must have the same 
            downstream label
        """

        super().__init__(directory=directory, split=split, label_list=label_list, subselect=subselect,
                         transform=transform)

        self.same_patient = same_patient
        self.same_study = same_study
        self.diff_study = diff_study
        self.same_laterality = same_laterality
        self.diff_laterality = diff_laterality
        # self.same_disease = same_disease

    def __getitem__(self, idx):

        exam = self.csv.iloc[idx]
        subject_id = exam.subject_id
        study_id = exam.study_id
        view = exam.view

        filename = self.get_filename(exam)
        # curr_disease = self.csv.at[idx, 'disease']
        # csv_cpy = self.csv.copy()

        if self.same_patient:
            poss_key_paths = self.csv[self.csv['subject_id'] == subject_id]

            if self.same_study:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['study_id'] == study_id]
            if self.diff_study:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['study_id'] != study_id]
            if self.same_laterality:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['view']
                                                    == view]
            if self.diff_laterality:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['view']
                                                    != view]
            # if self.same_disease:
            #     poss_key_paths = poss_key_paths.loc[poss_key_paths['disease']
            #                                         == curr_disease]
            poss_key_paths = poss_key_paths.reset_index(drop=True)
        else:
            poss_key_paths = exam

        query_image = self.open_image(filename)

        try:
            key_exam = poss_key_paths.sample()
            key_image = self.open_image(self.get_filename(key_exam))
        except:
            key_exam = exam
            key_image = query_image

        meta_info = {
            "id": [key_exam.subject_id, subject_id],
            "study": [key_exam.study_id, study_id],
            "lat": [key_exam.view, view],
        }

        if self.transform is not None:
            pos_pair = [self.transform(
                {'image': key_image}), self.transform({'image': query_image})]
        else:
            pos_pair = [key_image, query_image]

        return pos_pair, meta_info, idx


def view_to_int(view):
    if view == "frontal":
        return 0
    elif view == "lateral":
        return 1
    return -1
