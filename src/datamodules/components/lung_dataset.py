import os
import io
from pathlib import Path
from typing import Any, Callable, Optional
import pydicom
from copy import deepcopy, copy
import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import rootutils
rootutils.setup_root(search_from=__file__, pythonpath=True)
from src.utils.dicom.io import *
from src.utils.dicom.mask import find_bbox

class LungDataset(Dataset):
    def __init__(
        self,
        data_dir,
        label_dir,
        pred_dir=None,
        input_shape=[512, 512]
    ) -> None:
        """BaseDataset.

        Args:
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
        """

        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.data = []
        self.input_shape = input_shape
        self.pred_dir = pred_dir
        self.prepare()
    
    def prepare(self):
        bnids = os.listdir(self.data_dir)
        print(len(bnids))
        for bnid in bnids:
            data_dir = self.data_dir / bnid
            csv_dir =   self.label_dir / 'csv'
            dicom_files = [
                            f for f in os.listdir(data_dir)
                            if f.lower().endswith(".jpg")
                            ]
            if self.pred_dir is not None: 
                pred_label_dir = self.pred_dir / "csv"
                pred_logit_dir = self.pred_dir / "logit"
                raw_pred_df = pd.read_csv(pred_label_dir / f"{bnid}.csv")
            raw_df = pd.read_csv(csv_dir / f"{bnid}.csv")
            df = raw_df[raw_df['mask_id'] != -1]
            dicom_files = sorted(dicom_files)
            for dicom_file in dicom_files: 
                slice_idx = int(dicom_file.split('.')[0].split('_')[-1])
                slice_node = raw_df[raw_df['slice_id'] == slice_idx]
                raw_id = slice_node['raw_id'].tolist()[0]
                node_df = df[df['slice_id'] == slice_idx].copy()
                label_mask_files = []
                if self.pred_dir:
                    logit_file = dicom_file.split(".")[0] + ".jpg"
                    pred_df = raw_pred_df[raw_pred_df['slice_id'] == slice_idx]
                    
                if len(node_df) > 0:
                    for index, node in node_df.iterrows(): 
                        name = f"slice_{slice_idx:04d}_{node['mask_id']}.jpg"
                        # print(name)
                        label_mask_files.append(name)

                sample = {  "bnid": bnid,
                            "slice_idx": slice_idx,
                            "raw_id": raw_id,
                            "filename": dicom_file,
                            "label": slice_node.copy(),
                            "pred": None,
                            }
                if self.pred_dir:
                    sample["pred"]= {
                                    "logit": logit_file,
                                    "class": pred_df
                                    }
                self.data.append(sample)

    def __getitem__(self, index: int) -> Any:
        sample = deepcopy(self.data[index])
        bnid, slice_idx = sample["bnid"], sample["slice_idx"]
        data_dir = self.data_dir / bnid
        mask_dir =  self.label_dir / 'mask' / bnid
        ret = True
        input = cv2.imread(data_dir / sample["filename"])
        if input is None:
            input = np.zeros(self.input_shape, dtype=np.float32)
            ret = False
        label_mask_output = None
        bboxes = []
        for index, node in sample["label"].iterrows():
            name = f"slice_{slice_idx:04d}_{node['mask_id']}.jpg"
                # print(name)
            label_mask = cv2.imread(mask_dir / name)[..., 0].astype(float) / 255
            if label_mask_output is None:
                label_mask_output = label_mask
            else:
                label_mask_output = np.maximum(label_mask_output, label_mask)
            # print(label_mask.shape, np.unique(label_mask))
            _, slice_bbox = find_bbox(label_mask, ratio=True, ssize=label_mask.shape[:2][::-1])
            # print
            bboxes.append(deepcopy(slice_bbox[0]))
        if (sample["label"]["mask_id"] != "-1").all():
            sample["label"]["bbox"] = bboxes
        else:
            sample["label"]["bbox"] = [[0, 0, 1, 1]] * len(sample)
        pred = None
        if sample["pred"]:
            pred = dict()
            pred_logit_dir = self.pred_dir / "logit" / sample["bnid"]
            pred["logit"] = cv2.imread(pred_logit_dir / sample["pred"]["logit"])[..., 0].astype(float) / 255
            pred["class"] = sample["pred"]["class"]
        return {
                "ret": ret,
                "input": input,
                "mask": label_mask_output,
                "class": sample["label"],
                "pred": pred
                }

    def __len__(self) -> int:
        return len(self.data)

if __name__ == "__main__":
    dataset = LungDataset(data_dir=Path(r"data/final/cache"), 
                            label_dir=Path(r"data/final/ground_truth"), 
                            pred_dir=Path(r"data/final/pred"),
                            input_shape=[352, 352])
    print(len(dataset))
    sample = dataset[10]
    print(sample["pred"])

