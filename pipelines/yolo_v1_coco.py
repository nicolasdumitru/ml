import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF
from torchvision.datasets import CocoDetection

import numpy as np

from pathlib import Path
from typing import Union


def letterbox(
    image: torch.Tensor, bboxes: torch.Tensor, size: np.int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The `boxes` tensor is expected to contain bounding boxes in format [x, y, w, h],
    where x, y are absolute coordinates of the bounding box top left corner and w and
    h are the width and height of the bounding box in pixels.
    """

    # x = TF.pil_to_tensor(sample)
    _, h_orig, w_orig = image.shape
    scale = np.float32(size / max(h_orig, w_orig))
    image = TF.resize(image, size=None, max_size=size)
    _, h_scaled, w_scaled = image.shape
    assert np.isclose(h_orig * scale, h_scaled) and np.isclose(w_orig * scale, w_scaled)

    bboxes[:, 0:4] *= scale
    pad_size = np.int32((size - min(w_scaled, h_scaled)) / 2)
    if w_scaled < h_scaled:
        image = TF.pad(image, [pad_size, 0])  # left/right padding
        bboxes[:, 0] += pad_size
    else:
        image = TF.pad(image, [0, pad_size])  # top/bottom padding
        bboxes[:, 1] += pad_size
    return image, bboxes


class YoloCoco(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        image_size: np.int32 = 448,
        grid_size: np.int32 = 7,
    ):
        self.coco = CocoDetection(root, annFile)
        self.num_classes = len(self.coco.coco.getCatIds())
        self.cat_id_mapping = {
            cat_id: idx for idx, cat_id in enumerate(self.coco.coco.getCatIds())
        }  # make the category IDs contiguous
        self.image_size = image_size
        self.grid_size = grid_size

    def __len__(self):
        return len(self.coco)

    def _get_bboxes(self, label):
        bboxes = torch.zeros(  # we rely on zeroing for one hot encoding
            (len(label), 4 + self.num_classes)
        )  # x, y, w, h, confidence, one-hot encoded classes
        for i, ann in enumerate(label):
            bboxes[i, 0:4] = torch.tensor(ann["bbox"])  # x, y, w, h
            bboxes[i, 5] = 1
            cat_id = self.cat_id_mapping[ann["category_id"]]
            # ohot = F.one_hot(torch.tensor(cat_id), self.num_classes).to(bboxes.dtype)
            bboxes[i, 5 + cat_id] = 1
        return bboxes

    def _yolo_label(self, bboxes):
        label = bboxes  # TODO
        return self, label

    def __getitem__(self, index):
        image, label = self.coco[index]
        image = TF.pil_to_tensor(image)
        bboxes = self._get_bboxes(label)
        image, bboxes = letterbox(image, bboxes, self.image_size)
        label = self._yolo_label(bboxes)
        return image, label
