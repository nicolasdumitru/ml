import torch
from torch import nn
import torchvision.transforms.v2.functional as TF

import numpy as np

from PIL import Image


def letterbox(
    sample: torch.Tensor, bboxes: torch.Tensor, size: np.int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The `boxes` tensor is expected to contain bounding boxes in format [x, y, w, h],
    where x, y are absolute coordinates of the bounding box top left corner and w and
    h are the width and height of the bounding box in pixels.
    """

    # x = TF.pil_to_tensor(sample)
    _, h_orig, w_orig = sample.shape
    scale = np.float32(size / max(h_orig, w_orig))
    sample = TF.resize(sample, size=None, max_size=size)
    _, h_scaled, w_scaled = sample.shape
    assert np.isclose(h_orig * scale, h_scaled) and np.isclose(w_orig * scale, w_scaled)

    bboxes *= scale
    pad_size = np.int32((size - min(w_scaled, h_scaled)) / 2)
    if w_scaled < h_scaled:
        sample = TF.pad(sample, [pad_size, 0])  # left/right padding
        bboxes[:, 0] += pad_size
    else:
        sample = TF.pad(sample, [0, pad_size])  # top/bottom padding
        bboxes[:, 1] += pad_size
    return sample, bboxes


def get_bboxes(target):
    bboxes = torch.empty((len(target), 4))
    for i, ann in enumerate(target):
        bboxes[i] = torch.tensor(ann["bbox"])  # x, y, w, h
    return bboxes


class YOLOv1CocoTransform(nn.Module):
    '''
    WIP. Not ready yet.
    '''
    def __init__(self, image_size: np.int32 = 448, grid_size: np.int32 = 7):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size

    def forward(self, sample: Image, target: list) -> tuple[torch.Tensor, torch.Tensor]:
        bboxes = get_bboxes(target)
        x = TF.pil_to_tensor(sample)
        x, bboxes = letterbox(x, bboxes, self.image_size)
        return x, bboxes
