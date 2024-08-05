import argparse
import os
import math
import time
import math
from typing import Optional
from pydantic import BaseModel, Field, computed_field


class Options(BaseModel):
    dataset: str = "h36m"
    keypoints: str = "cpn_ft_h36m_dbb"
    data_augmentation: bool = True
    reverse_augmentation: bool = False
    test_augmentation: bool = True
    crop_uv: int = 0
    root_path: str = "dataset/"
    actions: str = "*"
    downsample: int = 1
    subset: float = 1.0
    stride: int = 1
    gpu: str = "0"
    train: bool = True
    test: bool = False
    nepoch: int = 50
    batch_size: int = Field(
        default=32, description="can be changed depending on your machine"
    )
    lr: float = 2e-4
    lr_decay_large: float = 0.98
    large_decay_epoch: int = 5
    workers: int = 8
    lr_decay: float = 0.98
    frames: int = 27
    checkpoint: str = ""
    previous_dir: str = ""
    n_joints: int = 17
    out_joints: int = 17
    out_all: int = 1
    in_channels: int = 2
    out_channels: int = 3
    previous_best_threshold: float = math.inf
    previous_name: str = ""
    mvf_kernel: int = 7
    manual_seed: Optional[int] = None
    subjects_train: str = "S1,S5,S6,S7,S8"
    subjects_test: str = "S9,S11"

    @computed_field
    @property
    def pad(self) -> int:
        return (self.frames - 1) // 2
