import os
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import argparse
import os
import time
from loguru import logger
import cv2
from torchvision import transforms
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


class myDataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            watermark_dir,
            width,
            high

    ):
        self.ids = os.listdir(images_dir)
        self.wids = os.listdir(watermark_dir)
        self.images = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.watermark = [os.path.join(watermark_dir, image_id) for image_id in self.wids]
        # convert str names to class values on masks
        self.width = width
        self.high = high

    def __getitem__(self, i):
        # read data
        cover = cv2.imread(self.images[i])
        cover = cv2.resize(cover, (self.width, self.high)) / 255.0
        watermark = cv2.imread(self.watermark[i])
        watermark = cv2.resize(watermark, (self.width, self.high)) / 255.0

        cover = cover.transpose(2, 0, 1).astype('float32')
        watermark = watermark.transpose(2, 0, 1).astype('float32')
        return cover, watermark

    def __len__(self):
        return len(self.ids)
