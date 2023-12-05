
import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform



def build_loader():
    dataset_train, num_classes = build_dataset(is_train=True, data_path='D:\Learning\Grad_0\Project\Swin-Transformer\data\dataset\ROI', num_classes=10)
    print("Successfully built train dataset")

    dataset_val, _ = build_dataset(is_train=False, data_path='D:\Learning\Grad_0\Project\Swin-Transformer\data\dataset\shell3', num_classes=10)
    print("Successfully built val dataset")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val


dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader()

# 打印训练集和验证集的大小
print(len(dataset_train), len(dataset_val))