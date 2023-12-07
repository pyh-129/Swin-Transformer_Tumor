# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from torch.utils.data import Dataset, Subset

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config,k_folds,current_fold):
    config.defrost()
    # dataset_train, config.MODEL.NUM_CLASSES = build_dataset(config=config,k_folds=5)
    dataset,nb_classes = build_dataset(config=config,k_folds=k_folds,current_fold=current_fold)
    dataset_train = Subset(dataset, dataset.train_indices)
    dataset_val = Subset(dataset, dataset.test_indices)
    config.freeze()
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
  
    # config.freeze()

    # train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
    # val_loader = DataLoader(val_data,batch_size=32,shuffle=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
   
    # config.freeze()
    # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    # dataset_val, _ = build_dataset(is_train=False, config=config)
    # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train,
    #     batch_size=config.DATA.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=0,
    #     pin_memory=False,
    #     drop_last=True,
    # )

    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val,
    #     batch_size=config.DATA.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=False,
    #     drop_last=False
    # )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(config,k_folds,current_fold):
    # dataset = IN22KDATASET(config.DATA.DATA_PATH, is_train)
    dataset = IN22KDATASET(config.DATA.DATA_PATH,k_folds,current_fold)
    nb_classes = config.MODEL.NUM_CLASSES

    return dataset, nb_classes


'''
def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
'''
