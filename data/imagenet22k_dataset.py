import os
import json
from abc import ABC

import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


from torch.utils.data import random_split, Subset

class IN22KDATASET(data.Dataset):
    def __init__(self, root, k_folds,current_fold):
        super(IN22KDATASET, self).__init__()

        self.data_path = root #D:\Learning\Grad_0\Project\Swin-Transformer_Tumor\Swin-Transformer_Tumor\data\dataset
        self.single_path = os.path.join(self.data_path, r'shell3')

        self.class_list = os.listdir(self.single_path)

        self.benign = os.listdir(os.path.join(self.single_path, self.class_list[0]))
        self.benign.sort(key=lambda x: int(x[:-4]))
        self.malignant = os.listdir(os.path.join(self.single_path, self.class_list[1]))
        self.malignant.sort(key=lambda x: int(x[:-4]))
        self.data_list = self.benign + self.malignant

        # Calculate the size of each fold
        fold_size = len(self.data_list) // k_folds

        # Create a list to hold the data indices for each fold
        self.folds = []

        # Create the folds
        for i in range(k_folds):
            start = i * fold_size
            end = start + fold_size if i != k_folds - 1 else len(self.data_list)
            self.folds.append(range(start, end))
    
        # Get the indices for the current fold
        self.test_indices = self.folds[current_fold]
        self.train_indices = [idx for fold in self.folds if fold is not self.test_indices for idx in fold]

       
    def __getitem__(self, index):
    # Use the current fold's train/test indices to access the data
        sample = self.data_list[index]
        if index in self.train_indices:
            # Training data handling
            images = self._load_image(os.path.join(self.single_path, self.class_list[0], sample))
            images = torch.from_numpy(images).to(torch.float32)

            # target == benign
            target = torch.tensor(0.0).to(torch.float32)

            return images, target
        elif index in self.test_indices:
            # Test data handling
            images = self._load_image(os.path.join(self.single_path, self.class_list[1], sample))
            images = torch.from_numpy(images).to(torch.float32)

            # target == malignant
            target = torch.tensor(1.0).to(torch.float32)

            return images, target

    def __len__(self):
        return len(self.data_list)
    def _load_image(self, path):
        try:
            # Load the image using numpy
            im = np.load(path)
        except Exception as e:
            print(f"ERROR IMG LOADED: {path}, due to {e}")
            # If an error occurs, generate a random image
            random_img = np.random.rand(35, 112, 112) * 255
            im = np.int32(random_img)
        return im
# class IN22KDATASET(data.Dataset):
#     def __init__(self, root, is_train):
#         super(IN22KDATASET, self).__init__()

#         self.data_path = root
#         if is_train:
#             self.single_path = os.path.join(self.data_path, r'训练集文件夹路径')
#         else:
#             self.single_path = os.path.join(self.data_path, r'测试集文件夹路径')

#         self.class_list = os.listdir(self.single_path)

#         self.benign = os.listdir(os.path.join(self.single_path, self.class_list[0]))
#         self.benign.sort(key=lambda x: int(x[:-4]))
#         self.malignant = os.listdir(os.path.join(self.single_path, self.class_list[1]))
#         self.malignant.sort(key=lambda x: int(x[:-4]))
#         self.data_list = self.benign + self.malignant

#     def _load_image(self, path):
#         try:
#             im = np.load(path)
#         except:
#             print("ERROR IMG LOADED: ", path)
#             random_img = np.random.rand(35, 112, 112) * 255
#             im = np.int32(random_img)
#         return im

#     def __getitem__(self, index):

#         sample = self.data_list[index]
#         if index < len(self.benign):
#             # images
#             images = self._load_image(os.path.join(self.single_path, self.class_list[0], sample))
#             images = torch.from_numpy(images).to(torch.float32)

#             # target == benign
#             target = torch.tensor(0.0).to(torch.float32)

#             return images, target
#         else:
#             # images
#             images = self._load_image(os.path.join(self.single_path, self.class_list[1], sample))
#             images = torch.from_numpy(images).to(torch.float32)

#             # target == malignant
#             target = torch.tensor(1.0).to(torch.float32)

#             return images, target

#     def __len__(self):
#         return len(self.data_list)


if __name__ == '__main__':
    pass
