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

        self.data_path = root 
        self.single_path = os.path.join(self.data_path, r'shell3')

        self.class_list = os.listdir(self.single_path)

        self.benign = os.listdir(os.path.join(self.single_path, self.class_list[0]))
        self.benign.sort(key=lambda x: int(x[:-4]))
        self.malignant = os.listdir(os.path.join(self.single_path, self.class_list[1]))
        self.malignant.sort(key=lambda x: int(x[:-4]))
        self.data_list = self.benign + self.malignant

        benign_fold_size = len(self.benign) // k_folds
        malignant_fold_size = len(self.malignant) // k_folds

        self.benign_folds = []
        self.malignant_folds = []

       
        # Calculate the size of each fold
        fold_size = len(self.data_list) // k_folds

        self.folds = []

        # Create the folds
        for i in range(k_folds):
            benign_start = i * benign_fold_size
            benign_end = benign_start + benign_fold_size if i != k_folds - 1 else len(self.benign)
            self.benign_folds.append(range(benign_start, benign_end))

            malignant_start = i * malignant_fold_size
            malignant_end = malignant_start + malignant_fold_size if i != k_folds - 1 else len(self.malignant)
            self.malignant_folds.append(range(malignant_start, malignant_end))

    
        # Get the indices for the current fold
        self.test_indices = list(self.benign_folds[current_fold]) + list(self.malignant_folds[current_fold])
        self.train_indices = [idx for fold in (self.benign_folds + self.malignant_folds) if fold is not self.test_indices for idx in fold]

       
    def __getitem__(self, index):
    # Use the current fold's train/test indices to access the data
        actual_index = self.train_indices[index] if index < len(self.train_indices) else self.test_indices[index - len(self.train_indices)]
        if actual_index < len(self.benign):
            images = self._load_image(os.path.join(self.single_path, self.class_list[0], self.benign[actual_index]))
            images = torch.from_numpy(images).to(torch.float32).permute(2,0,1)
            target = torch.tensor(0.0).to(torch.float32)
        else:
            images = self._load_image(os.path.join(self.single_path, self.class_list[1], self.malignant[actual_index - len(self.benign)]))
            images = torch.from_numpy(images).to(torch.float32).permute(2,0,1)
            target = torch.tensor(1.0).to(torch.float32)

        return images, target


    def __len__(self):
        return len(self.train_indices)+len(self.test_indices)
    def _load_image(self, path):
        try:
            # Load the image using numpy
            im = np.load(path).transpose(0,1,2)
            # print(im.shape)
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
