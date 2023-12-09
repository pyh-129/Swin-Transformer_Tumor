import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import torch


project_path = r'D:\Learning\Grad_0\Project\Swin-Transformer_Tumor\Swin-Transformer_Tumor\data'
input_path = os.path.join(project_path, r'dataset\shell6')
# print(input_path)
input_benign_path = os.path.join(input_path, os.listdir(input_path)[0])
input_malignant_path = os.path.join(input_path, os.listdir(input_path)[1])

origin_benign_list = os.listdir(input_benign_path)
origin_benign_list.sort(key=lambda x: int(x[:-4]))
origin_malignant_list = os.listdir(input_malignant_path)
origin_malignant_list.sort(key=lambda x: int(x[:-4]))

single_data = origin_benign_list + origin_malignant_list

output_path = os.path.join(project_path, r'dataset2\shell6')
os.makedirs(output_path, exist_ok=True)


output_benign_path = os.path.join(output_path, os.listdir(input_path)[0])
output_malignant_path = os.path.join(output_path, os.listdir(input_path)[1])
os.makedirs(output_benign_path, exist_ok=True)
os.makedirs(output_malignant_path, exist_ok=True)

data_transform = transforms.Compose([
            transforms.Resize((112, 112)),  
            transforms.ToTensor(),  
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

for index in range(len(origin_benign_list)):
    sample = origin_benign_list[index]
    images =np.load(os.path.join(input_benign_path, sample))

    img_transposed = np.transpose(images, (1, 2, 0)) 
    img_transformed_list=[]
    # 切分图像为 35 个单通道的图像
    img_split = np.split(img_transposed, 35, axis=2) 
    
    for i in range(35):
        # print(img_split[i].shape)
        img_pil = np.squeeze(img_split[i])
        # print(img_pil.shape)
        img_pil = Image.fromarray(img_pil)  # Convert numpy array to PIL Image
        img_transformed = data_transform(img_pil).permute(1,2,0)
        

        img_numpy = img_transformed.numpy()
        img_transformed_list.append(img_numpy)

# 使用 numpy.concatenate 将转换后的图像组合成一个新的图像
    img_concatenated = np.concatenate(img_transformed_list, axis=2) # Now img_concatenated is of shape (89, 89, 35)
    # print(img_concatenated.shape)
    # print(os.path.join(output_benign_path),fr'{index}.npy')
    np.save(os.path.join(output_benign_path,fr'{index}.npy'), img_concatenated)



for index in range(len(origin_malignant_list)):
    sample = origin_malignant_list[index]
    images =np.load(os.path.join(input_malignant_path, sample))

    img_transposed = np.transpose(images, (1, 2, 0)) 
    img_transformed_list=[]
    # 切分图像为 35 个单通道的图像
    img_split = np.split(img_transposed, 35, axis=2) 
    
    for i in range(35):
        # print(img_split[i].shape)
        img_pil = np.squeeze(img_split[i])
        # print(img_pil.shape)
        img_pil = Image.fromarray(img_pil)  # Convert numpy array to PIL Image
        img_transformed = data_transform(img_pil).permute(1,2,0)
        

        img_numpy = img_transformed.numpy()
        img_transformed_list.append(img_numpy)

# 使用 numpy.concatenate 将转换后的图像组合成一个新的图像
    img_concatenated = np.concatenate(img_transformed_list, axis=2) # Now img_concatenated is of shape (89, 89, 35)
    # print(img_concatenated.shape)
    # print(os.path.join(output_malignant_path),fr'{index}.npy')
    np.save(os.path.join(output_malignant_path,fr'{index}.npy'), img_concatenated)

