# Get dataloaders
# Self design dataset
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import torch
# Our Dataset
class Tumor_Dataset(Dataset):
    def __init__(self, data_info, transform=None):
        """
        直肠肿瘤分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.data_info = data_info  # data_info存储所有图片路径和标签
        self.transform = transform

    def __getitem__(self, index):
        path_img, label1, label2 = self.data_info[index]
        img = Image.open(path_img)
        img = np.array(img)
        max_c = img.shape[2]-1
        img = np.stack((img[:,:,0],img[:,:,min(1,max_c)],img[:,:,min(2,max_c)]))
        
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        img = torch.stack([img])
        return img, label1, label2, path_img

    def __len__(self):
        return len(self.data_info)