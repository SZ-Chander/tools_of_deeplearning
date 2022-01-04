# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 23:55:23 2021

@author: admin
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
# threshold表示: Total number of array elements to be print(输出数组的元素数目)
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
random.seed(1)



class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None, in_size = [512,512]):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.label_path_list = list()
        self.in_size = in_size

        # 获取mask的path
        self._get_img_path()

    def __getitem__(self, index):

        path_label = self.label_path_list[index]
        path_img = path_label[:-9] + ".png"
        img_pil = Image.open(path_img).convert('RGB')
        img_pil = img_pil.resize((self.in_size[0], self.in_size[1]), Image.BILINEAR)
        # 在神经网络中，图像被表示成[c, h, w]格式或者[n, c, h, w]格式，但如果想要将图像以np.ndarray形式输入，因np.ndarray默认将图像表示成[h, w, c]个格式，需要对其进行转化。
        img_hwc = np.array(img_pil)
        #  print(img_hwc)
        img_chw = img_hwc.transpose((2, 0, 1))
        # 标签
        label_pil = Image.open(path_label).convert('L')   # 灰度图，一通道
        label_pil = label_pil.resize((self.in_size[0], self.in_size[1]), Image.NEAREST)
        label_hw = np.array(label_pil)
        label_chw = label_hw[np.newaxis, :, :]
        label_hw[label_hw != 0] = 1    # 变成二分类的标签

        if self.transform is not None:
            print('=' * 60)
            img_chw_tensor = torch.from_numpy(self.transform(img_chw)).float()
            label_chw_tensor = torch.from_numpy(self.transform(label_chw)).float()
            print('=' * 60)
            print(type(img_chw_tensor))
            print(type(img_chw_tensor))

            # print(type(img_chw))
            # label_chw=Image.fromarray(label_chw)
            # img_chw_tensor =self.transform(img_chw)
            # label_chw_tensor=self.transform(label_chw)
        else:
            img_chw_tensor = torch.from_numpy(img_chw).float()
            label_chw_tensor = torch.from_numpy(label_chw).float()
            # img_chw=Image.fromarray(img_chw)
            # label_chw=Image.fromarray(label_chw)
            # img_chw_tensor =self.transform(img_chw)
            # label_chw_tensor=self.transform(label_chw)

        return img_chw_tensor, label_chw_tensor

    def __len__(self):
        return len(self.label_path_list)

    def _get_img_path(self):
        file_list = os.listdir(self.data_dir)
        file_list = list(filter(lambda x: x.endswith("_mask.png"), file_list))  # 尾缀是_matte.png是mask
        path_list = [os.path.join(self.data_dir, name) for name in file_list]
        
        random.shuffle(path_list)
        if len(path_list) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        self.label_path_list = path_list