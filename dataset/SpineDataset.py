#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import os.path as osp
import numpy as np
import SimpleITK as sitk
from PIL import Image
import random


class SpineDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        super(SpineDataset, self).__init__()
        self.root = root
        if split not in ['train', 'test', 'val']:
            raise RuntimeError("Unexpected split: {}".format(split))
        self.split = split
        self.image_list = sorted(os.listdir(osp.join(self.root, self.split, 'image')))
        self.transform = transform
        if self.split != 'test':
            self.mask_list = sorted(os.listdir(osp.join(self.root, self.split, 'groundtruth')))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = sitk.ReadImage(osp.join(self.root, self.split, 'image', self.image_list[idx]))
        if self.split != 'test':
            mask = sitk.ReadImage(osp.join(osp.join(self.root, self.split, 'groundtruth', self.mask_list[idx])))
            if self.transform:
                image, mask = self.transform(image, mask)
        else:
            if self.transform:
                image = self.transform(image)

        if self.split == 'test':
            return image
        elif self.split == 'train':
            return image, mask
        else:
            return image, mask, osp.join(self.root, self.split, 'groundtruth', self.mask_list[idx])


if __name__ == '__main__':
    def my_transform(image, mask):
        image = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(mask).astype(np.int64)
        # 随机裁剪 512x512 的尺寸
        x = random.randint(0, image.shape[1] - 512)
        y = random.randint(0, image.shape[2] - 512)
        image = image[:, y:y + 512, x:x + 512]
        mask = mask[:, y:y + 512, x:x + 512]
        image = np.expand_dims(image, axis=1)
        image = np.repeat(image, 3, 1)
        image = torch.from_numpy(image)
        # 缩放到 0-1 范围，然后归一化
        image /= image.max()
        for idx in range(image.shape[0]):
            image[idx] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image[idx])
        mask = torch.from_numpy(mask)

        return image, mask

    root = osp.expanduser('~/data/datasets/SpineData')
    dataset = SpineDataset(root=root, transform=my_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data, mask in dataloader:
        print(data.shape, mask.shape)
# vim:set ts=4 sw=4 et:
