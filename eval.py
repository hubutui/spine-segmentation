#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import tqdm
from torch.utils.data import DataLoader
from dataset.SpineDataset import SpineDataset
import SimpleITK as sitk
from sklearn import metrics
import gluoncvth as gcv
from model.UNet import UNet
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


def val_transform(image, mask):
    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask).astype(np.int64)
    image = np.expand_dims(image, axis=1)
    image = np.repeat(image, 3, 1)
    image = torch.from_numpy(image)
    # 缩放到 0-1 范围，然后归一化
    image /= image.max()
    for idx in range(image.shape[0]):
        image[idx] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image[idx])
    mask = torch.from_numpy(mask)

    return image, mask


def test_transform(image):
    image = sitk.GetArrayFromImage(image)
    image = np.expand_dims(image, axis=1)
    image = np.repeat(image, 3, 1)
    image = torch.from_numpy(image)
    # 缩放到 0-1 范围，然后归一化
    image /= image.max()
    for idx in range(image.shape[0]):
        image[idx] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image[idx])

    return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, required=True,
                        help='The path to the model file, default: None')
    parser.add_argument('--root', type=str, default=osp.expanduser('~/data/datasets/SpineData'),
                        help='Path of data, default: ~/data/datasets/SpineData')
    parser.add_argument('--cuda', type=str, default='1',
                        help='GPU ids used for testing, default: 1')
    parser.add_argument('--network', type=str, default='FCN',
                        choices=['DeepLab', 'FCN', 'PSPNet', 'UNet'],
                        help='network type, default: FCN')
    parser.add_argument('--backbone', type=str, default='vgg11_bn',
                        help='backbone for UNet, default: vgg11_bn')
    parser.add_argument('--result_dir', type=str, default='submission',
                        help='dir to save result for submission, default: submission')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes, default: 2')

    return parser.parse_args()


def main():
    # 参数
    args = get_args()
    if not osp.exists(args.result_dir):
        os.makedirs(args.result_dir)
    print("Evaluating configuration:")
    for arg in vars(args):
        print("{}:\t{}".format(arg, getattr(args, arg)))
    with open('eval-config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    # 数据
    dataset = SpineDataset(root=args.root, split='val', transform=val_transform)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # 模型
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    if args.network == 'DeepLab':
        model = gcv.models.DeepLabV3(nclass=args.num_classes,
                                     backbone=args.backbone)
    elif args.network == 'FCN':
        model = gcv.models.FCN(nclass=args.num_classes,
                               backbone=args.backbone)
    elif args.network == 'PSPNet':
        model = gcv.models.PSP(nclass=args.num_classes,
                               backbone=args.backbone)
    elif args.network == 'UNet':
        model = UNet(n_class=args.num_classes,
                     backbone=args.backbone)
    print('load model from {} ...'.format(args.model))
    model.load_state_dict(torch.load(args.model, map_location='cpu')['state_dict'])
    model = model.to(device)
    print('Done!')

    # 测试
    def eval():
        with torch.no_grad():
            model.eval()
            result = []
            tq = tqdm.tqdm(total=len(dataloader))
            tq.set_description('test')
            for i, (data, mask, mask_file) in enumerate(dataloader):
                tq.update(1)
                gt_img = sitk.ReadImage(mask_file[0])
                data = data.to(device)
                predict = np.zeros((data.size()[1], data.size()[3], data.size()[4]), dtype=np.uint16)
                for idx in range(data.size()[1]):
                    if args.network in ['DeepLab', 'FCN', 'PSPNet']:
                        final_out = model(data[:, idx])[0]
                    elif args.network == 'UNet':
                        final_out = model(data[:, idx])
                    predict[idx] = final_out.argmax(dim=1).cpu().squeeze().numpy().astype(np.uint16)
                pred_img = sitk.GetImageFromArray(predict)
                pred_img.CopyInformation(gt_img)
                sitk.WriteImage(pred_img, osp.join(args.result_dir, mask_file[0].split('/')[-1]))
                ppv, _, dice, _ = metrics.precision_recall_fscore_support(mask.numpy().flatten(), predict.flatten(), average='binary')
                result.append([ppv, dice])
            result = np.array(result)
            result_mean = result.mean(axis=0)
            result_std = result.std(axis=0)
            print(result_mean, result_std)
            np.savetxt(osp.join(args.result_dir, 'result.txt'),
                       result_mean,
                       fmt='%.3f',
                       header='PPV, Dice')

            tq.close()

    eval()


if __name__ == '__main__':
    main()
# vim:set ts=4 sw=4 et: