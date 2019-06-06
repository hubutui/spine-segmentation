#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path as osp
import horovod.torch as hvd
import torch
import torch.utils.data.distributed
from apex.parallel import convert_syncbn_model
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.SpineDataset import SpineDataset
import gluoncvth as gcv
from model.UNet import UNet
from torch import optim
import SimpleITK as sitk
import numpy as np
import random
from torchvision import transforms


def my_transform(image, mask):
    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask).astype(np.int64)
    # 随机裁剪 512x512 的尺寸
    x = random.randint(0, image.shape[1] - 512)
    y = random.randint(0, image.shape[2] - 512)
    image = image[:, y:y+512, x:x+512]
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

def getArgs():
    parser = argparse.ArgumentParser(description="Train network")
    parser.add_argument('--num_epochs', type=int, default=101,
                        help='Number of epochs to train for, default: 101')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate used for train, default: 0.003')
    parser.add_argument('--data', type=str, default=osp.expanduser('~/data/datasets/SpineData'),
                        help='root path of training data, default: ~/data/datasets/SpineData')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers, default: 16')
    parser.add_argument('--save_model_path', type=str, default='tmpdir',
                        help='path to save model, default: tmpdir')
    parser.add_argument('--power', type=float, default=0.9,
                        help='power to poly learning rate scheduler, default: 0.9')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how often to print log, defautl: 10')
    parser.add_argument('--network', type=str, default='FCN',
                        choices=['FCN', 'DeepLab', 'PSPNet', 'UNet'],
                        help='network type, default: FCN')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone network, default: resnet50')
    parser.add_argument('--voc', action='store_true',
                        help='use voc pretrained weight for PSPNet, FCN, or DeepLab')
    parser.add_argument('--ade', action='store_true',
                        help='use ade pretrained weight for PSPNet, FCN, or DeepLab')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes, default: 2')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='batch size, default: 3')

    return parser.parse_args()


def main():
    torch.backends.cudnn.benchmark = True
    args = getArgs()
    torch.manual_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # horovod 初始化
    hvd.init()
    torch.manual_seed(args.seed)
    # 打印一下训练使用的配置
    if hvd.rank() == 0:
        print("Training with configure: ")
        for arg in vars(args):
            print("{}:\t{}".format(arg, getattr(args, arg)))
        if not osp.exists(args.save_model_path):
            os.makedirs(args.save_model_path)
        # 保存训练配置
        with open(osp.join(args.save_model_path, 'train-config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
    # 设置随机种子，保证每个 GPU 上的权重初始化都一样
    if args.cuda:
        # Pin GPU to local rank
        torch.cuda.set_device(hvd.local_rank())
        # 这一句似乎没有用的吧。不过按照 horovod 的回复来说，还是加上好了。
        torch.cuda.manual_seed(args.seed)
    # data
    dataset_train = SpineDataset(root=args.data, transform=my_transform)
    # 分布式训练需要使用这个 sampler
    sampler_train = DistributedSampler(dataset_train, num_replicas=hvd.size(), rank=hvd.rank())
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=1,
                                  sampler=sampler_train,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    # model
    if args.network == 'DeepLab':
        if args.voc:
            model = gcv.models.get_deeplab_resnet101_voc(pretrained=True)
        elif args.ade:
            model = gcv.models.get_deeplab_resnet101_ade(pretrained=True)
        else:
            model = gcv.models.DeepLabV3(nclass=args.num_classes, backbone=args.backbone)
        model.auxlayer.conv5[-1] = nn.Conv2d(256, args.num_classes, kernel_size=1)
        model.head.block[-1] = nn.Conv2d(256, args.num_classes, kernel_size=1)
    elif args.network == 'FCN':
        if args.voc:
            model = gcv.models.get_fcn_resnet101_voc(pretrained=True)
        elif args.ade:
            model = gcv.models.get_fcn_resnet101_ade(pretrained=True)
        else:
            model = gcv.models.FCN(nclass=args.num_classes, backbone=args.backbone)
        model.auxlayer.conv5[-1] = nn.Conv2d(256, args.num_classes, kernel_size=1)
        model.head.conv5[-1] = nn.Conv2d(512, args.num_classes, kernel_size=1)
    elif args.network == 'PSPNet':
        if args.voc:
            model = gcv.models.get_psp_resnet101_voc(pretrained=True)
        elif args.ade:
            model = gcv.models.get_psp_resnet101_ade(pretrained=True)
        else:
            model = gcv.models.PSP(nclass=args.num_classes, backbone=args.backbone)
        model.auxlayer.conv5[-1] = nn.Conv2d(256, 2, kernel_size=1)
        model.head.conv5[-1] = nn.Conv2d(512, args.num_classes, kernel_size=1)
    elif args.network == 'UNet':
        model = UNet(n_class=args.num_classes,
                     backbone=args.backbone,
                     pretrained=True)
    model = convert_syncbn_model(model)
    model = model.to(device)

    # optimizer 要用 hvd 的版本包一下
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate * hvd.size())
    # 不同层使用不同的学习率
    if args.network == 'UNet':
        optimizer = torch.optim.SGD([{'params': model.down_blocks.parameters(), 'lr': args.learning_rate*0.5},
                                     {'params': model.bridge.parameters()},
                                     {'params': model.head.parameters()},],
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=0.0001)
    elif args.network in ['FCN', 'PSPNet', 'DeepLab']:
        optimizer = optim.SGD([{'params': model.pretrained.parameters(), 'lr': args.learning_rate*0.5},
                               {'params': model.auxlayer.parameters()},
                               {'params': model.head.parameters()}],
                              lr=args.learning_rate,
                              momentum=0.9,
                              weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=0.9,
                              weight_decay=0.0001)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters())
    # 将模型和优化器的参数广播到各个 GPU 上
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # lr scheduler
    def poly_lr_scheduler(epoch, num_epochs=args.num_epochs, power=args.power):
        return (1 - epoch / num_epochs) ** power

    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)

    def train(epoch):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        sampler_train.set_epoch(epoch)
        lr_scheduler.step()
        loss_fn = nn.CrossEntropyLoss()
        for batch_idx, (data, target) in enumerate(dataloader_train):
            data = data.to(device).squeeze()
            target = target.to(device).squeeze()
            for batch_data, batch_target in zip(torch.split(data, args.batch_size), torch.split(target, args.batch_size)):
                optimizer.zero_grad()
                output = model(batch_data)
                if args.network in ['FCN', 'PSPNet', 'DeepLab']:
                    loss = loss_fn(output[0], batch_target) \
                           + 0.2*loss_fn(output[1], batch_target)
                elif args.network == 'UNet':
                    loss = loss_fn(output, batch_target)
                loss.backward()
                optimizer.step()
            if hvd.rank() == 0 and batch_idx % args.log_interval == 0:
                print("Train loss: ", loss.item())

    for epoch in range(args.num_epochs):
        train(epoch)
        if hvd.rank() == 0:
            print("Saving model to {}".format(osp.join(args.save_model_path, "checkpoint-{:0>3d}.pth".format(epoch))))
            torch.save({'state_dict': model.state_dict()},
                   osp.join(args.save_model_path, "checkpoint-{:0>3d}.pth".format(epoch)))


if __name__ == '__main__':
    main()
# vim:set ts=4 sw=4 et:
