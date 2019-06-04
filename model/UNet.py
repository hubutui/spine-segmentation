#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)

        return x


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        """
        FCNHead, copy from torchvision 0.3.0
        :param in_channels: input channels
        :param out_channels: output channels
        :param dropout: set to None if don't wanna use dropout
        """
        if dropout:
            layers = [nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=3,
                                padding=1,
                                bias=False),
                      nn.BatchNorm2d(in_channels),
                      nn.ReLU(),
                      nn.Dropout2d(p=dropout),
                      nn.Conv2d(in_channels, out_channels, 1)]
        else:
            layers = [nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=3,
                                padding=1,
                                bias=False),
                      nn.BatchNorm2d(in_channels),
                      nn.ReLU(),
                      nn.Conv2d(in_channels, out_channels, 1)]
        super(FCNHead, self).__init__(*layers)


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet connecting encoder (downsample path) and decoder (upsamle path)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cat_channels=None, upsample_method='bilinear'):
        """
        :param in_channels: input feature map channel
        :param out_channels: output feature map channel
        :param cat_channels: feature map channel after cat [up_x, down_x],
            equals to in_channels + out_channels if not set
        :param upsample_method: bilinear or deconv
        """
        super(UpBlock, self).__init__()
        if upsample_method.lower() not in ['deconv', 'bilinear']:
            raise RuntimeError("Unexpected upsample method: {}".format(upsample_method))
        self.upsample_method = upsample_method
        if self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        if self.upsample_method == 'bilinear':
            self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        if cat_channels == None:
            cat_channels = in_channels + out_channels
        self.conv1 = ConvBlock(cat_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: output from the previous up block
        :param down_x: output from the down block
        :return: upsampled feature map
        """
        if self.upsample_method == 'bilinear':
            x = F.interpolate(up_x, size=down_x.shape[2:], mode='bilinear')
            x = self.conv1x1(x)
        if self.upsample_method == 'deconv':
            x = self.upsample(up_x)
        x = torch.cat([x, down_x], dim=1)
        assert self.conv1.conv.in_channels == x.shape[1], "Expect cat_channels to be {}, but get {}".format(x.shape[1],
                                                                                                            self.conv1.conv.in_channels)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class ResUNet(nn.Module):
    def __init__(self, n_class,
                 backbone='resnet50',
                 pretrained=True,
                 upsample_method='bilinear',
                 dropout=0.1):
        """
        UNet use ResNet as backbone
        :param n_class: number of classes
        :param backbone: backbone network
        :param pretrained: use pretrained weight on ImageNet
        :param upsample_method: bilinear or deconv to upsample
        :param dropout: add dropout before the last conv layer, default: 0.1, set to None you dont't want it
        """
        super(ResUNet, self).__init__()
        if backbone not in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            raise RuntimeError("Unexpected backbone: {}".format(backbone))
        model = getattr(models, backbone)(pretrained=pretrained)
        # downsample blocks
        down_blocks = dict()
        down_blocks['down0'] = nn.Sequential(*list(model.children())[:3])
        down_blocks['down1'] = nn.Sequential(model.maxpool, model.layer1)
        down_blocks['down2'] = model.layer2
        down_blocks['down3'] = model.layer3
        down_blocks['down4'] = model.layer4
        self.down_blocks = nn.ModuleDict(down_blocks)
        # 最底下多加几个卷积作为过渡
        if backbone in ['resnet18', 'resnet34']:
            self.bridge = Bridge(512, 512)
        else:
            self.bridge = Bridge(2048, 2048)
        # 上采样层，一共4次上采样
        up_blocks = dict()
        if backbone in ['resnet18', 'resnet34']:
            up_blocks['up4'] = UpBlock(512, 256, 768, upsample_method=upsample_method)
            up_blocks['up3'] = UpBlock(256, 128, 384, upsample_method=upsample_method)
            up_blocks['up2'] = UpBlock(128, 64, 192, upsample_method=upsample_method)
            up_blocks['up1'] = UpBlock(64, 64, 128, upsample_method=upsample_method)
            up_blocks['up0'] = UpBlock(64, 64, 67, upsample_method=upsample_method)
        else:
            up_blocks['up4'] = UpBlock(2048, 1024, 3072, upsample_method=upsample_method)
            up_blocks['up3'] = UpBlock(1024, 512, 1536, upsample_method=upsample_method)
            up_blocks['up2'] = UpBlock(512, 256, 768, upsample_method=upsample_method)
            up_blocks['up1'] = UpBlock(256, 128, 320, upsample_method=upsample_method)
            up_blocks['up0'] = UpBlock(128, 64, 131, upsample_method=upsample_method)
        self.up_blocks = nn.ModuleDict(up_blocks)

        # 最后修改通道数为类别数
        self.head = FCNHead(64, n_class, dropout)

    def forward(self, x):
        # 保存每一次下采样前的 tensor，方便后面的上采样使用
        pre_pools = dict()
        pre_pools['layer0'] = x
        for idx in range(len(self.down_blocks)):
            pre_pools["layer{}".format(idx)] = x
            x = self.down_blocks["down{}".format(idx)](x)
        x = self.bridge(x)
        for idx in reversed(range(len(self.up_blocks))):
            x = self.up_blocks["up{}".format(idx)](x, pre_pools["layer{}".format(idx)])
        del pre_pools
        x = self.head(x)

        return x


class VGGUNet(nn.Module):
    def __init__(self, n_class,
                 backbone='resnet50',
                 pretrained=True,
                 upsample_method='deconv',
                 dropout=0.1):
        """
        UNet using VGG as backbone
        :param n_class: number of classes
        :param backbone: backbone network
        :param pretrained: use pretrained weight on ImageNet
        :param upsample_method: bilinear or deconv
        :param dropout: 0.1 be default, set to None to disable
        """
        super(VGGUNet, self).__init__()
        if backbone not in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
            raise RuntimeError("Unexpected backbone: {}".format(backbone))
        model = getattr(models, backbone)(pretrained=pretrained)

        down_blocks = dict()
        seq = list()
        idx = 0
        for block in model.features:
            seq.append(block)
            if isinstance(block, nn.MaxPool2d):
                down_blocks['down{}'.format(idx)] = nn.Sequential(*seq)
                seq = list()
                idx += 1
        self.down_blocks = nn.ModuleDict(down_blocks)
        self.bridge = Bridge(512, 512)

        up_blocks = dict()
        up_blocks['up4'] = UpBlock(512, 512, 1024, upsample_method=upsample_method)
        up_blocks['up3'] = UpBlock(512, 256, 768, upsample_method=upsample_method)
        up_blocks['up2'] = UpBlock(256, 128, 384, upsample_method=upsample_method)
        up_blocks['up1'] = UpBlock(128, 64, 192, upsample_method=upsample_method)
        up_blocks['up0'] = UpBlock(64, 64, 67, upsample_method=upsample_method)
        self.up_blocks = nn.ModuleDict(up_blocks)

        self.head = FCNHead(64, n_class, dropout)

    def forward(self, x):
        # 保存每一次下采样前的 tensor，方便后面的上采样使用
        pre_pools = dict()
        for idx in range(len(self.down_blocks)):
            pre_pools["layer{}".format(idx)] = x
            x = self.down_blocks["down{}".format(idx)](x)
        x = self.bridge(x)
        for idx in reversed(range(len(self.up_blocks))):
            x = self.up_blocks["up{}".format(idx)](x, pre_pools["layer{}".format(idx)])
        del pre_pools
        x = self.head(x)

        return x


class AlexUNet(nn.Module):
    def __init__(self, n_class, pretrained=True, upsample_method='bilinear', dropout=0.1):
        """
        UNet using AlexNet as backbone
        :param n_class: number of classes
        :param pretrained: use pretrained weight on ImageNet
        :param upsample_method: bilinear or deconv
        :param dropout: 0.1 by default, set to None to disable
        """
        super(AlexUNet, self).__init__()
        if upsample_method != 'bilinear':
            print("Only bilinear upsample is support for alexnet as backbone, "
                  "reset upsamle method to bilinear.")
        model = models.alexnet(pretrained=pretrained)
        down_blocks = dict()
        down_blocks['down0'] = model.features[:2]
        down_blocks['down1'] = model.features[2]
        down_blocks['down2'] = model.features[3:6]
        down_blocks['down3'] = model.features[6:]
        self.down_blocks = nn.ModuleDict(down_blocks)
        self.bridge = Bridge(256, 256)
        up_blocks = dict()
        up_blocks['up3'] = UpBlock(256, 256, 448, upsample_method='bilinear')
        up_blocks['up2'] = UpBlock(256, 192, 320, upsample_method='bilinear')
        up_blocks['up1'] = UpBlock(192, 128, 256, upsample_method='bilinear')
        up_blocks['up0'] = UpBlock(128, 64, 131, upsample_method='bilinear')
        self.up_blocks = nn.ModuleDict(up_blocks)
        self.head = FCNHead(64, n_class, dropout)

    def forward(self, x):
        pre_pools = dict()
        for idx in range(len(self.down_blocks)):
            pre_pools["layer{}".format(idx)] = x
            x = self.down_blocks["down{}".format(idx)](x)
        x = self.bridge(x)
        for idx in reversed(range(len(self.up_blocks))):
            x = self.up_blocks["up{}".format(idx)](x, pre_pools["layer{}".format(idx)])
        del pre_pools
        x = self.head(x)

        return x


class DenseUNet(nn.Module):
    def __init__(self, n_class,
                 backbone='densenet121',
                 pretrained=True,
                 upsample_method='bilinear',
                 dropout=0.1):
        """
        UNet using DenseNet as backbone
        :param n_class: number of classes
        :param backbone: backbone network
        :param pretrained: pretrained on ImageNet
        :param upsample_method: bilinear or deconv
        :param dropout: 0.1, set to None to disable
        """
        super(DenseUNet, self).__init__()
        if backbone not in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            raise RuntimeError("Unexpected backbone: {}".format(backbone))
        model = getattr(models, backbone)(pretrained=pretrained)
        down_blocks = dict()
        down_blocks['down0'] = model.features[:3]
        down_blocks['down1'] = nn.Sequential(model.features[3], model.features.denseblock1)
        down_blocks['down2'] = nn.Sequential(model.features.transition1, model.features.denseblock2)
        down_blocks['down3'] = nn.Sequential(model.features.transition2, model.features.denseblock3)
        down_blocks['down4'] = nn.Sequential(model.features.transition3, model.features.denseblock4,
                                             model.features.norm5)
        self.down_blocks = nn.ModuleDict(down_blocks)
        if backbone == 'densenet121':
            self.bridge = Bridge(1024, 1024)
        elif backbone == 'densenet161':
            self.bridge = Bridge(2208, 2208)
        elif backbone == 'densenet169':
            self.bridge = Bridge(1664, 1664)
        elif backbone == 'densenet201':
            self.bridge = Bridge(1920, 1920)
        up_blocks = dict()
        if backbone == 'densenet121':
            up_blocks['up4'] = UpBlock(1024, 512, 2048, upsample_method=upsample_method)
            up_blocks['up3'] = UpBlock(512, 256, 1024, upsample_method=upsample_method)
            up_blocks['up2'] = UpBlock(256, 128, 512, upsample_method=upsample_method)
            up_blocks['up1'] = UpBlock(128, 64, 192, upsample_method=upsample_method)
            up_blocks['up0'] = UpBlock(64, 64, 67, upsample_method=upsample_method)
        elif backbone == 'densenet161':
            # 上采样部分也可以用这几行被注释掉的替代
            # up_blocks['up4'] = UpBlock(2208, 1024, 4320, upsample_method=upsample_method)
            # up_blocks['up3'] = UpBlock(1024, 512, 1792, upsample_method=upsample_method)
            # up_blocks['up2'] = UpBlock(512, 256, 896, upsample_method=upsample_method)
            # up_blocks['up1'] = UpBlock(256, 128, 352, upsample_method=upsample_method)
            # up_blocks['up0'] = UpBlock(128, 64, 131, upsample_method=upsample_method)
            up_blocks['up4'] = UpBlock(2208, 512, 4320, upsample_method=upsample_method)
            up_blocks['up3'] = UpBlock(512, 256, 1280, upsample_method=upsample_method)
            up_blocks['up2'] = UpBlock(256, 128, 640, upsample_method=upsample_method)
            up_blocks['up1'] = UpBlock(128, 64, 224, upsample_method=upsample_method)
            up_blocks['up0'] = UpBlock(64, 64, 67, upsample_method=upsample_method)
        elif backbone == 'densenet169':
            up_blocks['up4'] = UpBlock(1664, 512, 2944, upsample_method=upsample_method)
            up_blocks['up3'] = UpBlock(512, 256, 1024, upsample_method=upsample_method)
            up_blocks['up2'] = UpBlock(256, 128, 512, upsample_method=upsample_method)
            up_blocks['up1'] = UpBlock(128, 64, 192, upsample_method=upsample_method)
            up_blocks['up0'] = UpBlock(64, 64, 67, upsample_method=upsample_method)
        elif backbone == 'densenet201':
            up_blocks['up4'] = UpBlock(1920, 512, 3712, upsample_method=upsample_method)
            up_blocks['up3'] = UpBlock(512, 256, 1024, upsample_method=upsample_method)
            up_blocks['up2'] = UpBlock(256, 128, 512, upsample_method=upsample_method)
            up_blocks['up1'] = UpBlock(128, 64, 192, upsample_method=upsample_method)
            up_blocks['up0'] = UpBlock(64, 64, 67, upsample_method=upsample_method)
        self.up_blocks = nn.ModuleDict(up_blocks)
        self.head = FCNHead(64, n_class, dropout)

    def forward(self, x):
        pre_pools = dict()
        for idx in range(len(self.down_blocks)):
            pre_pools["layer{}".format(idx)] = x
            x = self.down_blocks["down{}".format(idx)](x)
        x = self.bridge(x)
        for idx in reversed(range(len(self.up_blocks))):
            x = self.up_blocks["up{}".format(idx)](x, pre_pools["layer{}".format(idx)])
        del pre_pools
        x = self.head(x)

        return x


def UNet(n_class, backbone='resnet50', pretrained=True, upsample_method='bilinear', dropout=0.1):
    if backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model = ResUNet(n_class=n_class,
                        backbone=backbone,
                        pretrained=pretrained,
                        upsample_method=upsample_method,
                        dropout=dropout)
    elif backbone in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
        model = DenseUNet(n_class=n_class,
                          backbone=backbone,
                          pretrained=pretrained,
                          upsample_method=upsample_method)
    elif backbone in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
        model = VGGUNet(n_class=n_class,
                        backbone=backbone,
                        pretrained=pretrained,
                        upsample_method=upsample_method)
    elif backbone == 'alexnet':
        model = AlexUNet(n_class=n_class,
                         pretrained=pretrained,
                         upsample_method=upsample_method)

    return model


if __name__ == '__main__':
    backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                 'alexnet',
                 'densenet121', 'densenet161', 'densenet169', 'densenet201']
    for upsample_method in ['deconv', 'bilinear']:
        for backbone in backbones:
            with torch.no_grad():
                print("{}-{}".format(upsample_method, backbone))
                data = torch.randn((1, 3, 256, 256)).cuda()
                model = UNet(n_class=21, upsample_method=upsample_method, backbone=backbone).cuda()
                print(data.shape)
                y = model(data)
                print(y.shape)
# vim:set ts=4 sw=4 et:
