import collections
from itertools import repeat
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class EddyNet(nn.Module):
    """
    PyTorch implementation of EddyNet from Lguensat et al. (2018)
    Original implementation in TensorFlow: https://github.com/redouanelg/EddyNet
    """
    def __init__(self, num_classes, num_filters, kernel_size):
        super(EddyNet, self).__init__()
        # encoder
        self.encoder1 = EddyNet._block(1, num_filters, kernel_size, "enc1", dropout=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = EddyNet._block(
            num_filters, num_filters, kernel_size, "enc2", dropout=0.3
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = EddyNet._block(
            num_filters, num_filters, kernel_size, "enc3", dropout=0.4
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = EddyNet._block(
            num_filters, num_filters, kernel_size, "enc4", dropout=0.5
        )

        # decoder
        self.decoder3 = EddyNet.decoder_block(
            num_filters * 2, num_filters, kernel_size, "dec3", dropout=0.4
        )
        self.decoder2 = EddyNet.decoder_block(
            num_filters * 2, num_filters, kernel_size, "dec2", dropout=0.3
        )
        self.decoder1 = EddyNet.decoder_block(
            num_filters * 2, num_filters, kernel_size, "dec1", dropout=0.2
        )

        # final layer
        self.final_conv = nn.Conv2d(
            num_filters, num_classes, kernel_size=1, padding=0, bias=False
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, name, num, dropout=0):
        layers = {
            f"{name}_conv{num}": Conv2dSame(in_channels, out_channels, kernel_size),
            f"{name}_bn{num}": nn.BatchNorm2d(out_channels),
            f"{name}_relu{num}": nn.ReLU(inplace=True),
        }
        if dropout > 0:
            layers[f"{name}_dropout"] = nn.Dropout(p=dropout)

        return nn.Sequential(OrderedDict(layers))

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, name, dropout=0):
        conv1 = EddyNet.conv_block(in_channels, out_channels, kernel_size, name, 1)
        conv2 = EddyNet.conv_block(
            out_channels, out_channels, kernel_size, name, 2, dropout=dropout
        )
        return nn.Sequential(conv1, conv2)

    @staticmethod
    def decoder_block(in_channels, out_channels, kernel_size, name, dropout=0):
        return EddyNet._block(in_channels, out_channels, kernel_size, name, dropout)

    def forward(self, x):
        # encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)

        # bottleneck?
        enc4 = self.encoder4(pool3)

        # decoder
        dec3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)(enc4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # final layer
        final = self.final_conv(dec1)

        # softmax
        final = nn.Softmax(dim=1)(final)

        return final


class Conv2dSame(nn.Module):
    """Manual convolution with same padding
    https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/9
    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0,
    so we need to implement the internal torch logic manually.

    Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs
    ):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)
