import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import Attention
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d
import sys


class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class USPADEResGenerator(BaseNetwork):  
    def __init__(self,opt):
        super(USPADEResGenerator, self).__init__()
        self.opt = opt

        in_ch = 6
        self.Convolution1 = ConvolutionBlock(in_ch, 64)
        self.maxpooling1 = nn.MaxPool2d(2)  
        self.Convolution2 = ConvolutionBlock(64, 128)
        self.maxpooling2 = nn.MaxPool2d(2)
        self.Convolution3 = ConvolutionBlock(128, 256)
        self.maxpooling3 = nn.MaxPool2d(2)
        self.Convolution4 = ConvolutionBlock(256, 512)
        self.maxpooling4 = nn.MaxPool2d(2)
        self.Convolution5 = ConvolutionBlock(512, 1024)

        self.maxpooling6 = nn.ConvTranspose2d(1024, 256, 2, stride=2) 
        self.Convolution6 = ConvolutionBlock(1024, 512)
        self.maxpooling7 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.Convolution7 = ConvolutionBlock(512, 256)
        self.maxpooling8 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.Convolution8 = ConvolutionBlock(256, 128)
        self.maxpooling9 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        self.Convolution9 = ConvolutionBlock(128, 64)
        self.Convolution10 = nn.Conv2d(64, 3, 1)

        self.spade0 = SPADEResnetBlock(1024, 256, opt)
        self.spade1 = SPADEResnetBlock(512, 128, opt)
        self.spade2 = SPADEResnetBlock(256, 64, opt)
        self.spade3 = SPADEResnetBlock(128, 32, opt)

        self.up = nn.Upsample(scale_factor=2)


        nf = 64
        self.reveal_func = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, 3, 3, 1, 1),
            nn.Tanh() 
        )


    def forward(self, x, warp_out=None, secret_input=None):
        seg = warp_out
        x = torch.cat((warp_out,secret_input),1)
        conv1 = self.Convolution1(x)
        pool1 = self.maxpooling1(conv1)
        conv2 = self.Convolution2(pool1)
        pool2 = self.maxpooling2(conv2)
        conv3 = self.Convolution3(pool2)
        pool3 = self.maxpooling3(conv3)
        conv4 = self.Convolution4(pool3)
        pool4 = self.maxpooling4(conv4)
        conv5 = self.Convolution5(pool4)
        up_6 = self.maxpooling6(conv5)
        spade_6 = self.spade0(self.up(conv5),seg)
        merge6 = torch.cat([up_6, conv4,spade_6], dim=1)
        conv6 = self.Convolution6(merge6)
        up_7 = self.maxpooling7(conv6)
        spade_7 = self.spade1(self.up(conv6), seg)
        merge7 = torch.cat([up_7, conv3,spade_7], dim=1)
        conv7 = self.Convolution7(merge7)
        up_8 = self.maxpooling8(conv7)
        spade_8 = self.spade2(self.up(conv7), seg)
        merge8 = torch.cat([up_8, conv2,spade_8], dim=1)
        conv8 = self.Convolution8(merge8)
        up_9 = self.maxpooling9(conv8)
        spade_9 = self.spade3(self.up(conv8), seg)
        merge9 = torch.cat([up_9, conv1,spade_9], dim=1)
        conv9 = self.Convolution9(merge9)
        out = self.Convolution10(conv9)

        reveal_out = self.reveal_func(out)
        return out,reveal_out

