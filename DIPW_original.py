from lhhload import myDataset
from torch.utils.data import DataLoader

# encoding: utf-8
import sys
import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import cv2
import os
# import utils.transformed as transforms
from torchvision import transforms
# from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
from torchvision.datasets import ImageFolder
import pdb
import math
import random
import numpy as np
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR
import cv2
import PerceptualSimilarity.models
from tqdm import tqdm
from detection import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


def save_checkpoint(state, save_path):
    filename = os.path.join(save_path, 'best_loss_model.pth')
    torch.save(state, filename)


Hnet = UnetGenerator(input_nc=3, output_nc=3, num_downs=5, norm_layer=nn.BatchNorm2d,
                     output_function=nn.Tanh)
Rnet = RevealNet(input_nc=3, output_nc=3, nhf=64,
                 norm_layer=nn.BatchNorm2d, output_function=nn.Sigmoid)
# Hnet.apply(weights_init)
# Rnet.apply(weights_init)

Hnet = torch.nn.DataParallel(Hnet).cuda()
Rnet = torch.nn.DataParallel(Rnet).cuda()

checkpoint = torch.load("./training/main_udh/checkPoints/" + "checkpoint.pth.tar")
# R_path = torch.load(r'D:\code\DIPW-main\DIPW\checkpoint\DIPW_99_epoch.pth')
Hnet.load_state_dict(checkpoint['H_state_dict'])
Rnet.load_state_dict(checkpoint['R_state_dict'], False)
train_cover_dir = r'D:\code\DIPW-main\DIPW\dataset\coverV2'
train_watermark_dir = r'D:\code\DIPW-main\DIPW\dataset\watermarkV2'
tamp_dir = r'D:\code\DIPW-main\DIPW\dataset\containerV2'
rev_dir = r'D:\code\DIPW-main\DIPW\dataset\rev_watermark'

width = 128
high = 128
epoch = 1
batch_size = 1
resize = True
train_dataset = myDataset(train_cover_dir, train_watermark_dir, width, high, resize=resize)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

optimizer = optim.SGD(Hnet.parameters(), lr=0.0001, momentum=0.9)
L1_loss = nn.L1Loss().cuda()
L2_loss = nn.MSELoss().cuda()
a = 0.75
loss = 10000

psnr_p = np.zeros((batch_size, 3))
psnr_w = np.zeros((batch_size, 3))
ssim_p = np.zeros(batch_size)
ssim_w = np.zeros(batch_size)
predictor = Predictor(
    model, exp, COCO_CLASSES, trt_file, decoder,
    args.device, args.fp16, args.legacy,
)
Hnet.eval()
Rnet.eval()

for i in range(epoch):
    for patch, cover, watermark, box, cover_name, water_name in train_loader:
        patch = patch.cuda()
        cover = cover.cuda()
        watermark = watermark.cuda()
        itm_secret_img = Hnet(watermark)
        container_patch = itm_secret_img + patch

        container_patch_numpy = container_patch.clone().cpu().detach().numpy()
        container_patch_numpy = container_patch_numpy.transpose(0, 2, 3, 1)

        cover_numpy = cover.clone().cpu().detach().numpy()
        cover_numpy = cover_numpy.transpose(0, 2, 3, 1)

        cover_name_numpy = cover_name.clone().cpu().detach().numpy()

        water_name_numpy = water_name.clone().cpu().detach().numpy()

        N, _, _, _ = container_patch.shape
        for B1 in range(batch_size):
            box_p = box[B1]
            cover_name_str = '%012d' % cover_name_numpy[B1]
            water_name_str = '%012d' % water_name_numpy[B1]
            cover_numpy[B1][box_p[0]:box_p[1], box_p[2]:box_p[3]] = container_patch_numpy[B1]
            container_patch_show = cv2.cvtColor(cover_numpy[B1], cv2.COLOR_RGB2BGR)
            # cv2.imshow('1', container_patch_show)
            # cv2.waitKey()
            # print('======save container======')
            cv2.imwrite(tamp_dir + "//" + cover_name_str + '.png', container_patch_show * 255)
            # print('======save done======')

    test_dataset = myDataset(tamp_dir, train_watermark_dir, width, high, resize=resize)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for patch_r, cover_r, watermark_r, box_r, cover_r_name, water_r_name in test_loader:
        rev_secret_img = Rnet(patch_r)
        rev_secret_img_numpy = rev_secret_img.clone().cpu().detach().numpy()
        rev_secret_img_numpy = rev_secret_img_numpy.transpose(0, 2, 3, 1)
        water_r_name_numpy = water_r_name.clone().cpu().detach().numpy()
        for B2 in range(batch_size):
            #     print('======save rev======')
            water_r_name_str = '%012d' % water_r_name_numpy[B2]
            rev_secret_img_show = cv2.cvtColor(rev_secret_img_numpy[B2], cv2.COLOR_RGB2BGR)
            # cv2.imshow('1', rev_secret_img_show)
            # cv2.waitKey()
            cv2.imwrite(rev_dir + "//" + water_r_name_str + '.png', rev_secret_img_show * 255)
