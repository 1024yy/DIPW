import copy

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
import datetime

now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')


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
Hnet.load_state_dict(checkpoint['H_state_dict'])
Rnet.load_state_dict(checkpoint['R_state_dict'], False)
train_cover_dir = r'D:\code\DIPW-main\DIPW\dataset\dataset_before_process\cover'
train_watermark_dir = r'D:\code\DIPW-main\DIPW\dataset\dataset_before_process\watermark'
tamp_c_dir = r'D:\code\DIPW-main\DIPW\dataset\trainingdata\temp_container_mini'
tamp_w_dir = r'D:\code\DIPW-main\DIPW\dataset\trainingdata\temp_watermark_mini'
rev_dir = r'D:\code\DIPW-main\DIPW\dataset\temp_rev'

width = 128
high = 128
epoch = 100
batch_size = 20
resize = True
train_dataset = myDataset(train_cover_dir, train_watermark_dir, width, high, resize=resize)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# val_dataset = myDataset(train_cover_dir, train_watermark_dir, width, high, resize=False)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

save_path = 'D:\code\DIPW-main\DIPW\ckpt\checkpoint_' + now_time

if not os.path.exists(save_path):
    os.makedirs(save_path)
optimizer = optim.Adam(Rnet.parameters(), lr=0.001)
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


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# print("dataset generate")
# for patch, cover, watermark, box, cover_name, water_name in train_loader:
#     Hnet.eval()
#     patch = patch.cuda()
#     cover = cover.cuda()
#     watermark = watermark.cuda()
#     itm_secret_img = Hnet(watermark)
#     container_patch = itm_secret_img + patch
#
#     container_patch_numpy = container_patch.clone().cpu().detach().numpy()
#     container_patch_numpy = container_patch_numpy.transpose(0, 2, 3, 1)
#
#     cover_numpy = cover.clone().cpu().detach().numpy()
#     cover_numpy = cover_numpy.transpose(0, 2, 3, 1)
#
#     watermark_numpy = watermark.clone().cpu().detach().numpy()
#     watermark_numpy = watermark_numpy.transpose(0, 2, 3, 1)
#
#     cover_name_numpy = cover_name.clone().cpu().detach().numpy()
#
#     water_name_numpy = water_name.clone().cpu().detach().numpy()
#
#     N, _, _, _ = container_patch.shape
#     for B1 in range(batch_size):
#         box_p = box[B1]
#         cover_numpy[B1][box_p[0]:box_p[1], box_p[2]:box_p[3]] = container_patch_numpy[B1]
#         watermark_show = cv2.cvtColor(watermark_numpy[B1], cv2.COLOR_RGB2BGR)
#         container_patch_show = cv2.cvtColor(cover_numpy[B1], cv2.COLOR_RGB2BGR)
#         # print('======save container======')
#
#         cover_name_str = '%012d' % cover_name_numpy[B1]
#         water_name_str = '%012d' % water_name_numpy[B1]
#         print(cover_name_str)
#         cv2.imwrite(tamp_c_dir + "//" + cover_name_str + '.png', container_patch_show * 255)
#         cv2.imwrite(tamp_w_dir + "//" + cover_name_str + '.png', watermark_show * 255)
# print('======generation done======')

test_dataset = myDataset(tamp_c_dir, tamp_w_dir, width, high, resize=resize)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
min_loss_val = 10
best_model = None
print('======Rnet training======')
for i in range(epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    Psnr = AverageMeter()

    for patch_r, cover_r, watermark_r, box_r, cover_r_name, water_r_name in test_loader:
        # cover_name_numpy = cover_r_name.clone().cpu().detach().numpy()
        # #
        watermark_r_numpy = watermark_r.clone().cpu().detach().numpy()
        watermark_r_numpy = watermark_r_numpy.transpose(0, 2, 3, 1)
        # cover_name_str = '%012d' % cover_name_numpy
        # water_name_str = '%012d' % water_name_numpy
        # print(cover_name_str)
        Rnet.train()
        optimizer.zero_grad()
        patch_r = patch_r.cuda()
        watermark_r = watermark_r.cuda()
        rev_secret_img = Rnet(patch_r)
        # L_1 = L1_loss(watermark_r, rev_secret_img)
        L_2 = L2_loss(watermark_r, rev_secret_img)
        L_R = L_2 * 100
        L_R.backward()
        optimizer.step()

        diffR = (rev_secret_img - watermark_r).abs().mean() * 255
        rev_secret_img_numpy = rev_secret_img.clone().cpu().detach().numpy()
        rev_secret_img_numpy = rev_secret_img_numpy.transpose(0, 2, 3, 1)

        watermark_r_numpy = watermark_r.clone().cpu().detach().numpy()
        watermark_r_numpy = watermark_r_numpy.transpose(0, 2, 3, 1)

        for i1 in range(batch_size):
            rev_secret_img_show = cv2.cvtColor(rev_secret_img_numpy[i1], cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(rev_dir, str(i1) + '.jpg'), rev_secret_img_show * 255)
            # cv2.imwrite(os.path.join(save_path, str(i1) + '.jpg'), rev_secret_numpy[i1]*255)
            psnr_p[i1, 0] = PSNR(rev_secret_img_numpy[i1, :, :, 0], watermark_r_numpy[i1, :, :, 0])
            psnr_p[i1, 1] = PSNR(rev_secret_img_numpy[i1, :, :, 1], watermark_r_numpy[i1, :, :, 1])
            psnr_p[i1, 2] = PSNR(rev_secret_img_numpy[i1, :, :, 2], watermark_r_numpy[i1, :, :, 2])
        # # print("Avg. PSNR P:", psnr_p.mean().item())

        Rlosses.update(L_R.item(), batch_size)  # R loss
        Psnr.update(psnr_p.mean().item(), batch_size)
        Rdiff.update(diffR.item(), batch_size)
        log = '[%d/%d]\t Loss_R: %.6f ADP: %.4f PSNR: %.4f' % (
            i, epoch, Rlosses.val, Rdiff.val, Psnr.val)
        # if i % batch_size == 0:
        print(log)
        if Rlosses.val < min_loss_val:
            min_loss_val = Rlosses.val
            best_model = copy.deepcopy(Rnet)
    torch.save(Rnet, os.path.join(save_path, 'DIPW_' + str(i) + '_epoch.pth'))
torch.save(best_model, os.path.join(save_path, 'DIPW_best_loss.pth'))
