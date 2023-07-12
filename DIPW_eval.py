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

# from time import *
# begin_time = time()

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
Rnet = torch.load(r'D:\code\DIPW-main\DIPW\checkpoint_back_up\DIPW_99_epoch.pth')
# D:\code\DIPW-main\DIPW\checkpoint_back_up\DIPW_99_epoch.pth
# D:\code\DIPW-main\DIPW\ckpt\checkpoint_2022_03_30_16_52\DIPW_best_loss.pth
Hnet.load_state_dict(checkpoint['H_state_dict'])
# Rnet.load_state_dict(checkpoint['R_state_dict'], False)
train_cover_dir = r'D:\code\DIPW-main\DIPW\dataset\coverV2'
train_watermark_dir = r'D:\code\DIPW-main\DIPW\dataset\watermarkV2'
tamp_c_dir = r'D:\code\DIPW-main\DIPW\dataset\containerV2'
tamp_w_dir = r'D:\code\DIPW-main\DIPW\dataset\temp_watermarkV2'
rev_dir = r'D:\code\DIPW-main\DIPW\dataset\rev_watermark'

width = 128
high = 128
epoch = 100
batch_size = 1
resize = True
train_dataset = myDataset(train_cover_dir, train_watermark_dir, width, high, resize=resize)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

val_dataset = myDataset(train_cover_dir, train_watermark_dir, width, high, resize=False)
val_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

save_path = 'D:\code\DIPW-main\DIPW\checkpoint'
optimizer = optim.SGD(Hnet.parameters(), lr=0.0001, momentum=0.9)
L1_loss = nn.L1Loss().cuda()
L2_loss = nn.MSELoss().cuda()
a = 0.75
loss = 10000

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


psnr_c = np.zeros((batch_size, 3))
psnr_w = np.zeros((batch_size, 3))
ssim_c = np.zeros(batch_size)
ssim_w = np.zeros(batch_size)
lpipc_sum = []
diffC_sum = []
psnr_c_sum = []
ssim_c_sum = []
lpipw_sum = []
diffw_sum = []
psnr_w_sum = []
ssim_w_sum = []


# print("dataset generate")
for patch, cover, watermark, box, cover_name, water_name in train_loader:
    Hnet.eval()
    with torch.no_grad():
        patch = patch.cuda()
        cover = cover.cuda()
        cover_o = torch.clone(cover)
        watermark = watermark.cuda()
        itm_secret_img = Hnet(watermark)
        container_patch = itm_secret_img + patch

        container_patch_numpy = container_patch.clone().cpu().detach().numpy()
        container_patch_numpy = container_patch_numpy.transpose(0, 2, 3, 1)

        cover_o_numpy = cover_o.clone().cpu().detach().numpy()
        cover_o_numpy = cover_o_numpy.transpose(0, 2, 3, 1)

        cover_numpy = cover.clone().cpu().detach().numpy()
        cover_numpy = cover_numpy.transpose(0, 2, 3, 1)

        watermark_numpy = watermark.clone().cpu().detach().numpy()
        watermark_numpy = watermark_numpy.transpose(0, 2, 3, 1)

        cover_name_numpy = cover_name.clone().cpu().detach().numpy()

        water_name_numpy = water_name.clone().cpu().detach().numpy()

        N, _, _, _ = container_patch.shape
        import PerceptualSimilarity.models
        lpip_model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

        for B1 in range(batch_size):
            box_p = box[B1]
            cover_numpy[B1][box_p[0]:box_p[1], box_p[2]:box_p[3]] = container_patch_numpy[B1]
            watermark_show = cv2.cvtColor(watermark_numpy[B1], cv2.COLOR_RGB2BGR)
            container_patch_show = cv2.cvtColor(cover_numpy[B1], cv2.COLOR_RGB2BGR)
            # print('======save container======')

            cover_name_str = '%012d' % cover_name_numpy[B1]
            water_name_str = '%012d' % water_name_numpy[B1]
            path = r'D:\code\DIPW-main\DIPW\dataset\coverV2'

            img = cv2.imread(os.path.join(path, cover_name_str + '.jpg'))
            print(cover_name_str + '.png')
            # container_patch_show = cv2.resize(container_patch_show, (img.shape[1], img.shape[0]))
            cv2.imwrite(tamp_c_dir + "//" + cover_name_str + '.png', container_patch_show * 255)
            cv2.imwrite(tamp_w_dir + "//" + cover_name_str + '.png', watermark_show * 255)
        a = cover_numpy.transpose(0, 3, 1, 2)
        a = torch.from_numpy(a)
        lpipc = lpip_model.forward(cover.cpu().detach(), a)
        print("LPIPS C:", lpipc.mean().item())
        lpipc_sum.append(lpipc.mean().item())

        for i2 in range(batch_size):
            psnr_c[i2, 0] = PSNR(cover_o_numpy[i2, :, :, 0], cover_numpy[i2, :, :, 0])
            psnr_c[i2, 1] = PSNR(cover_o_numpy[i2, :, :, 1], cover_numpy[i2, :, :, 1])
            psnr_c[i2, 2] = PSNR(cover_o_numpy[i2, :, :, 2], cover_numpy[i2, :, :, 2])
            ssim_c[i2] = SSIM(cover_o_numpy[i2], cover_numpy[i2], multichannel=True)
        print("Avg. PSNR C:", psnr_c.mean().item())
        psnr_c_sum.append(psnr_c.mean().item())
        print("Avg. SSIM C:", ssim_c.mean().item())
        ssim_c_sum.append(ssim_c.mean().item())
        diffC = (cover.cpu().detach() - a).abs().mean() * 255
        diffC_sum.append(diffC)
        print("Secret APD C:", diffC.item())

    # print('======generation done======')

test_dataset = myDataset(tamp_c_dir, tamp_w_dir, width, high, resize=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

for patch_r, cover_r, watermark_r, box_r, cover_r_name, water_r_name in test_loader:
    Rnet.eval()
    with torch.no_grad():
        patch_r = patch_r.cuda()
        watermark_r = watermark_r.cuda()
        rev_secret_img = Rnet(patch_r)

        diffR = (rev_secret_img - watermark_r).abs().mean() * 255
        diffw_sum.append(diffR.item())

        cover_img_numpy = cover_r.clone().cpu().detach().numpy()
        cover_img_numpy = cover_img_numpy.transpose(0, 2, 3, 1)

        container_patch = torch.clone(cover_r)
        container_patch = container_patch.clone().cpu().detach().numpy()
        container_patch = container_patch.transpose(0, 2, 3, 1)

        container_img_numpy = patch_r.clone().cpu().detach().numpy()
        container_img_numpy = container_img_numpy.transpose(0, 2, 3, 1)

        rev_secret_img_numpy = rev_secret_img.clone().cpu().detach().numpy()
        rev_secret_img_numpy = rev_secret_img_numpy.transpose(0, 2, 3, 1)

        watermark_r_numpy = watermark_r.clone().cpu().detach().numpy()
        watermark_r_numpy = watermark_r_numpy.transpose(0, 2, 3, 1)

        # LPIPS
        import PerceptualSimilarity.models
        lpip_model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        lpipw = lpip_model.forward(rev_secret_img, watermark_r)

        water_r_name_numpy = water_r_name.clone().cpu().detach().numpy()
        lpipw_sum.append(lpipw.mean().item())

        for i1 in range(batch_size):
            water_r_name_str = '%012d' % water_r_name_numpy[i1]
            rev_secret_img_show = cv2.cvtColor(rev_secret_img_numpy[i1], cv2.COLOR_RGB2BGR)
            print(water_r_name_str + '.png')
            cv2.imwrite(os.path.join(rev_dir, water_r_name_str + '.png'), rev_secret_img_show * 255)

            psnr_w[i1, 0] = PSNR(rev_secret_img_numpy[i1, :, :, 0], watermark_r_numpy[i1, :, :, 0])
            psnr_w[i1, 1] = PSNR(rev_secret_img_numpy[i1, :, :, 1], watermark_r_numpy[i1, :, :, 1])
            psnr_w[i1, 2] = PSNR(rev_secret_img_numpy[i1, :, :, 2], watermark_r_numpy[i1, :, :, 2])

            ssim_w[i1] = SSIM(rev_secret_img_numpy[i1], watermark_r_numpy[i1], multichannel=True)

        psnr_w_sum.append(psnr_w.mean().item())
        ssim_w_sum.append(ssim_w.mean().item())
        print("Avg. PSNR W:", psnr_w.mean().item())
        print("Avg. SSIM W:", ssim_w.mean().item())
        print("Avg. LPIPS W:", lpipw.mean().item())
        print("Secret APD W:", diffR.item())


avg_psnr_c = np.mean(np.array(psnr_c_sum))
avg_ssim_c = np.mean(np.array(ssim_c_sum))
avg_lpipc_sum = np.mean(np.array(lpipc_sum))
avg_diffc = np.mean(np.array(diffC_sum))

avg_psnr_w = np.mean(np.array(psnr_w_sum))
avg_ssim_w = np.mean(np.array(ssim_w_sum))
avg_lpipw_sum = np.mean(np.array(lpipw_sum))
avg_diffw = np.mean(np.array(diffw_sum))

print('========')
print('|Average PSNR C %5f|' % avg_psnr_c)
print('|Average SSIM C %5f|' % avg_ssim_c)
print('|Average LPIP C %5f|' % avg_lpipc_sum)
print('|Average APD  C %5f|' % avg_diffc)
print('|Average PSNR W %5f|' % avg_psnr_w)
print('|Average SSIM W %5f|' % avg_ssim_w)
print('|Average LPIP W %5f|' % avg_lpipw_sum)
print('|Average APD  W %5f|' % avg_diffw)

# end_time = time()
# run_time = end_time-begin_time
# print ('该循环程序运行时间：',run_time)