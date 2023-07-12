from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR
import cv2
import numpy as np

img1 = cv2.imread(r'D:\code\DIPW-main\DIPW\dataset\rev_watermark\000000000019.png')
img1 = cv2.resize(img1, (128, 128)) / 255
img2 = cv2.imread(r'D:\code\DIPW-main\DIPW\dataset\watermarkV2\1.PNG')
img2 = cv2.resize(img2, (128, 128)) / 255
a = np.mean(np.abs(img1-img2))*255
print(a)

