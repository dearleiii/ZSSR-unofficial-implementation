#!/usr/bin/env python
import sys
import os
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def _open_img(img_p):
    F = scipy.misc.fromimage(Image.open(img_p)).astype(float)/255.0
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def compute_psnr(ref_im, res_im):
    return output_psnr_mse(
        _open_img(os.path.join(hr_dir,ref_im)),
        _open_img(os.path.join(output_dir,res_im))
        )

SCALE = 1
# as per the metadata file, input and output directories are the arguments
# [_, input_dir, output_dir] = sys.argv
# print(input_dir, output_dir)
hr_dir = '/Users/dearleiii/Desktop/2019Duke/SISR/sr_algorithms/RCAN/RCAN-master/RCAN_TestCode/HR/Set5/x2'
output_dir = '/Users/dearleiii/Desktop/2019Duke/SISR/sr_algorithms/ZSSR/unofficial_implement/pytorch-zssr-master/result'

ref_im = 'baby_HR_x2.png'
zssr_im = 'zssr.png'
# res_dir = os.path.join(input_dir, 'res/')
# ref_dir = os.path.join(input_dir, 'ref/')
#print("REF DIR")
#print(ref_dir)

zssr_psnr = compute_psnr(ref_im,zssr_im)
print(zssr_psnr)

output_dir = '/Users/dearleiii/Desktop/2019Duke/SISR/sr_algorithms/RCAN/RCAN-master/RCAN_TestCode/SR/BI/RCAN/Set5/x2'
rcan_im = 'baby_RCAN_x2.png'
rcan_psnr = compute_psnr(ref_im,rcan_im)
print(rcan_psnr)