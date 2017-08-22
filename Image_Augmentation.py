
# coding: utf-8

import numpy as np
import cv2
from scipy.misc import imread, imsave, imresize
import scipy.ndimage
import matplotlib.pyplot as plt


def Basic_Augmentation(image,
                       resize_use=True,min_scale=150,max_scale=230,
                       rotate_use=True,min_angle=0,max_angle=360,
                       pad_use=True,pad_size=300,
                       shift_use=True,min_shift=-30,max_shift=30,
                       output_size=224):
    if resize_use:
        resize_row = np.uint8(np.random.random()*(max_scale-min_scale)+min_scale)
    else:
        resize_row = image.shape[0]
    
    if rotate_use:
        rotate_angle = np.uint8(np.random.random()*(max_angle-min_angle)+min_angle)
    else:
        rotate_angle = 0
        
    if shift_use:
        shift_row = np.int8((np.random.random()*(max_shift-min_shift))+min_shift)
        shift_col = np.int8((np.random.random()*(max_shift-min_shift))+min_shift)
    else:
        shift_row = 0
        shift_col = 0
    

    rotate_image = scipy.ndimage.rotate(image, rotate_angle,mode = 'nearest',reshape = False)
    resize_col = np.float32(rotate_image.shape[1])*resize_row/np.float32(image.shape[0])
    resize_image = imresize(rotate_image,[int(resize_row),int(resize_col)],interp ='bicubic')

    expected_size = pad_size
    pad_width = ((int(np.floor((expected_size-resize_row)/2)),int(np.ceil((expected_size-resize_row)/2))),
                 (int(np.floor((expected_size-resize_col)/2)),int(np.ceil((expected_size-resize_col)/2))),(0,0))
    pad_image = np.pad(resize_image,pad_width=pad_width,mode='edge')

    shift_factor = [shift_row,shift_col]
    
    center = [int(np.floor(expected_size/2))+shift_factor[0],int(np.floor(expected_size/2))+shift_factor[1]]
    crop_image = pad_image[center[0]-int(np.floor(output_size/2)):center[0]+int(np.ceil(output_size/2)),
                           center[1]-int(np.floor(output_size/2)):center[1]+int(np.ceil(output_size/2))]
    crop_image = np.uint8(crop_image)
    
    return crop_image,resize_row,rotate_angle,shift_row,shift_col
