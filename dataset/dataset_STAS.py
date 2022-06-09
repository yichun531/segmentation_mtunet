#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage.util.shape import view_as_windows
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import json
import cv2
import glob


def random_rot_flip(image, label):
    k = np.random.randint(1, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def resize_with_pad(image=None, label=None, output_size=(224,224), soft_label=False):
    input_img = image if image is not None else label
    # if mask is not one-hot pixel: (H,W,2) -> prevent more unwanted classes (other than 0,1) happens in resized mask
    if not soft_label:
        interpolate = T.InterpolationMode.BILINEAR if image is not None else T.InterpolationMode.NEAREST
    # if mask is one-hot pixel: (H,W,1) -> let pixels in mask become  soft label (probability) (0/1 is hard label)
    else:
        interpolate = T.InterpolationMode.BICUBIC
    ######
    x, y, c = input_img.shape
    max_length=max(x,y)
    input_img = np.pad(input_img, [((max_length-x)//2, (max_length-x)-((max_length-x)//2)),
                           ((max_length-y)//2, (max_length-y)-((max_length-y)//2)),
                           (0,0)], mode='constant', constant_values=0.)
    input_img = np.transpose(input_img, [2,0,1]) # (C, H, W)
    transform = T.Resize(output_size,interpolation=interpolate)
    input_img = transform(torch.from_numpy(input_img))
    return input_img

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # random crop
        w_image = view_as_windows(image, (512,512,3), step=256)
        w_label = view_as_windows(label, (512,512,label.shape[-1]), step=256)
        row, col, depth = w_image.shape[:3]
        r = np.random.randint(low=0, high=row)
        c = np.random.randint(low=0, high=col)
        d = np.random.randint(low=0, high=depth)
        image = w_image[r,c,d]
        label = w_label[r,c,d]
        # random rotate and flip
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # resize
        image = resize_with_pad(image=image, output_size=self.output_size)
        label = resize_with_pad(label=label, output_size=self.output_size)
        image = image / 255. # normalize to [0,1]
        return image, label

def binary_mask(mask):
    # input `mask` has shape (H,W,1)
    mask_0 = np.ones_like(mask)
    mask_0 = mask_0-mask
    mask_1 = mask
    mask = np.concatenate([mask_0,mask_1],axis=-1) # (H,W,2)
    return mask
    
class STASdataset(Dataset):
    def __init__(self, data_dir, split, output_size, soft_label=False, transform=None, data_augment=False):
        self.soft_label = soft_label
        self.transform = transform  # using transform in torch!
        self.output_size = output_size
        self.split = split
        self.data_dir = data_dir
        self.data_augment = data_augment
        if self.split == 'train':
            self.sample_list = sorted(glob.glob(os.path.join(self.data_dir,'Train_Images','*.jpg')))
            self.annotat_list = sorted(glob.glob(os.path.join(self.data_dir,'Train_Annotations','*.json')))
            # take back 2/3 samples as training data
            self.sample_list = self.sample_list[len(self.sample_list)//3:] 
            self.annotat_list = self.annotat_list[len(self.annotat_list)//3:] 
        elif self.split == "valid":
            self.sample_list = sorted(glob.glob(os.path.join(self.data_dir,'Train_Images','*.jpg')))
            self.annotat_list = sorted(glob.glob(os.path.join(self.data_dir,'Train_Annotations','*.json')))
            # take front 1/3 samples as training data
            self.sample_list = self.sample_list[:len(self.sample_list)//3] 
            self.annotat_list = self.annotat_list[:len(self.annotat_list)//3] 
        else: # testing data
            self.sample_list = sorted(glob.glob(os.path.join(self.data_dir,'*.jpg')))
        
        self.init_len=len(self.sample_list)
        if self.split=='train' and self.data_augment:
            self.sample_list = self.sample_list*3 #.extend(self.sample_list)
            self.annotat_list = self.annotat_list*3 #.extend(self.annotat_list)
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            # load image
            image = Image.open(self.sample_list[idx]).convert("RGB") # (H, W, 3)
            image = np.asarray(image)
            self.init_shape=image.shape[:-1]
            # load mask from annotation
            with open(self.annotat_list[idx], 'r') as f:
                ant = json.load(f)
            coordinates = []    
            for item in ant['shapes']:
                coordinates.append(np.array(item['points'],dtype=np.int32))
            mask=cv2.fillPoly(np.zeros(image.shape[:-1]+(1,)),
                              pts=coordinates,color=1) # (H, W, 1)
            # processing
            # if mask is one-hot pixel : (H,W,2)
            if self.soft_label:
                mask = binary_mask(mask) # (H,W,2)
            if self.data_augment and idx>=self.init_len and self.split == "train" and self.transform:
                image, mask = self.transform({'image': image, 'label': mask})
            else:
                image = resize_with_pad(image=image, output_size=self.output_size) # [C,H,W]
                mask = resize_with_pad(label=mask, output_size=self.output_size) # [C,H,W]
                image = image / 255. # normalize to [0,1]
            # if mask is not one-hot pixel : (1, H, W)
            if not self.soft_label:
                mask = mask.squeeze(0) # (H, W)
            filename = self.sample_list[idx].split(os.sep)[-1].split('.jpg')[0]
            sample = {'image': image, 'label': mask, 'filename': filename}
        else: # testing data
            # load image
            image = Image.open(self.sample_list[idx]).convert("RGB")
            image = np.asarray(image)
            self.init_shape=image.shape[:-1]
            # processing
            image = resize_with_pad(image=image, output_size=self.output_size) # [C,H,W]
            image = image / 255. # normalize to [0,1]
            filename = self.sample_list[idx].split(os.sep)[-1].split('.jpg')[0]
            sample = {'image': image, 'filename': filename}
        
        return sample
    
    def original_shape(self):
        return self.init_shape
        