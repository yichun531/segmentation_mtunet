#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
from medpy import metric
import torch.nn as nn
from PIL import Image
import torchvision
import SimpleITK as sitk
from scipy.ndimage import zoom

class Normalize():
    def __call__(self, sample):

        function = torchvision.transforms.Normalize((.5 , .5, .5), (0.5, 0.5, 0.5))
        return function(sample[0]), sample[1]


class ToTensor():
    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        function = torchvision.transforms.ToTensor()
        return function(sample[0]), function(sample[1])


class RandomRotation():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        random_angle = np.random.randint(0, 360)
        return img.rotate(random_angle, Image.NEAREST), label.rotate(random_angle, Image.NEAREST)


class RandomFlip():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        temp = np.random.random()
        if temp > 0 and temp < 0.25:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        elif temp >= 0.25 and temp < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)
        elif temp >= 0.5 and temp < 0.75:
            return img.transpose(Image.ROTATE_90), label.transpose(Image.ROTATE_90)
        else:
            return img, label


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        # return ` reduction='mean' ` loss
        return loss / self.n_classes 


# def calculate_metric_percase(output, target):
#     smooth = 1e-5  

#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     if output.sum() > 0 and target.sum() > 0:
#         hd = metric.binary.hd(output, target)
#     else:
#         hd = 0
#     intersection = (output * target).sum()

#     return (2. * intersection + smooth) / \
#            (output.sum() + target.sum() + smooth), hd

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_image(image, filename, model, output_size, test_save_path=None):
    if isinstance(image, np.ndarray):
        input = torch.from_numpy(image).float().cuda()
    else:
        input = image.float().cuda()
    model.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(model(input), dim=1), dim=1) # (1,H,W)
        prediction = out.float().cpu().detach()
        # resizing prediction (mask) into original shape (resize and crop out zero-padding) 
        # reverse of `dataset_STAS.resize_with_pad()`
        max_length = output_size[np.argmax(output_size)]
        min_length = output_size[np.argmin(output_size)]
        transform = torchvision.transforms.Resize((max_length,max_length),interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        prediction = transform(prediction)
        prediction =torchvision.transforms.functional.crop(prediction, top=(max_length-min_length)//2, left=0,  height=output_size[0],  width=output_size[1])
     
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        torchvision.utils.save_image(prediction, os.path.join(test_save_path, filename+".png"))
