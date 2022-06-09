#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils_STAS import test_single_image


def inference(args, model, testloader, output_size, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, filename = sampled_batch["image"], sampled_batch["filename"]
            test_single_image(image, filename[0], model, output_size=output_size, test_save_path=test_save_path)
        logging.info("Testing Finished!")
        
