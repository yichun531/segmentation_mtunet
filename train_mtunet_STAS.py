#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
# import matplotlib.pyplot as plt
from utils.utils_STAS import DiceLoss
from torch.utils.data import DataLoader
from dataset.dataset_STAS import STASdataset, RandomGenerator
import argparse
from tqdm import tqdm
import os, sys, random
from torchvision import transforms
from utils.test_STAS import inference
from model.MTUNet_STAS import MTUNet
import numpy as np
from medpy.metric import dc,hd95
from tensorboardX import SummaryWriter
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-exp_name", default='exp', type=str,help="name of this experiment. It decides the save_model_path and save_log_path")
parser.add_argument("-batch_size", default=12, type=int, help="batch size")
parser.add_argument("-lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("-max_epochs", default=100, type=int)
parser.add_argument("-img_size", default=224, type=int)
parser.add_argument("-no_global_self_attention",default=False,action='store_true',help="remove global self-attention in model")
parser.add_argument("-data_augment",default=False,action='store_true',help="augmenting 2-times training data")
parser.add_argument("-soft_label",default=False,action='store_true',help="mask pixel values are probabilities")
parser.add_argument("-save_model_path", default="saved/checkpoints", type=str)
parser.add_argument("-save_image_path", default="saved/images", type=str)
parser.add_argument("-save_log_path", default="logs/", type=str)
parser.add_argument("-n_gpu", default=1, type=int)
parser.add_argument("-checkpoint", default=None, type=str)
parser.add_argument("-train_dir", default="path/to/STAS_data/SEG_Train_Datasets/", type=str)
parser.add_argument("-test_dir", default="path/to/STAS_data/Public_Image/", type=str)
parser.add_argument("-num_classes", default=2, type=int)
parser.add_argument("-n_skip", default=1, type=int)
args = parser.parse_args()

args.save_model_path = os.path.join(args.save_model_path, args.exp_name)
args.save_image_path = os.path.join(args.save_image_path, args.exp_name)
args.save_log_txt_path = os.path.join(args.save_log_path, 'text_records', args.exp_name)
args.save_log_tfb_path = os.path.join(args.save_log_path, 'tensorboard', args.exp_name)
os.makedirs(args.save_model_path,exist_ok=True)
os.makedirs(args.save_image_path,exist_ok=True)
os.makedirs(args.save_log_txt_path,exist_ok=True)
os.makedirs(args.save_log_tfb_path,exist_ok=True)
# save config
with open(os.path.join(args.save_model_path, 'config.json'), 'w', encoding='utf-8') as fout:
    fout.write('{')
    fout.write(''.join('"'+str(k)+'"'+':'+str(v)+',\n' for k,v in vars(args).items()))
    fout.write('}')

# save logging text 
run_log_counter = 0
while(os.path.exists(os.path.join(args.save_log_txt_path, 'run_{}.txt'.format(run_log_counter)))):
    run_log_counter += 1
file_log = open(os.path.join(args.save_log_txt_path, 'run_{}.txt'.format(run_log_counter)),'w')  # File where you need to keep the logs
file_log.write("")
    
# Write the data of stdout here to a text file as well
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        file_log.write(data)    # Write the data of stdout here to a text file as well
    def flush(self):
        pass
sys.stdout = Unbuffered(sys.stdout)
    
# tensorboard logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_writer = SummaryWriter(os.path.join(args.save_log_tfb_path, current_time, 'train'))
eval_writer = SummaryWriter(os.path.join(args.save_log_tfb_path, current_time, 'eval'))

# set up random seed
def seed_torch(seed=1203):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

train_dataset = STASdataset(args.train_dir, split="train", output_size=[args.img_size, args.img_size],
                            soft_label=args.soft_label, data_augment=args.data_augment, 
                            transform = transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val=STASdataset(args.train_dir, split="valid", output_size=[args.img_size, args.img_size], 
                   soft_label=args.soft_label)
val_loader=DataLoader(db_val, batch_size=1, shuffle=False)
db_test =STASdataset(args.test_dir, split="test", output_size=[args.img_size, args.img_size],
                    soft_label=args.soft_label)
test_loader = DataLoader(db_test, batch_size=1, shuffle=False)

model=MTUNet(args.num_classes, not args.no_global_self_attention)
device=torch.device("cuda")
# if args.n_gpu > 1:
#     device_id=[i for i in range(args.n_gpu)]
#     model = nn.DataParallel(model,device_ids=device_id).to(device)    
# model = model.cuda()
# if args.checkpoint:
#     model.load_state_dict(torch.load(args.checkpoint))
if args.checkpoint:
    state_dict=torch.load(args.checkpoint)
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Skipped:" + name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print("Successfully loaded: "+name)
        except:
            print("Part load failed: " + name)
if args.n_gpu > 1:
    device_id=[i for i in range(args.n_gpu)]
    model = nn.DataParallel(model,device_ids=device_id).to(device)   
model = model.cuda()

    
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
save_interval = args.n_skip  # int(max_epoch/n_skip)

Best_dc = 0.85
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

max_iterations = args.max_epochs * len(train_loader)
base_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

def val():
    logging.info("Validation ===>")
    dc_sum = 0
    val_loss = 0
    model.eval()
    for i, val_sampled_batch in enumerate(tqdm(val_loader)):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.to(device), val_label_batch.to(device)
        
        val_outputs = model(val_image_batch)
        # if mask is one-hot pixel : (B, 2, H ,W) -> need to softmax mask to [B, H, W]
        if args.soft_label:
            val_label_batch = torch.argmax(torch.softmax(val_label_batch, dim=1), dim=1)
        # if mask is not one-hot pixel: (B, H, W)
        loss_ce = ce_loss(val_outputs, val_label_batch[:].long())
        loss_dice = dice_loss(val_outputs, val_label_batch[:], softmax=True)
        loss = loss_dice * 0.5 + loss_ce * 0.5
        val_loss += loss.item()
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1) # (B, H, W)
        dc_sum += dc(val_outputs.cpu().data.numpy(),val_label_batch[:].cpu().data.numpy())
    logging.info("avg_dc: %f loss : %f" % (dc_sum/len(val_loader), val_loss/len(val_loader)))
    return dc_sum/len(val_loader), val_loss/len(val_loader)

iter_num = 0
for epoch in range(args.max_epochs):
    logging.info("Training ===> Epoch %d : " % (epoch+1))
    model.train()
    train_loss = 0
    dc_sum = 0
    for i_batch, sampled_batch in enumerate(tqdm(train_loader)):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        
        outputs = model(image_batch)

        # if mask is one-hot pixel: (B, 2, H ,W) -> need to softmax mask to [B, H, W]
        if args.soft_label:
            label_batch = torch.argmax(torch.softmax(label_batch, dim=1), dim=1)
        # if mask is not one-hot pixel: (B, H, W)
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
        loss = loss_dice * 0.5 + loss_ce * 0.5
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if epoch > args.max_epochs//2 :
        #    iter_num = iter_num + 1
        #    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        #else:
        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        train_loss += loss.item()
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1) # (B, H, W)
        dc_sum += dc(outputs.cpu().data.numpy(),label_batch[:].cpu().data.numpy())
        
    logging.info('avg_dc: %f loss : %f lr_: %f' % (dc_sum/len(train_loader), train_loss/len(train_loader), lr_))
    train_writer.add_scalar('loss', train_loss/len(train_loader), epoch+1)
    train_writer.add_scalar('avg_dc', dc_sum/len(train_loader), epoch+1)
    
    if (epoch + 1) % save_interval == 0:
        avg_dc, val_loss = val()
        eval_writer.add_scalar('avg_dc', avg_dc, epoch+1)
        eval_writer.add_scalar('loss', val_loss, epoch+1)
        # avg_dc = dc_sum/len(train_loader)
    
        if avg_dc > Best_dc:
            save_model_path = os.path.join(args.save_model_path, 'epoch=%d_lr=%f_avg_dc=%.3f_loss=%.4f.pth' % (epoch+1, lr_, avg_dc, train_loss/len(train_loader)))
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            #temp = 1
            Best_dc = avg_dc

            inference(args, model, test_loader, output_size=train_dataset.original_shape(), test_save_path=os.path.join(args.save_image_path, 'e_{}'.format(epoch+1)))
            
    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(args.save_model_path, 'epoch=%d_lr=%f_avg_dc=%.3f_loss=%.4f.pth' % (epoch+1, lr_, avg_dc, train_loss/len(train_loader)))
        torch.save(model.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        
