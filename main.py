import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn

#import network
import dataloader
from train_and_test import train_one_epoch, evaluate_one_epoch

from colorama import Fore, Style

from torch.utils.tensorboard import SummaryWriter

#from vit import ViT
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import CLIPVisionModel

#from networks import GeoCLIP, VGGTriplet, BasicNetVLAD
import networks 
import RepNetModel


parser = argparse.ArgumentParser()

opt = parser.parse_args()
opt.kernels = multiprocessing.cpu_count()
opt.size = 224

opt.n_epochs = 300

opt.description = "Full S4 decoders reducing backbone (one extra)"
opt.evaluate = False

opt.lr = 6e-6

opt.batch_size = 5


train_dataset = dataloader.Countix(split='train', dir='/home/alec/Documents/BigDatasets/Countix', prefix='countix')
val_dataset = dataloader.Countix(split='val', dir='/home/alec/Documents/BigDatasets/Countix', prefix='countix', inference=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)

#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)

opt.device = torch.device('cuda')

#ground_model = ViT(image_size = 224, patch_size = 32, dim = 768, depth = 16, heads = 8)
#aerial_model = ViT(image_size = 224, patch_size = 32, dim = 768, depth = 16, heads = 8)


criterion = torch.nn.CrossEntropyLoss()
#model = RepNetModel.RepNet(num_frames=64)
model = networks.Network7()
#model.load_state_dict(torch.load('bce_mae_count_one_tr9.pt')['state_dict'], strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.1)

writer = SummaryWriter(log_dir='runs/'+opt.description)

_ = model.to(opt.device)

acc10 = 0
for epoch in range(opt.n_epochs): 

    if not opt.evaluate:
        _ = model.train()

        #train_one_epoch(train_dataloader, ground_model, aerial_model, ground_optimizer, aerial_optimzer, criterion, opt, epoch, writer)
        #train_one_epoch_temp1(train_dataloader, model, optimizer, opt, epoch, writer)
        train_one_epoch(train_dataloader, model, optimizer, criterion, opt, epoch, writer)
    
    
    
    #acc10 = max(acc10, validate_one_epoch(val_dataloader, model, opt, epoch, writer))

    #print("Best acc10 is", acc10)
    evaluate_one_epoch(val_dataloader, model, opt, epoch, writer)

