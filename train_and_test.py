import numpy as np
from dataloader import Countix
from networks import *
import torch
import torchvision
from torch import nn
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
import os

from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
import time

import sys

from tqdm import tqdm

from einops import rearrange

def evaluate_one_epoch(test_dataloader, model, opt, epoch, writer):

    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = test_dataloader 

    losses = []

    print("Evaluating Epoch", epoch)

    total_vids = len(test_dataloader.dataset.name_list)

    fi = 0
    pred = np.zeros(total_vids, 'float32')
    truth = np.zeros(total_vids, 'float32')

    counting_tensor = torch.Tensor(np.arange(2,36)).type(torch.FloatTensor).cuda().unsqueeze(0)
    
    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))
    for i, (video, period, truecount) in bar:

        bs, ch, f, h, w = video.shape
        #print(video.shape)
        #N = N.to(opt.device)

        batch_times.append(time.time() - tt_batch)
        #s = list(A.shape)    

        with torch.no_grad():
             x, within_period_x = model(video.cuda())
        #x = torch.sum(x*counting_tensor, dim=1)
        #print(x.dtype)
        #print(count.dtype)


        x = x.mean(1)
        loss = F.cross_entropy(x.detach().cpu(), period - 1)

        losses.append(loss.cpu().data)

        pred[fi:fi + x.shape[0]] = 64 / (np.argmax(x.detach().cpu().numpy(), axis=1) + 1)
        truth[fi:fi + period.shape[0]] = 64 / period.cpu().numpy()
        fi += x.shape[0]

        #print(crit.shape)

        #print(x.shape)
        #print(count.shape)
        #print(np.absolute(x - count) > 1)
        #print(total_mae)
        #print(F.l1_loss(x.squeeze(1).cpu(), count.cpu()))
        
    print("total vids is", total_vids)
    mae = np.sum(np.absolute(pred - truth) / truth) / total_vids
    #mae = mean_absolute_error(truth, pred)
    gaps = np.absolute(pred - truth)

    obo = 0
    for item in gaps:
        if item > 1:
            obo += 1


    pred = pred.astype(int)
    truth = truth.astype(int)
    print(np.histogram(pred, np.arange(33)))
    print(np.histogram(truth, np.arange(33)))
    print("Epoch", epoch, "MAE:", mae, "OBO:", obo / total_vids)

    writer.add_scalar('Loss/Test', np.mean(losses), epoch)
    writer.add_scalar('MAE/Test', mae, epoch)
    writer.add_scalar('OBO/Test', obo / total_vids, epoch)

def train_one_epoch(train_dataloader, model, optimizer, criterion ,opt, epoch,  writer):

    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []
    running_loss = 0.0
    dataset_size = 0

    criterion1 = torch.nn.MSELoss()

    fi = 0
    total_vids = len(train_dataloader.dataset.name_list)
    pred = np.zeros(total_vids, 'float32')
    truth = np.zeros(total_vids, 'float32')
    
    torch.set_printoptions(profile="full")
    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))
    for i, (video, period, truecount) in bar:

        #print(video.shape)
        video = video.to(opt.device)
        period = period.to(opt.device)
        batch_size = video.shape[0]

        batch_times.append(time.time() - tt_batch)
        #s = list(A.shape)    

        x, within_period_x = model(video)
        #print("X shape is", x)

        optimizer.zero_grad()

        x = x.mean(1)

        loss = criterion(x, period - 1)

        pred[fi:fi + x.shape[0]] = 64 / (np.argmax(x.detach().cpu().numpy(), axis=1) + 1)
        truth[fi:fi + period.shape[0]] = 64 / period.cpu().numpy()
        fi += x.shape[0]

        losses.append(loss.cpu().data)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
        loss.backward()

        optimizer.step()
        #break

        #losses.append(crit.item())

    mae = np.sum(np.absolute(pred - truth) / truth) / total_vids
    #mae = mean_absolute_error(truth, pred)
    gaps = np.absolute(pred - truth)

    obo = 0
    for item in gaps:
        if item > 1:
            obo += 1

    pred = pred.astype(int)
    truth = truth.astype(int)
    print(np.histogram(pred, np.arange(33)))
    print(np.histogram(truth, np.arange(33)))
    print("Epoch", epoch, "MAE:", mae, "OBO:", obo / total_vids)
    print("The loss of epoch", epoch, "was ", np.mean(losses))

    writer.add_scalar('Loss/Train', np.mean(losses), epoch)
    writer.add_scalar('MAE/Train', mae, epoch)
    writer.add_scalar('OBO/Train', obo / total_vids, epoch)

if __name__ == '__main__':
    data_dir='/home/alec/Documents/BigDatasets/CountixAV_trimmed/'
    mode = 1

    train_dataset = Countix(dir=data_dir, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=True, drop_last=False)

    test_dataset = Countix(dir=data_dir, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=True, drop_last=False)

    output_data = pd.DataFrame(columns=['Train Loss', 'Test Loss'])

    if (mode == 1):
        network = Network1()


    n_epochs = 8
    lr=0.001
    criterion = nn.MSELoss()
    optimizer = Adam(network.parameters(), lr=lr)
    output_path = 'network1_%depoch.csv' % n_epochs

    print('Process ID: %d' % (os.getpid()))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = network.to(device)

    for i in range(n_epochs):
        print("EPOCH %d" % (i))
        # Training
        network.train()
        losses = []
        for j, (video, count, class1) in enumerate(train_dataloader):

            print(video.shape)
            print(count.shape)
            print(class1.shape)
            optimizer.zero_grad()
            video = video.to(device)
            outputs = network(video)

            print(outputs.shape)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.cuda.FloatTensor)
            count = count.type(torch.cuda.FloatTensor)
            count = count.to(device)
            loss = criterion(outputs.T[0], count)

            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        train_loss = np.mean(losses)
        print('Train Loss: %.4f' % train_loss)

        network.eval()
        losses = []
        with torch.no_grad():
            for j, (video, count, class1) in enumerate(test_dataloader):
                video = video.to(device)
                count = count.to(device).type(torch.cuda.FloatTensor)
                outputs = network(video).to(device).type(torch.cuda.FloatTensor)
                loss = criterion(outputs.T[0], count)

                losses.append(loss.item())
        
        test_loss = np.mean(losses)
        print('Test Loss: %.4f\n' % test_loss)
       
        output_data = output_data.append({'Train Loss' : train_loss, 'Test Loss' : test_loss}, ignore_index=True)
        output_data.to_csv(output_path, index=False)

