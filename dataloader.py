import pathlib
from re import L
from timeit import repeat
from turtle import speed
import torch.utils.data
import os
import numpy as np
import cv2
import collections
import skimage.draw
import math
import csv
from os.path import exists
import random

import torch.nn.functional as F
from os.path import exists

from transforms import get_transform

import time


from einops import rearrange
def repeat_video(frames, count):

    original_frames = frames.shape[0]
    i=0
    while frames.shape[0] < 64:
        frames = np.concatenate((frames, frames), axis=0)
    new_count = count * (frames.shape[0] / original_frames)

    #print("output is going to be", frames.shape)
    return frames, new_count

def speedup(frames, speedup):
    new_frames = []

    i = 0
    while i < frames.shape[0]:
        #print(frames[i].shape)
        new_frames.append(frames[i])
        i += speedup

    #print("New frames here", new_frames)
    return np.stack(new_frames)

def read_video(video_filename):
  """Read video from file."""
  #print(video_filename)
  cap = cv2.VideoCapture(video_filename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = []
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      frames.append(frame_rgb)
  frames = np.asarray(frames)
  #print("Here is the frames coming out", frames.shape, "And the video was", video_filename)

  return frames, fps


class Countix(torch.utils.data.Dataset):

    def __init__(self,split='train', dir='/home/alec/Documents/BigDatasets/CountixAV_trimmed', prefix='CountixAV', inference=False):

        self.inference = inference
        name_list = []
        count_list = []
        start_list = []
        start_crop_list = []
        end_list = []
        class_list = []
        end_crop_list = []
        self.split = split
        class_dict = {'battle rope training':0, 'bouncing ball (not juggling)':1, 'bouncing on trampoline':2, 'clapping':3, 'gymnastics tumbling':4, 'juggling soccer ball':5, 'jumping jacks':6,
                      'mountain climber (exercise)':7, 'planing wood':8, 'playing ping pong':9, 'playing tennis':10, 'running on treadmill': 11, 'sawing wood':12, 'skipping rope':13,
                      'slicing onion':14, 'swimming':15, 'tapping pen':16, 'using a wrench':17, 'using a sledge hammer':18,'bench pressing':19, 'bouncing on bouncy castle':20, 'crawling baby':21,
                      'doing aerobics':22, 'exercising arm': 23, 'front raises':24, 'hammer throw':25, 'headbanging':26, 'hula hooping':27, 'lunge':28, 'pirouetting':29, 'playing ukulele':30, 'pull ups':31,
                      'pumping fist':32, 'push up':33, 'rope pushdown':34, 'shaking head':35, 'shoot dance':36, 'situp':37, 'skiing slalom':38, 'spinning poi':39,'squat':40, 'swinging on something':41, 'else': 42
                      }

        #name_list[name_id], start_list[name_id], end_list[name_id], start_crop_list[name_id], end_crop_list[name_id],count_list[name_id]chiseling wood
        neg_list = []
        self.dir = dir
        with open(self.dir+ '/' + prefix+"_"+split+"_final.csv") as f:
            f_csv = csv.reader(f)
            next(f_csv)
            for i, row in enumerate(f_csv):
                #print(row)
                #print(self.dir + '/' + self.split+"_video/"+row[0]+".mp4", flush=True)
                if not exists(self.dir + '/' + self.split+"_video/"+row[0]+".mp4"):
                    continue
                if int(row[6]) > 32:
                    continue
                name_list.append(row[0])
                if row[1].startswith("swimming"):
                    class_list.append(15)
                try:
                    class_list.append(class_dict[row[1]])
                except:
                    class_list.append(class_dict['else'])
                start_list.append(float(row[2]))
                end_list.append(float(row[3]))
                start_crop_list.append(float(row[4]))
                end_crop_list.append(float(row[5]))
                count_list.append(float(row[6]))

        self.name_list = name_list
        self.count_list = count_list
        self.start_list = start_list
        self.end_list = end_list
        self.start_crop_list = start_crop_list
        self.end_crop_list = end_crop_list
        self.split = split
        self.class_list = class_list


        self.crop_size = 112
        #if split == 'train':
        #    self.is_validation = False
        #else:
        #    self.is_validation = True
        self.is_validation = False


        self.transform = get_transform(self.is_validation, self.crop_size, input_type='rgb')

        self.clip_len = 64
        self.n_clips = 1

    def __getitem__(self, idx):
        #print("Line 132", flush=True) 

        video1, fps = read_video(self.dir + '/' + self.split+"_video/"+self.name_list[idx]+".mp4")
        #print(video1.shape)
        #print("Here is the actual thing")
        #print(video1.shape)
        #print("Line 136", flush=True)
        count = self.count_list[idx]
            
        if len(video1) == 0:
            print('Error found in video loading!')
            buffer = np.random.rand(3, self.clip_len, self.crop_size, self.crop_size).astype('float32')
            buffer = torch.from_numpy(buffer)
            count = np.zeros(64)
            count.fill(np.round(64 / 1))
            return buffer, 42, count, 1

        
        ## USELESS OLD STUFF ##

        #print("old count", count)
        # Keep speeding up the video until the periodicity is under 32
        #print("Line 149", flush=True)
        #while video1.shape[0] / count >= 32.0:
        #    video1 = speedup(video1)
        # Repeat the video until there are at least 64 frames

        #print("Line 154", flush=True)
        #video1, count = repeat_video(video1, count)

        #print("Line 157", flush=True)
        true_count = count 
      
        # Create the count array for the entire vidoe
        #print("Line 161", flush=True)

        # Take the first 64 frames
        # TODO: Take different sample rates of the video
        #print("Line 167", flush=True)




        original_frames = video1.shape[0]
        original_period = original_frames / count
        speed = min(4, random.randint(1, (original_frames // 64) + 1))
        if self.split != 'train':
            speed = 1
        while (original_period / speed) >= 32:
            speed += 1 
        new_period = round(original_period / speed)

        #print("The original period will be", original_period)
        #print("The new period will be", new_period)

        #print("The speed will be", speed)
        video1 = speedup(video1, speed)

        video1, count = repeat_video(video1, count)
        #print(video1.shape)
        buffer = video1[:64]




        #print("Count is", count.shape)
        s = buffer.shape
        
        #buffer = buffer.reshape(s[0], s[1], s[3], s[4])
        #buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        #print("Line 180", flush=True)
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer)
        #print(buffer.shape)
        #buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)

        #print("Line 186", flush=True)
        class1 = self.class_list[idx]

        #rint("Line 189", flush=True)
        #print("Buffer is", buffer.shape)
        return buffer, new_period, true_count

    def __len__(self):
        return len(self.name_list)

    @staticmethod
    def clean_data(fnames, labels):
        if not isinstance(fnames[0], str):
            print('Cannot check for broken videos')
            return fnames, labels
        broken_videos_file = 'assets/kinetics_broken_videos.txt'
        if not os.path.exists(broken_videos_file):
            print('Broken video list does not exists')
            return fnames, labels

        t = time()
        with open(broken_videos_file, 'r') as f:
            broken_samples = [r[:-1] for r in f.readlines()]
        data = [x[75:] for x in fnames]
        keep_sample = np.in1d(data, broken_samples) == False
        fnames = np.array(fnames)[keep_sample]
        labels = np.array(labels)[keep_sample]
        print('Broken videos %.2f%% - removing took %.2f' % (100 * (1.0 - keep_sample.mean()), time() - t))
        return fnames, labels

import sys
if __name__ == "__main__":

    train_dataset = Countix(split='train', dir='/home/alec/Documents/BigDatasets/Countix', prefix='countix')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    np.set_printoptions(threshold=sys.maxsize)
    j=0
    
    for i, (video, count, classlabels, truecount) in enumerate(train_dataloader):
        video = video
        #print(video.shape)
