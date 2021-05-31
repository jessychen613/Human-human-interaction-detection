from random import shuffle
import glob
import os
import math
import numpy as np
import cv2 as cv2
import keras
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from video2tfrecordnew import convert_videos
from video2tfrecordnew import save_numpy_to_tfrecords
import tensorflow as tf

shuffle_data = True # shuffle the addresses before saving
train_path = './data/dataset/video/*.avi'
train_path2 = './data/dataset/video1/*.avi'
train_path3 = './data/dataset/video3/*.avi'
train_path4 = './data/dataset/video4/*.avi'
train_test = './data/dataset/videotest/*.avi'
video_path = './data/dataset/video/'
video_path2 = './data/dataset/video1/'
video_path3 = './data/dataset/video3/'
video_path4 = './data/dataset/video4/'
video_pathtest = './data/dataset/videotest/'
datasetpath = './data/dataset/dataset1'
datasetpath4 = './data/dataset/dataset4'
datasettest = './data/dataset/datasettest'

n_videos_in_record=201
color_depth="uint8"
num_classes=2

# read addresses and labels from the 'train' folder
addrs = glob.glob(train_test)
#addrs = glob.glob(train_path2)
#addrs = glob.glob(train_path3)
labels = [1 if 'int' in addr else 0 for addr in addrs] # 0 = non-interaction, 1 = interaction

print("labels", labels)

#data = convert_videos(video_path4, datasettest)
#data = convert_videos(video_path2, datasetpath)
#data = convert_videos(video_path3, datasetpath)
data = convert_videos(video_pathtest, datasettest)

#print(data)
print("data.shape", data.shape)

# to shuffle data
if shuffle_data:
    c = list(zip(data, labels))
#    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 80% train, 20% test
#train_addrs = addrs[0:int(0.8*len(addrs))]
#train_labels = labels[0:int(0.8*len(labels))]

#test_addrs = addrs[int(0.8*len(addrs)):]
#test_labels = labels[int(0.8*len(labels)):]

#print(train_addrs)
#print(train_labels)

#print(test_addrs)
#print(test_labels)

num_video = len(addrs)
print("num_video", num_video)

num_label = len(labels)
print("num_labels", labels)

i=0
#for example in addrs:
#  print(i)
#  print("video sample", example)
#  print("label sample", labels[i])
#  i=i+1

for i, batch in enumerate(addrs):

    if n_videos_in_record > num_video:
      total_batch_number = 1
    else:
      total_batch_number = int(math.ceil(len(filenames) / n_videos_in_record))
    print('Batch ' + str(i + 1) + '/' + str(total_batch_number) + " completed")
#    assert data.size != 0, 'something went wrong during video to numpy conversion'
    print(addrs[i].shape)
 #   labelb=keras.utils.to_categorical(labels[i], num_classes)
 #   print(labelb, num_classes)
    save_numpy_to_tfrecords(addrs[i], labels[i], datasettest, 'batch_',
                           n_videos_in_record, i + 1, num_video,
                           color_depth=color_depth)
