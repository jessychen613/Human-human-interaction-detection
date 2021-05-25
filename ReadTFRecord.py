
import tensorflow as tf 
from IPython.display import Image, display
import numpy as np
from random import shuffle
from tensorflow.python.platform import gfile
import os

filepath = './data/dataset/dataset1/'
datasetpath1 = './data/dataset/dataset/batch_1_of_2.tfrecords'
file_suffix = '*.tfrecords'

HEIGHT = 240
WIDTH = 320
CHANNEL = 3

#def read_and_decode(filename_queue):

#    video = tf.decode_raw(features['video'], tf.uint8)
#    video = tf.reshape(video, [features['num_images'],HEIGHT, WIDTH, CHANNEL])
#    video = tf.cast(video, tf.float32)
#    label = tf.cast(features['label'], tf.int64)
#    return video, label

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

image_feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channel': tf.io.FixedLenFeature([], tf.int64),
        'num_images': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
}

def load_dataset(filepath):

    filenames = gfile.Glob(os.path.join(filepath, file_suffix))
    filenum = len(filenames)

    print('Total videos found: ' + str(filenum))
    print(filenames)
    print(filenames[0])

    num_frame = 0
    video_train = []
 #  video_test = np.zeros((124,240,320,3))
    label_train = []

    for file_i in range(filenum):
        print("file loop",file_i)
        filepath = filenames[file_i]
        print("filepath",filepath)

        raw_image_dataset = tf.data.TFRecordDataset(filepath)
        print("before parsed_image_dataset")
        parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

        print("after parsed_image_dataset")

        i=num_frame
        for image_features in parsed_image_dataset:
            num_images = image_features['num_images'].numpy()
            height = image_features['height'].numpy()
            width = image_features['width'].numpy()
            channel = image_features['channel'].numpy()
#           print(num_images,height,width,channel)
            video = tf.io.decode_raw(image_features['video'], tf.uint8)
            video = tf.reshape(video, [HEIGHT, WIDTH, CHANNEL])
#            print("before video_train.append")
#            tf.image.convert_image_dtype(video, dtype=tf.float32)
            video_train.append(tf.cast(video, tf.float32) / 255.0)
#            print("after video_train.append")
#           display(Image(data=video))
            label = image_features['label'].numpy()
#            if file_i==1:
#                print(video)
#                print(label)
            label_train.append(tf.cast(label, tf.float32))
#           label_train[i] = label
            i=i+1
#            print(label_train)

        num_frame = i
        file_i=file_i+1
 #   video_traint = tf.stack(video_train)
 #   label_traint = tf.stack(label_train)

    video_trainl = np.array(video_train)
    label_trainl = np.array(label_train)

    print("video_train shape",video_trainl.shape)
#    print("video_train",video_trainl)
#    print("label_train",label_trainl)
    print("label_train shape",label_trainl.shape)

    c = list(zip(video_trainl, label_trainl))
    shuffle(c)
    addrs, labels = zip(*c)

    # Divide the hata into 80% train, 20% test
    train_addrs = np.array(addrs[0:int(0.8*len(addrs))])
    train_labels = np.array(labels[0:int(0.8*len(labels))])

    test_addrs = np.array(addrs[int(0.8*len(addrs)):])
    test_labels = np.array(labels[int(0.8*len(labels)):])

    return train_addrs, train_labels, test_addrs, test_labels

#video_train, label_train, video_test, label_test = load_dataset(filepath)
#print("video_train shape",video_train.shape)
#print("video_train",video_train)
#print("label_train shape",label_train.shape)
#print("label_train",label_train)
#print("video_test shape",video_test.shape)
#print("video_test",video_test)
#print("label_test shape",label_test.shape)
#print("label_test",label_test)