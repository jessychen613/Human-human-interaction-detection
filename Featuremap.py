# TensorFlow and TF-Hub modules.
from absl import logging


import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow_hub as hub
import matplotlib.pyplot as plt
#from tensorflow_docs.vis import embed

# Some modules to help with reading the UCF101 dataset.
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np

# Some modules to display an animation using imageio.
import imageio
from IPython import display

from urllib import request
import tensorflow_datasets as tfds

import keras
import math
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D, Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from ReadTFRecord import load_dataset
from datetime import datetime
import io
from tensorflow.keras import backend as K

#Adjust OS memory allocation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#print(tfds.list_builders())
filepath = './data/dataset/predictset/'

video_train, label_train, video_test, label_test = load_dataset(filepath)

growth_rate        = 12 
depth              = 100
compression        = 0.5

#img_rows, img_cols = 32, 32
#img_rows, img_cols = 256, 256
img_rows, img_cols = 192, 256
#img_rows, img_cols = 240, 280
img_channels       = 3
#num_classes        = 10
#num_classes        = 104
num_classes        = 2
#batch_size         = 64         # 64 or 32 or other
batch_size         = 2         # 64 or 32 or other
#epochs             = 300
epochs             = 1
#iterations         = 782    
iterations         = 2      
weight_decay       = 1e-4

mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

#from keras import backend as K
#if('tensorflow' == K.backend()):
#    import tensorflow.compat.v1 as tf
#    tf.disable_v2_behavior()
#    from keras.backend.tensorflow_backend import set_session
#    from tensorflow.python.keras.backend import set_session
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    sess = tf.Session(config=config)

def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001

def densenet(img_input,classes_num):
    def conv(x, out_filters, k_size):
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    def dense_layer(x):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1,1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
        return x

    def single(x):
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1,1))
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x,blocks,nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels


    nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2


    x = conv(img_input, nchannels, (3,3))
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

#reference from https://segmentfault.com/a/1190000022507255
def image_grid(images, labels):

    # Create a figure to contain the plot.
    images_num = len(images)
    col_num = 5
    row_num = math.ceil(images_num / col_num)

    fig_width = col_num * 2
    fig_height = row_num * 2
    figure = plt.figure(figsize=(fig_width, fig_height))
    
    for i in range(images_num):
        # Start next subplot.
        plt.subplot(row_num, col_num, i + 1, title = labels[i])#每行5个
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap = plt.cm.binary)

    # plt.ion();
    # plt.show();
    # plt.pause(0)
    figure.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def saveTrainInputImageToTf(train_images, train_labels):
    # Prepare the plot
    figure = image_grid(train_images, train_labels)
    # plot_to_image(figure)
    logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    # # Convert to image and log
    with file_writer.as_default():
        tf.summary.image("Training data", plot_to_image(figure), step=0)


def saveTrainKernalToTf(model):
    weights = model.weights
    for i in range(len(model.weights)):

        if len(model.weights[i].shape) != 4:
            continue

        inputs_len = model.weights[i].shape[2]
        outputs_len = model.weights[i].shape[3]

        kernel_array = []
        kernal_label_array = []
        for j in range(inputs_len):
            for k in range(outputs_len):
                kernel = weights[i][:, :, j, k]
                kernel_array.append(kernel)
                kernal_label_array.append(str(j) + "_" + str(k)+ "_kernal")

        kernel_array = np.array(kernel_array)

        print("Kernal Shape: ", kernel_array.shape)

        figure = image_grid(kernel_array, kernal_label_array)
        # plot_to_image(figure)
        logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(logdir)
        # # Convert to image and log
        with file_writer.as_default():
            tf.summary.image("Training kernal", plot_to_image(figure), step=0)


def saveConvImage(model, train_images, train_labels):
    layernum=0
    for layer in model.layers:
        print("layer no",layernum)
        if 'conv' in layer.name and ((layernum%30==0)or(layernum<10)or(layernum>330)):
            print(layer.name)
            sampleimage = train_images[0]
            print("sampleimage.shape",sampleimage.shape)
            sampleimage = np.expand_dims(sampleimage, axis=0)

            activations_f = K.function([model.layers[0].input], [layer.output])
            print("activations_f", activations_f)

 #           activations = activations_f((sampleimage, False))
            activations = activations_f(sampleimage)

#            print("activations",activations)
            activations = np.array(activations)
            print(activations.shape)

            conv_layer = []
            conv_layer_label = []
            for sample_index in range(activations.shape[1]):
                featurenum = 0
                for i in range(activations.shape[4]):
                    #print("i",i)
                    # activations[0, sample_index, :, :, i] 
                    conv_layer.append(activations[0, sample_index, :, :, i])
                    conv_layer_label.append(train_labels[sample_index])
                    featurenum = i
                print("featurenum",featurenum)

            conv_layer = np.array(conv_layer)
            print(conv_layer.shape)
            figure = image_grid(conv_layer, conv_layer_label)
            # plot_to_image(figure)
            logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            file_writer = tf.summary.create_file_writer(logdir)
            # # Convert to image and log
            with file_writer.as_default():
                tf.summary.image(layer.name, plot_to_image(figure), step=layernum)
        layernum =layernum+1



#if __name__ == '__main__':

    # load data
x_train1 = video_train
x_test1 = video_test
y_train1 = label_train
y_test1 = label_test
print("x_train1 shape", x_train1.shape)
print("y_train1 shape", y_train1.shape)
print("x_test1 shape", x_test1.shape)
print("y_test1 shape", y_test1.shape)

#for i in range(10):  
#    plt.imshow(x_train1[i])
#    plt.show()
#    print(y_train1[i])

#print("num_classes", num_classes)
y_train1 = keras.utils.to_categorical(y_train1, num_classes)
y_test1  = keras.utils.to_categorical(y_test1, num_classes)
#x_train1 = x_train1.astype('float32')
#x_test1  = x_test1.astype('float32')

print("y_train1 shape after to_categorical", y_train1.shape)
print("y_test1 shape after to_categorical", y_test1.shape)

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#print("y_train shape", y_train.shape)
#print("x_train[0]", x_train[0])
#print("y_train[0]", y_train[0])
#print("y_train", y_train)

#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test  = keras.utils.to_categorical(y_test, num_classes)
#x_train = x_train.astype('float32')
#x_test  = x_test.astype('float32')

#print("x_train shape", x_train.shape)
#print("y_train shape", y_train.shape)
    
    # - mean / std
#for i in range(3):
#    x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
#    x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

#print("x_train[0] after processing", x_train[0])

# build network
img_input = Input(shape=(img_rows,img_cols,img_channels))
output    = densenet(img_input,num_classes)
model     = Model(img_input, output)

print("img_input.shape",img_input.shape)

model.load_weights('ckpt.h5')

print(model.summary())

    # from keras.utils import plot_model
    # plot_model(model, show_shapes=True, to_file='model.png')

    # set optimizer
#sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # set callback
tb_cb     = TensorBoard(log_dir='./densenet/', histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
ckpt      = ModelCheckpoint('./ckpt.h5', save_best_only=False, mode='auto', period=10)
cbks      = [change_lr,tb_cb,ckpt]

    # set data augmentation
print('Using real-time data augmentation.')
datagen   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(x_train1)

# start training
#model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_test, y_test))
model.fit_generator(datagen.flow(x_train1, y_train1,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_test1, y_test1))
model.save('densenet.h5')

# 绘制训练完成后的卷积核
#saveTrainKernalToTf(model)
# 显示经过卷积层的 feature map
saveConvImage(model, x_train1, y_train1)