import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import os.path
from ReadTFRecord import load_dataset
from ReadTFRecord import load_datasetforpredict
from keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input
# Using ImageNet pre_trained weights to predict image's class(1000 class)
# ImageNet -- http://www.image-net.org/
# make sure your package pillow is the latest version 

datasetpath1 = './data/dataset/predictset'

model = models.load_model("densenet.h5", compile = True)

model.summary()

model.load_weights('ckpt.h5')

#data, labels, testvideo, testlabel = load_dataset(datasetpath1)
data, labels = load_datasetforpredict(datasetpath1)
print("label", labels)
#print("data", data)

samples_to_predict = []

#test = np.zeros((1, 240,320,3))

#predict1 = model.predict(test)
#print("predict",predict1)
# Generate arg maxes for predictions

#classes = np.argmax(predict1, axis = 1)
#print("classes", classes)

#predict2 = model.predict(np.array(data))
predictclass = []
zeronum = 0
onenum = 0

for example in data:

#    print("example", example)
#    print("example.shape", example.shape)
#    plt.imshow(example)
#    plt.show()
#    samples_to_predict = np.array(tf.cast(example, tf.float32) / 255.0)
    samples_to_predict = np.array(example)
    samples_to_predict = np.expand_dims(samples_to_predict, axis=0)
#    print("samples_to_predict.shape", samples_to_predict.shape)
#    predictdata.append(samples_to_predict)
#    print("samples_to_predict", samples_to_predict)
    predict2 = model.predict(samples_to_predict)
#    print("predict2",predict2)
    classes = np.argmax(predict2, axis = 1)
#    print("classes",classes)

    if (classes[0]==0):
        zeronum = zeronum + 1
    if (classes[0]==1):
        onenum = onenum + 1
    predictclass.append(classes[0])

#    print("predict2",predict2)

print("predictclass", predictclass)
print("zeronum", zeronum)
print("onenum", onenum)
