import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from yolov3.yolov3_tf2.models import YoloV3, YoloV3Tiny

from Utils import *

flags.DEFINE_string('classes', './yolov3/data/full.names', 'path to classes file')
flags.DEFINE_string('weights', './yolov3/checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')

flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# test data paths
#testDataPath = './data/dataset/testData'
testDataPath = './data/dataset/tempData'
outputDataPath = './data/dataset/outputData'
video_suffix = '*.*'
targetClasses = ['person']
targetScore = 0.8

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    # load all videos from the test data set path
    print('######################################################################')
    print('Test Dataset: ', os.path.join(testDataPath, video_suffix))
    filenames = gfile.Glob(os.path.join(testDataPath, video_suffix))    
    filenum = len(filenames)
    print('Total videos found: ' + str(filenum))
    print('######################################################################')


    for video in filenames:
        print('######################################################################')
        print('Processing video : ' + str(video))

        try:
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        frame = 0

        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        
        while True:
            _, img = vid.read()

            frame += 1
            print("\nframe: ", frame)
            if img is None:
                logging.warning("End of video")
                break   
            
            # prediect human with bounding boxes using Yolo3 model
            boxes, scores, classes, nums = yoloPredictFromImage(yolo, img, FLAGS.size)     
 
            # find potential interaction with AABB detection algorithm
            found, intersections = findOverlappingBoxes((boxes, scores, classes, nums), class_names, targetClasses, targetScore)

            # draw prediction info on each frame at runtime            
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names, targetClasses, targetScore)
            img = draw_interaction(img, intersections)            
            cv2.imshow('output', img)

            if found:  
                # output images with human interaction 
                outputImagePath = os.path.join(outputDataPath,os.path.basename(video)) + "_" + str(frame)
                print("Write interaction frame: ", outputImagePath)
                outputImage(outputImagePath, img)

            if FLAGS.output:
                out.write(img)
            
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
