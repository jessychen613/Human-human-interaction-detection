import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.keras import models

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
#testDataPath = './data/dataset/tempData'
testDataPath = './data/dataset/testData'

# output data paths
yoloOutputPath = './data/dataset/yoloOutput'
densenetOutputPath = './data/dataset/densenetOutput'
finalOutputPath = './data/dataset/finalOutput'

# interaction only predicted by yolo
yoloOnlyPath = './data/dataset/yoloOnly'

# interaction only predicted by densenet
densenetOnlyPath = './data/dataset/densenetOnly'

video_suffix = '*.*'
targetClasses = ['person']
targetScore = 0.6

def main(_argv):

    # initialize yolo model
    yolo = initializeYolo(FLAGS.num_classes, FLAGS.weights, FLAGS.tiny)
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]  
    times = []

    # initialize densenet model
    densenet = models.load_model("densenet.h5", compile = True)
    densenet.summary()
    densenet.load_weights('ckpt.h5')

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
            
            # step 1: Yolo pre-prediction
            # prediect human with bounding boxes using Yolo3 model
            boxes, scores, classes, nums = yoloPredictFromImage(yolo, img, FLAGS.size)     
 
            # find potential interaction with AABB detection algorithm
            found1, intersections = findOverlappingBoxes((boxes, scores, classes, nums), class_names, targetClasses, targetScore)

            if found1:                  
                print ("Interaction founded by Yolo preprediction!")
                # output images with human interaction 
                outputImagePath = os.path.join(yoloOutputPath,os.path.basename(video)) + "_" + str(frame)
                print("Write interaction frame: ", outputImagePath)
                outputImage(outputImagePath, img)          
            else:
                # skip densenet if yolo cannot find any overlapping boundingboxes 
                continue

            # step 2: Final prediction
            # prediect human interaction with densenet model
            predict = densenetPredictFromImage(densenet, img, (192, 256))
            #print("predict2",predict)
            found2 = np.argmax(predict, axis = 1)           

            if found2:                  
                print ("Interaction founded by Densenet prediction!")
                # output images with human interaction 
                outputImagePath = os.path.join(densenetOutputPath,os.path.basename(video)) + "_" + str(frame)
                print("Write interaction frame: ", outputImagePath)
                outputImage(outputImagePath, img)           

            # draw diff data for comparison
            if found1 and not found2:
                outputImagePath = os.path.join(yoloOnlyPath,os.path.basename(video)) + "_" + str(frame)                
                outputImage(outputImagePath, img)
            
            #if found2 and not found1:
            #    outputImagePath = os.path.join(densenetOnlyPath,os.path.basename(video)) + "_" + str(frame)                
            #    outputImage(outputImagePath, img)

            # draw final result agreed by both steps
            if found1 and found2:
                outputImagePath = os.path.join(finalOutputPath,os.path.basename(video)) + "_" + str(frame)                
                outputImage(outputImagePath, img)
           
            # draw prediction info on each frame at runtime            
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names, targetClasses, targetScore)
            img = draw_interaction(img, intersections)            
            cv2.imshow('output', img)
         
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
