import tensorflow as tf 
import numpy as np
import os
import cv2

from yolov3.yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3.yolov3_tf2.dataset import transform_images

import matplotlib.pyplot as plt


def initializeYolo(num_classes, weights, tiny):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights)

    return yolo


def yoloPredictFromImage(yolo, img, size):
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, size)        
    
    boxes, scores, classes, nums = yolo.predict(img_in)        
    
    return boxes[0], scores[0], classes[0], nums[0]     


def densenetPredictFromImage(densenet, img, size):     
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           
    img_in = tf.image.resize(img_in, size)          
    img_in = img_in / 255
    #plt.imshow(img_in)
    #plt.show()

    img_in = tf.expand_dims(img_in, 0)
    #print("img", img_in)
    #print("img.shape", img_in.shape)          
    predict = densenet.predict(img_in)
                
    return predict

def outputImage(imageName, cv_img, fileExtension=".jpg"):
    imageName = imageName + fileExtension
    #print("Frame output: ", imageName)
    cv2.imwrite(imageName, cv_img) 



# draw overlay info on top of a video
def draw_outputs(img, 
                 outputs, 
                 class_names, 
                 targetClasses, 
                 targetScore):
    
    boxes, objectness, classes, nums = outputs
    
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        className = class_names[int(classes[i])]
        score = objectness[i]
        if className in targetClasses and score >= targetScore:
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            #print ("classes[i]", classes[i])
            #print ("objectness[i]", objectness[i])
            #print ("boxes[i]", boxes[i])
            #img = cv2.putText(img, '{} {:.4f}'.format(
            #   className, score),
            #    x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


# draw interaction info on top of a video
def draw_interaction(img, 
                     interactions):    
    
    nums = len(interactions)
    #print ("interactions", interactions)  
    #print ("nums", nums)  
    #print ("img.shape", img.shape)
    wh = np.flip(img.shape[0:2])
    for i in range(nums):  
        x1y1 = tuple((np.array(interactions[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(interactions[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), -2)
        img = cv2.putText(img, 'interaction',
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def findOverlappingBoxes(outputs, 
                         class_names, 
                         targetClasses, 
                         targetScore, 
                         earlyTermination=False):
    
    boxes, objectness, classes, nums = outputs    
    found = False;
    outputBoxes = []

    for i in range(nums):        
        #only count the box if it's a human 
        for j in range (i+1, nums):  
            #print ("class_names[int(objectness[i] )]", objectness[i] )       
            #print ("class_names[int(objectness[j] )]", objectness[j] )       
            if class_names[int(classes[i])] in targetClasses and class_names[int(classes[j])] in targetClasses and objectness[i] >= targetScore and  objectness[j] >= targetScore:
                #print("check box ", i, " and ", j)
                intersect, box = checkIntersection(boxes[i], boxes[j])
                if intersect:
                    #print ("find interaction at: ", box)
                    found = True
                    outputBoxes.append(box)
                    if earlyTermination:
                        return found, outputBoxes
   
    return found, outputBoxes


def checkIntersection(boxA, boxB):
    
    intersect = True
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1]) 
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    if x2 < x or y2 < y:
        intersect = False

    return(intersect, [x, y, x2, y2])