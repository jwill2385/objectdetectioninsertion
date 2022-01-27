import cv2
import os
import pyrealsense2
import numpy as np
import time
from realsense_depth.realsense_depth import *

# Initialize Camera Intel Realsense
dc = DepthCamera() #Class from realsense depth

# Create a mouse event
cv2.namedWindow("Color_frame")

confThreshold = 0.4 #percentage confident of the object detection
nmsThreshold = 0.4 # Non Maximum suppression threshold

#Get location of weights and CFG file for Yolo
#weight_path = '/home/cvdarbeloff/Documents/Realsense/darknet/yolov3.weights'
#cfg_path = '/home/cvdarbeloff/Documents/Realsense/darknet/cfg/yolov3.cfg'

weight_path = '/home/cvdarbeloff/Documents/Realsense/darknet/custom-yolov4-detector_best.weights'
cfg_path = '/home/cvdarbeloff/Documents/Realsense/darknet/custom-yolov4-detector.cfg'

# load in classes file
#classFile = "/home/cvdarbeloff/Documents/Realsense/darknet/coco.names"
classFile = "/home/cvdarbeloff/Documents/Realsense/darknet/custom_data/custom.names"
classes = []
with open(classFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def postProcess(frame, outputs):
    #Will remove bounding box from obj with low confidence
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    #define arrays we need for detection
    classIDs = []
    confidences = []
    boxes = []
    #scan through every single bounding box and keep ones with highest confidence scores
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            #classId = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
            confidence = scores[classId]
            if confidence > confThreshold:
                #box = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                #(center_x, center_y, width, height) = box.astype("int")
                #center_x = int(detection[0] * frameWidth)
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIDs.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    
    # Perform a non maximum suppression to eliminate redundant overlapping boxes with low confidence
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        #print(i)
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

# This function will draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    #Draw box
    #print(classId)
    classId = classId % 2
    color = [int(c) for c in colors[classId]]
    cv2.rectangle(color_frame, (left, top), (right, bottom), color, 2)
    
    label = '%.2f' % conf
    # get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(color_frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(color_frame, label, (left + 5, top + 2), 0, 0.5, (0, 0, 0), 2)




# Set up colors
np.random.seed(42) #this lets me get same set of random colors everytime
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

#Begin processing with camera
while True:

    ret, depth_frame, color_frame = dc.get_frame()

    #
    #distance = depth_frame[point[1], point[0]]
    #cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,  (255, 0, 0), 2)
    #print(distance)
    
    
    #print(classes)    
    # set up Yolo network
    #since i'm using yolov5 i need to read from ONNX
    #net = cv2.dnn.readNetFromONNX("/home/cvdarbeloff/Documents/Realsense/yolov5/yoloV5insertweights.onnx")

    net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
    # USE CPU
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    #Use GPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
   

    # determine the output layer
    ln = net.getLayerNames()
    #print(len(ln))
    outputLayer = []
    for i in net.getUnconnectedOutLayers():
        outputLayer.append(ln[i - 1])
    
    #print(len(outputLayer))

    # construct a blob from the image
    #blob = cv2.dnn.blobFromImage(color_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(color_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    #Display blob in new window . Not needed
    cv2.imshow('blob', r)
    text = f'Blob shape={blob.shape}'
    cv2.displayOverlay('blob', text)

    net.setInput(blob)  
    t0 = time.time()
    
    outputs = net.forward(outputLayer)
    #outputs = net.forward()
    t = time.time() - t0

 
    #now time to post process
    postProcess(color_frame, outputs)

    # Make sure to show image at very end so that it shows rectangles
    cv2.imshow("Depth_frame", depth_frame)
    cv2.imshow("Color_frame", color_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows