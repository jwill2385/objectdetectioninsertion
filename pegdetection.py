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

#This function takes in canny image and location of peg/hole and isolates region
def img_mask(canny_image, left, top, right, bottom):
    #Create empty image of same size
    mask = np.zeros_like(canny_image)
    #Fill in bounding box region
    cv2.rectangle(mask, (left, top), (right, bottom), (255,255,255), -1)
    # combine mask with canny image
    masked_img = cv2.bitwise_and(canny_image, mask)
    return masked_img


#This function generates hough_transform_lines
def hough_transform(img, lines, precise):
    new_img = np.copy(img)
    empty_img = np.zeros((new_img.shape[0], new_img.shape[1], 3), dtype=np.uint8)
   
    if not precise:

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho

            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 100*(-b)), int(y0 - 1000 * a))
            cv2.line(empty_img, pt1, pt2, (255, 0, 255), 3)
        #new_img = cv2.addWeighted(new_img, 0.8, empty_img, 1, 0, 0)
    else:
        #Precise is true so we used houghTransformP
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(empty_img, (x1,y1), (x2, y2), (255, 0, 255), 3)
    return empty_img

# This function does canny edge detection on image
def canny_convert(img):
  
    #convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    #blur img
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #Abstract edges
    canny = cv2.Canny(blur, 10, 80, apertureSize = 3)
    #canny = cv2.Canny(blur, 20, 70, apertureSize = 3)
    return canny

def postProcess(frame, outputs):
    #Will remove bounding box from obj with low confidence
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    #define arrays we need for detection
    classIDs = []
    confidences = []
    boxes = []
    #Image Processing for canny
    canny_img = canny_convert(frame)
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
        frame = drawPred(classIDs[i], confidences[i], left, top, left + width, top + height, canny_img, frame) #Update image frame of interest with bbox and hough lines
    return frame

# This function will draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, canny_image, frame):
    #Draw box
    classId = classId % 2
    color = [int(c) for c in colors[classId]]

    #Add hough transform logic before adding rectange
    #Step 1 create a image mask isloating region of interest
    mask = img_mask(canny_image, left, top, right, bottom)

    #Step 2 generate hough lines
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, 90, minLineLength = 10, maxLineGap = 30)
    #Step 3 perform hough transform
    if lines is not None:
        hough_lines = hough_transform(mask, lines, True)
   
    # Now add bounding box around actual image
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    label = '%.2f' % conf
    # get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left + 5, top + 2), 0, 0.5, (0, 0, 0), 2)
    #Todo merge hough image with color frame
    if lines is not None:
        output = cv2.addWeighted(frame, 0.8, hough_lines, 1, 0, 0)
        return output
    return frame


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
    #cv2.imshow('blob', r)
    #text = f'Blob shape={blob.shape}'
    #cv2.displayOverlay('blob', text)

    net.setInput(blob)  
    t0 = time.time()
    
    outputs = net.forward(outputLayer)
    #outputs = net.forward()
    t = time.time() - t0

 
    #now time to post process
    result_img = postProcess(color_frame, outputs)

    # Make sure to show image at very end so that it shows rectangles
    cv2.imshow("Depth_frame", depth_frame)
    cv2.imshow("Color_frame", result_img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows