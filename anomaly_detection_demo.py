import cv2 as cv
import numpy as np
import math

from turtle import xcor
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import os

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time

inpWidth = 416
inpHeight = 416
#connect to  robot IP 
# rtde_c = RTDEControl("192.168.1.102")
# rtde_r = RTDEReceive("192.168.1.102")
# init_q = rtde_r.getActualQ()
# print(init_q)


# load class names
classesFile = './anomaly_files/obj.names'
classes = open(classesFile, 'r').read().splitlines()

# give configuration, weight files, and autoenconder path location
config = './anomaly_files/yolo-obj.cfg'
weights = './anomaly_files/yolo-obj_best.weights'
vae_path = './anomaly_files/vae.pth'

# generate colors for bounding boxes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


# weights = './yolo-obj_last.weights'

# load YOLO network model
net = cv.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# get names of output layers (required for forward pass; need to know until when it should run)
layer_names = net.getLayerNames()                                                   # all layers in network
output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]    # layers with unconnected outputs


def detect_objects(frame): 
    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (inpWidth, inpHeight), swapRB=True, crop=False)

    # set input to network
    net.setInput(blob)

    # run forward pass of the NN to get output of output layers
    outputs = net.forward(output_layer_names)

    # post process the images
    boxes = [] 
    confidences = []
    class_ids = [] 
    frame_height, frame_width = frame.shape[:2]

    confidence_threshold = 0.55  # confidence threshold for final image
    nms_threshold = 0.4         # non-maximum supprssion threshold

    # remove bounding boxes with low confidence
    for output in np.vstack(outputs): 
        class_confidences = output[5:]  # confidences associated with each class
        class_id = np.argmax(class_confidences) # class with highest confidence
        confidence = class_confidences[class_id]
        if confidence > confidence_threshold: # width < 600 
            box = output[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
            center_x, center_y, w, h = box.astype("int")
            left_x = int(center_x - w / 2)
            top_y = int(center_y - h / 2)
            box = [left_x, top_y, int(w), int(h)]
            boxes.append(box)
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)  # indices of resulting boxes

    hole_boxes = []

    # draw the boxes
    for i in indices:
        x, y, w, h = boxes[i][:4]
        color = [int(c) for c in colors[class_ids[i]]]
        # if classes[class_ids[i]] != "hole" or (classes[class_ids[i]] == "hole" ):
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)   # bounding box
        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}" # label with confidence (rounded to 2 dec pt)
        cv.rectangle(frame, (max(0, x), max(0, y - 18)), (max(0, x) + 75, max(0, y - 18) + 18), (255, 255, 255), -1)
        cv.putText(frame, text, (max(0, x), max(0, y - 18) + 13), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if classes[class_ids[i]] == "hole" : 
            hole_boxes.append((x, y, w, h)) #We want to isolate hole boxes for post processing

    return hole_boxes

# Class Modules for Auto Encoder

class Encoder(nn.Module): 
  def __init__(self, latent_dims): 
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    self.batch2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  
    self.linear1 = nn.Linear(27*27*32, 128)
    self.linear2 = nn.Linear(128, latent_dims)
    self.linear3 = nn.Linear(128, latent_dims)

    self.N = torch.distributions.Normal(0, 1)
    # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
    # self.N.scale = self.N.scale.cuda()
    self.kl = 0
  
  def forward(self, x):
    x = x.to(device)
    x = F.relu(self.conv1(x))
    x = F.relu(self.batch2(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.linear1(x))
    mu =  self.linear2(x)
    sigma = torch.exp(self.linear3(x))
    z = mu + sigma*self.N.sample(mu.shape)
    self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return z 


class Decoder(nn.Module):
    
  def __init__(self, latent_dims):
    super().__init__()

    self.decoder_lin = nn.Sequential(
      nn.Linear(latent_dims, 128),
      nn.ReLU(True),
      nn.Linear(128, 27 * 27 * 32),
      nn.ReLU(True)
    )

    self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 27, 27))

    self.decoder_conv = nn.Sequential(
      nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(8),
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
    )
      
  def forward(self, x):
    x = self.decoder_lin(x)
    x = self.unflatten(x)
    x = self.decoder_conv(x)
    x = torch.sigmoid(x)
    return x


class VariationalAutoencoder(nn.Module):
  def __init__(self, latent_dims):
    super(VariationalAutoencoder, self).__init__()
    self.encoder = Encoder(latent_dims)
    self.decoder = Decoder(latent_dims)

  def forward(self, x):
    x = x.to(device)
    z = self.encoder(x)
    return self.decoder(z)


def detect_anomaly(vae, device, img, losses): 
    running_average_size = 5 #avg of past 4 error losses
    threshold = 27500#24000
    img = img.unsqueeze(0).to(device)
    vae.eval()
    with torch.no_grad():
        encoded_data = vae.encoder(img)
        x_hat = vae(img)
        loss = ((img - x_hat)**2).sum() + vae.encoder.kl
        losses.append(loss.item())
        running_average = sum(losses[-running_average_size:]) / min(len(losses), running_average_size) 
        print(running_average)
        if running_average > threshold: 
            return True
        return False


# load autoencoder model for anomaly detection
d = 4

model = VariationalAutoencoder(latent_dims=d)
device = torch.device("cpu")
model.to(device)
model.load_state_dict(torch.load(vae_path, map_location=torch.device("cpu")))
model.eval()

# transform for anomaly detection

pil = transforms.ToPILImage()

test_transform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
])

camera = cv.VideoCapture(8)

#Live camera image was detected
if camera.isOpened(): 
    losses = []
    
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)#1280
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)#720
    counter = 0 #lets set a counter to minimize anomoly processing
    while True: 
        ret, img = camera.read()
        img_copy = img.copy()

        if not ret: 
            print("Error. Unable to capture the frame.")
            break
        #reduces image processing to every frames
        if counter % 10 != 0:
            continue
        hole_boxes = detect_objects(img)

        for (x, y, w, h) in hole_boxes: 

            maskimg = np.zeros_like(img)
            cv.rectangle(maskimg, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # cv.imshow('maskimg', maskimg)

            mergedimg = cv.bitwise_and(img, maskimg, mask=None)

            imggray = cv.cvtColor(mergedimg, cv.COLOR_BGR2GRAY)

            ret, thresh = cv.threshold(imggray, 100, 255, 0)
            # cv.imshow('thresh', thresh)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            # min_area_line = 0.05
            # max_area_line = 0.08
            # min_area_blob = 0.13 #.14
            # max_area_blob = 0.24 #.18
            # for cnt in contours:
            #     area = cv.contourArea(cnt) / (w * h)
            #     if min_area_blob < area < max_area_blob: 
            #         # cv.drawContours(img, [cnt], 0, (0,255,0), 3)
            #         # label
            #         m = cv.moments(cnt)
            #         cx = int(m['m10']/m['m00'])
            #         cy = int(m['m01']/m['m00'])
            #         cv.rectangle(img, (cx + 65, cy - 20), (cx + 150, cy + 5), (255, 255, 255), -1)
            #         # cv.putText(img, "Oil stain", (cx + 70, cy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            #         cv.putText(img, "Anomaly", (cx + 70, cy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Anomaly Detection
            cropped_img = img_copy[y:y+h, x:x+w]

            transformed_img = test_transform(pil(mergedimg))

            anomaly = detect_anomaly(model, device, transformed_img, losses)

            # label
            if anomaly: 
                cv.rectangle(img, (x + 1, y + 1), (x + 200, y + 35), (255, 255, 255), -1)
                cv.putText(img, "Warning: anomaly", (x + 6, y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                #Target in the Y-Axis of the TCP
                # target = rtde_r.getActualTCPPose()
                # target[1] += 0.001
                # # input("Press enter to get closer to object")
                # rtde_c.moveL(target, 0.25, 0.5, True)
                # time.sleep(0.2)

        cv.imshow('window', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        

        key = cv.waitKey(1)
        if key == ord('q'):
            break

cv.destroyAllWindows()