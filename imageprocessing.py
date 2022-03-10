from pickle import FALSE
from tkinter import TRUE
import cv2
import os
from matplotlib import projections
import numpy as np
import pyrealsense2
import matplotlib.pyplot as plt
import pandas as pd
import math

#This function takes in an array of a [u,v,1] image point and converts into point in camera frame
def convert_to_camera_frame(array):
    fx = 919.84
    fy = 917.85
    ox = 640
    oy = 360
    cam_to_img_mat = [[fx, 0, ox], [0, fy, oy], [0, 0, 1]]
    cam_t_img_arr = np.array(cam_to_img_mat)
    image_cor = np.array(array)
    cam_coor = np.dot(np.linalg.inv(cam_t_img_arr), image_cor) #inv(Image_mat) * [u,v,1]
    theta = 60 * math.pi / 180
    c = math.cos(theta)
    s = math.sin(theta)
    world_to_camera_mat = [[1, 0, 0], [0, c, -s], [0, s, c]]
    translation = [[-.179], [.3215], [.2281]]
    translation = np.array(translation)
    world_to_camera_mat = np.array(world_to_camera_mat)
    # add a 1 for the world transform
    #np.append(cam_coor, [1], axis = 0)
    #np.append(translation, [[1]], axis = 0)
    world_coor =  np.dot(np.linalg.inv(world_to_camera_mat), np.subtract(cam_coor, translation))
    #print("UV: ")
    #print(image_cor)
    print("Camer:")
    print(cam_coor)
    camera_xyz = [cam_coor[0], cam_coor[1], cam_coor[2]]
    world_xyz = [world_coor[0], world_coor[1], world_coor[2]]
    return camera_xyz

def process_img(image):
    #convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    #blur img
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #Abstract edges
    canny = cv2.Canny(blur, 10, 80, apertureSize = 3)
    #canny = cv2.Canny(blur, 20, 70, apertureSize = 3)
    return canny

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

#This function reads every image from the folder
def load_img_from_folder(folder_path):
    pictures = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures.append(img)
    return pictures

#This function creates image mask over desired area
def img_mask(image):
    mask = np.zeros_like(image)
    #For testing purposes we will fix the desireed area 
    cv2.rectangle(mask, (551, 89), (870, 480), (255, 255, 255), -1)
   
    #Show region to edit
    cv2.imshow('Mask_Frame', mask)
    cv2.waitKey(0)
     # now combine the images
    masked_image = cv2.bitwise_and(image,image, mask=mask)
    #cv2.imshow("Mask Applied to Image", masked_image)
    return masked_image
    


# file path for pictures
path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/Hole_exp_photos'
#path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/photo_clear'
images = load_img_from_folder(path)
cur_img = images[0] # lets choose one image to work with



#print(circles.shape)
# plt.show()
cv2.namedWindow('Color_Frame')
cv2.namedWindow('Mask_Frame')
cv2.imshow('Color_Frame', cur_img)
cv2.waitKey(0)

canny = process_img(cur_img)
cv2.imshow('Color_Frame', canny)
cv2.waitKey(0)

# Create image mask
masked = img_mask(canny)
#Grab all contours of masked image
contours, hierarch = cv2.findContours(masked, mode=cv2.RETR_LIST, method= cv2.CHAIN_APPROX_SIMPLE )
for i, cnt in enumerate(contours):
    print(i, len(cnt))
    # print(cnt)
# This is the contour of the ellipse (Second largest. Largest is mask frame)
cnt = contours[6]
x = []
y = []
#Store x,y coordinate of all the points on ellipse here
for i in range(len(cnt)):
    x.append(cnt[i][0][0])
    y.append(cnt[i][0][1])
plt.plot(x, y)
plt.show()

# extract camera frame points of ellipse as x,y are in pixels
camera_coordinates = []
for a in range(len(x)):
    arr = []
    arr.append(x[a])
    arr.append(y[a])
    arr.append(1.0)
    #print("I am at setep " + str(a))
    cam_arry_temp = convert_to_camera_frame(arr)
    #print(cam_arry_temp)
    camera_coordinates.append(cam_arry_temp)

camera_coordinates = np.array(camera_coordinates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(camera_coordinates[:,0], camera_coordinates[:,1], camera_coordinates[:,2])
plt.show()
#method 2 to get ellipse
ellipse = cv2.fitEllipse(cnt)
draw_elipse = cv2.ellipse(cur_img, ellipse, (0,255,0),2)
center_x = ellipse[0][0]
center_y = ellipse[0][1]
a = ellipse[1][0] / 2
b = ellipse[1][1] / 2
tilt_angle = ellipse[2]
major_axis = 0
minor_axis = 0
if (a > b):
    major_axis = a
    minor_axis = b
else:
    major_axis = b
    minor_axis = a

print("Ellipse Values")
print('Major, minor, angle','centerxy')
print(major_axis)
print(minor_axis)
print(tilt_angle)
print(center_x)
print(center_y)
cv2.imshow('Mask_Frame', masked)

"""
"""
while True:

    key = cv2.waitKey(1)
    #Commenting out Hough logic for this experiment
    #lines = cv2.HoughLines(masked, 1, np.pi/180, 110 ) # list of r, theas
    #lines = cv2.HoughLinesP(masked, 1, np.pi/180, 100, minLineLength = 10, maxLineGap = 30)
    #img = hough_transform(masked, lines, True) # set true if using oughlinesP
    # merge houghlines with original img
    #new_img = cv2.addWeighted(cur_img, 0.8, img, 1, 0, 0) 
    #cv2.imshow('Color_Frame', new_img)
    cv2.imshow('Color_Frame', cur_img)
    if key == 27:
        break
    if key == ord('q'):
        break
   
        

