import cv2
import os
import numpy as np
# This file extracts harris corner information from robot

#This function reads every image from the folder
def load_img_from_folder(folder_path):
    pictures = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures.append(img)
    return pictures

def process_img(image):
    #convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur img
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #Abstract edges
    canny = cv2.Canny(blur, 10, 80, apertureSize = 3)
    #canny = cv2.Canny(blur, 20, 70, apertureSize = 3)
    return canny

# file path for pictures
path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/irregular_base'
images = load_img_from_folder(path)
cur_img = images[0] # lets choose one image to work with

cv2.namedWindow('Color_Frame')
cv2.imshow('Color_Frame', cur_img)
cv2.waitKey(0)

gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.erode(dst, np.ones((3,3),np.uint8), 1)
# Threshold for an optimal value, it may vary depending on the image.
cur_img[dst>0.01*dst.max()]=[0,0,255]

ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
#np.set_printoptions(threshold=np.inf)
print(dst)
cv2.imshow('dst', cur_img)


# # find centroids
# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# # define the criteria to stop and refine the corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# print(corners)
# # Now draw them
# res = np.hstack((centroids,corners))
# res = np.int0(res)
# cur_img[res[:,1],res[:,0]]=[0,0,255]
# cur_img[res[:,3],res[:,2]] = [0,255,0]
# cv2.imshow('dst',cur_img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()