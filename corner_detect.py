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

#This function creates image mask over desired area
def img_mask(image):
    mask = np.zeros_like(image)
    #For testing purposes we will fix the desireed area 
    cv2.rectangle(mask, (37, 655), (1060, 30), (255, 255, 255), -1)
   
    #Show region to edit
    cv2.imshow('Mask_Frame', mask)
    cv2.waitKey(0)
     # now combine the images
    masked_image = cv2.bitwise_and(image,image, mask=mask)
    #cv2.imshow("Mask Applied to Image", masked_image)
    return masked_image

#This function gets moments of img contour
def generate_moments(cur_img, gray_image):
    my_img = cur_img.copy()
    ret,thresh = cv2.threshold(gray_image,127,255,0) # coverts grayscale to a binary img
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
   
    #Manual calculation of X center of mass
    """
    counter = 0
    total_x = 0
    for y in range(32,651):
        for x in range(45,1046):
            if(thresh[y][x] == 0):
                #We found a pixel of our obj
                total_x = total_x + x
                counter = counter + 1
    x_com = total_x / counter
    print("x center of mass", x_com)
    
    """
    
    print('Contours')
    for i, cnt in enumerate(contours):
        print(i, len(cnt))
    #Pick contour corresponding to object
    large_contour = contours[0]
    cv2.drawContours(my_img, [large_contour], 0, (0,255,0), 2)
    

    M = cv2.moments(large_contour)
    print('Moments')
    print(M)
    cx = int(M['m10']/ M['m00'])
    cy = int(M['m01']/ M['m00'])
    print('center xy = ', cx,':', cy)
    m11 = int(M['m11']/ M['m00']) - (cx * cy)
    m20 = int(M['m20']/ M['m00']) - (cx * cx)
    m02 = int(M['m02']/ M['m00']) - (cy * cy)
    print('m11: ', m11)
    print('m20: ', m20)
    print('m02: ', m02)
    #Third order moment
    m21 = int(M['m21']) - 2 * cx * int(M['m11']) - cy * int(M['m20']) + 2 * cx * cx * int(M['m01'])
    m21 = int(m21 / int(M['m00'])) #normalize value
    m12 = int(M['m12']) - 2 * cy * int(M['m11']) - cx * int(M['m02']) + 2 * cy * cy * int(M['m10'])
    m12 = int(m12 / int(M['m00'])) #normalize value
    m30 = int(M['m30']) - 3 * cx * int(M['m20']) + 2 * cx * cx  * int(M['m10'])
    m30 = int(m30 / int(M['m00'])) #normalize value
    m03 = int(M['m03']) - 3 * cy * int(M['m02']) + 2 * cy *cy * int(M['m01'])
    m03 = int(m03 / int(M['m00'])) #normalize value
    print('m21: ', m21)
    print('m12: ', m12)
    print('m30: ', m30)
    print('m03: ', m03)
    #draw center
    cv2.circle(my_img, (cx, cy), 8, (0,0,255), -1)
    cv2.imshow('Color_Frame',my_img)
    cv2.waitKey(0)
    return
# file path for pictures
path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/read_irr_base'
images = load_img_from_folder(path)
cur_img = images[0] # lets choose one image to work with

cv2.namedWindow('Color_Frame')
cv2.imshow('Color_Frame', cur_img)
cv2.waitKey(0)
#If needed crop image here before grayscale
gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
#gray = img_mask(gray)
generate_moments(cur_img, gray)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray,3,3,0.04)
# Threshold for an optimal value, it may vary depending on the image.
cur_img[dst>0.01*dst.max()]=[0,0,255]

ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
np.set_printoptions(threshold=np.inf)
cv2.imshow('dst', cur_img)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(3,3),(-1,-1),criteria)
#print(corners)
#print(corners.shape)
# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
cur_img[res[:,1],res[:,0]]=[0,0,255]
cur_img[res[:,3],res[:,2]] = [0,255,0]
cv2.imshow('dst',cur_img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()