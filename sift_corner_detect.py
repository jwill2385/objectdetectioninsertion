import cv2
import os
from cv2 import drawContours
import numpy as np
import matplotlib.pyplot as plt
# This file extracts harris corner information from robot
#Using SIFT we store corners of interest and then

#This function reads every image from the folder
def load_img_from_folder(folder_path):
    pictures = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures.append(img)
    return pictures

#Find largest contour and crop
def find_largest_contour(image):
    #convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur img
    blur = cv2.GaussianBlur(gray, (5,5),0)
    ret,thresh = cv2.threshold(blur,127,255,0) # coverts grayscale to a binary img
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    copy_img = image.copy()
    sort =  sorted(contours, key=cv2.contourArea, reverse=True)
    large_contour = sort[1]
    cv2.drawContours(copy_img, [large_contour], 0, (0,255,0), 2)
    cv2.imshow('Find contour', copy_img)
    cv2.waitKey(0)
    cv2.destroyWindow('Find contour')
    #Abstract edges
    # canny = cv2.Canny(blur, 10, 80, apertureSize = 3)
    #canny = cv2.Canny(blur, 20, 70, apertureSize = 3)
    return large_contour

#This function creates a image mask isolating just the desired contour
def contonur_mask(img, cnt):
    mask= np.zeros_like(img) # we want all our images to be the same size
    cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
    #Show region to edit
    print(mask.shape)
    # cv2.imshow('Mask_Frame', mask)
    # cv2.waitKey(0)
    masked_image = cv2.bitwise_and(img,mask)

    # cv2.imshow("Mask Applied to Image", masked_image)
    # cv2.waitKey(0)

    return masked_image
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
def generate_moments(cur_img, large_contour):
    my_img = cur_img.copy()
    
   
    

    #Pick contour corresponding to object
    #sort =  sorted(contours, key=cv2.contourArea, reverse=True) # This will sort contours in reverse order by area
    # large_contour = contours[0]
    cv2.drawContours(my_img, [large_contour], 0, (0,255,0), 2)
    

    M = cv2.moments(large_contour)
    print('Moments')
    # print(M)
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

#This function gets circle contour information
def generate_ellipse(img, contour):
    my_img = img.copy()
    #TODO: For now this draws contour directly but we will adapt to draw ellipse
    cv2.drawContours(my_img, [contour], 0, (0,255,0), 2)
    

    M = cv2.moments(contour)
    print('Moments')
    # print(M)
    cx = int(M['m10']/ M['m00'])
    cy = int(M['m01']/ M['m00'])
    print('center xy = ', cx,':', cy)
    #draw center
    cv2.circle(my_img, (cx, cy), 8, (0,0,255), -1)
    cv2.imshow('Color_Frame',my_img)
    cv2.waitKey(0)
    # ellipse = cv2.fitEllipse(contour)
    # draw_elipse = cv2.ellipse(img, ellipse, (0,255,0),2)
    # center_x = ellipse[0][0]
    # center_y = ellipse[0][1]
    
# This function gets harris corners of img
def generate_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #update identified corners to ones above threshold
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(3,3),(-1,-1),criteria)
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    print("num of Corners ", corners.shape)
    print(corners)
    cv2.imshow('dst',img)
    cv2.waitKey(0)
    cv2.destroyWindow('dst')

#This function takes my isolated object region and finds all the circle contours
def analyze_circles(img):
    #set size of region
    threshold_area = 200
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur img
    blur = cv2.GaussianBlur(gray, (5,5),0)
    ret,thresh = cv2.threshold(blur,127,255,0) # coverts grayscale to a binary img
    contours,hierarchy = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        #check area of each region if its larger than threshold then generate moments
        area = cv2.contourArea(cnt)
        if area > threshold_area:
            generate_ellipse(img, cnt)

#This function uses harris corner detection to isolate keypoint for sift
def generate_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # ret,thresh = cv2.threshold(gray,127,255,0) # coverts grayscale to a binary img
    #Testing getting corners w threshold img
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #update identified corners to ones above threshold
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(3,3),(-1,-1),criteria)
    #Generate keypoints from corners found
    keypoints = [cv2.KeyPoint(float(x[0]), float(x[1]), 13) for x in corners]
    # Now draw Corners on picture
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    return (keypoints, img)

# file path for pictures
path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/read_irr_base'
images = load_img_from_folder(path)
ref_img = images[0] # lets choose one image to work with
# img2 = images[2]
cv2.namedWindow('Color_Frame')
cv2.imshow('Color_Frame', ref_img)
cv2.waitKey(0)
#If needed crop image here before grayscale
large_contour = find_largest_contour(ref_img)
generate_corners(ref_img)
generate_moments(ref_img, large_contour)
print('NOW LOOKING AT INNER CONTOURS')
# large_contour2 = find_largest_contour(img2)
ref_img_iso = contonur_mask(ref_img, large_contour)
analyze_circles(ref_img_iso)
# img_2_iso = contonur_mask(img2, large_contour2 )
# ref_keypoints, ref_img_copy = generate_keypoints(ref_img_iso)
# kp2, img2_copy = generate_keypoints(img_2_iso)


# keypoints = [cv2.KeyPoint]
np.set_printoptions(threshold=np.inf)


#Compute the SIFT descriptors from harris corner keypoints
# sift = cv2.SIFT_create()

# kp, des = sift.compute(ref_img_copy, ref_keypoints)
# kp2, des2 = sift.compute(img2_copy, kp2)
# Feature matching
#TODO Change feature matching formula
# FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des,des2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# #print("matches: ", len(matches))
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     print("m: ", m, " n: ", n)
#     if m.distance < 0.9*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv2.DrawMatchesFlags_DEFAULT)
# img3 = cv2.drawMatchesKnn(ref_img_copy,kp,img2_copy,kp2,matches,None,**draw_params)

# bf = cv2.BFMatcher()
#find the best 2 matches for each descriptor
#matches = bf.knnMatch(des, des2, k=2)
# matches = bf.match(des, des2)

# matches = sorted(matches, key = lambda x:x.distance)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# print(len(good))

#When using Knn add drawMatchesKnn, otherwise just use drawmatches

# keypoint_img = cv2.drawKeypoints(ref_img_copy, ref_keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
# keypoint_img2 = cv2.drawKeypoints(img2_copy, kp2, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
# img_matching = cv2.drawMatches(ref_img_copy, kp, img2_copy, kp2, matches[:10], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('ref keypoint',keypoint_img)
# cv2.imshow('2nd keypoint',keypoint_img2)
#cv2.imshow('Matches', img_matching)
# plt.imshow(img_matching), plt.show()

print("Finished")
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()