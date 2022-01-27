import cv2
import os
import pyrealsense2

def process_img(image):
    #convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

#This function reads every image from the folder
def load_img_from_folder(folder_path):
    pictures = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures.append(img)
    return pictures

# file path for pictures
#path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/Photos'
path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/photo_clear'
images = load_img_from_folder(path)
print(images)

cv2.namedWindow('Color_Frame')
cv2.imshow('Color_Frame', images[2])
cv2.waitKey(0)
while True:

    key = cv2.waitKey(1)

    gray = cv2.cvtColor(images[2], cv2.COLOR_BGR2GRAY)
    #new_pic = process_img(images[0])
    #cv2.imshow('Color_Frame', gray)

    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 10, 80)
    cv2.imshow('Color_Frame', canny)
    if key == 27:
        break
    if key == ord('q'):
        break

