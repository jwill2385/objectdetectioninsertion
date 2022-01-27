#Testing
import cv2
import os
import pyrealsense2
from realsense_depth.realsense_depth import *

point = (300, 400)

def show_distance(event, x, y, args, params):
    global point
    point = (x,y) #update position of point of interest(red circle) to mouse position

# Initialize Camera Intel Realsense
dc = DepthCamera() #Class from realsense depth

# Create a mouse event
cv2.namedWindow("Color_frame")
cv2.setMouseCallback("Color_frame", show_distance) #Captures position of mouse on screen

#Create image path to write to
path = '/home/cvdarbeloff/Documents/Realsense/realsense_depth/photo_train'
img_counter = 43
while True:

    ret, depth_frame, color_frame = dc.get_frame()

    # Show distance for a specific that
    #cv2.circle(color_frame, point, 4, (0, 0, 255) )

    # We want to capture distance at this specific point
    #Note when using array we plot Y coordinate then X Coordinate
    distance = depth_frame[point[1], point[0]]
    #cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,  (255, 0, 0), 2)
    #print(distance)
    cv2.imshow("Depth_frame", depth_frame)
    cv2.imshow("Color_frame", color_frame)
    
 
    key = cv2.waitKey(1)
    if key == ord('p'):
        #Capture image
        #Create name for new image
        img_name = "hd_photo_{}.jpg".format(img_counter)
        # store image in correct folder
        currnet_pic = cv2.imwrite(os.path.join(path, img_name), color_frame)
        print("{} written".format(img_name))
        #increment counter for new images
        img_counter += 1
    elif key == ord('q'):
        break

cv2.destroyAllWindows

# Now time to process images.
#ok