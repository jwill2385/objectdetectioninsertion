from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import os
import string
import csv
import cv2
from cv_bridge import CvBridge

#NOTE: YOU have to import rosbags to use this file. pip install rosbags
#Rosbags documentation link "https://ternaris.gitlab.io/rosbags/index.html"
#NOTE: Install cv_bridge here: https://github.com/ros-perception/vision_opencv/tree/ros2/cv_bridge
#Converts sensor_msg/image into a opencv image 
# - documentation http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython



#This function takes in time string and parses into seconds, nanoseconds
def time_parser(time):
    # We need to parse data to get just time numbers
    words = time.split(',')
    #lets get just the time numbers data
    start = words[0].find("sec=")
    seconds = (words[0])[start + 4:]
    start = words[1].find("=")
    nanoseconds = (words[1])[start +1:]
    #Now convert objects from strings to integer
    seconds = int(seconds)
    nanoseconds = int(nanoseconds)
    return seconds, nanoseconds


# create reader instance and open for reading
#STEP 1: select bag file path and image save path location
img_save_path = "/home/cvdarbeloff/Documents/Realsense/realsense_depth/roscam_images"
bag_path = "/home/cvdarbeloff/workspace/bag_files/camera_and_joints"
csv_path = "/home/cvdarbeloff/workspace/ros_ur_driver/src/move_arm/src/joint_angles_3.csv"
bridge = CvBridge()
#cv2.namedWindow("Color_frame")
array = []


    
with Reader(bag_path) as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    img_counter = 0
    # There are 2 message types here sensor_msgs/msg/Image, and sensor_msgs/msg/JointState
    # 2 connection topics /camera/color/image_raw, /joint_states
    num_photos = 0
    num_states = 0
    #TODO: Read all timesteps of image into an array
    # Read all timesteps for joint states. Match each img timestep with one from joint state. save to CSV
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/camera/color/image_raw' :
            msg = deserialize_cdr(rawdata, connection.msgtype) #converts data into a readable format
            num_photos += 1
            print(msg.header.stamp)
        if connection.topic == '/joint_states' :
            msg = deserialize_cdr(rawdata, connection.msgtype) #converts data into a readable format
            # print(msg.header.stamp)
            num_states +=1
        # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #save to bgr image format
        # img_name = "auto_base_{}.jpg".format(img_counter)
        # time = str(msg.header.stamp)
        # sec, nano = time_parser(time)
        # store image in correct folder
        # currnet_pic = cv2.imwrite(os.path.join(img_save_path, img_name), cv_image)
        # print("{} written".format(img_name))
        # print(sec, nano, "Yes")
        print("Photo count", num_photos)
        print('State count', num_states)
        key = cv2.waitKey(1)
        # cv2.imshow('Color_frame', cv_image)
        if key == ord('q'):
            break
        img_counter += 1
    
    # # messages() accepts connection filters
    # connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
    # for connection, timestamp, rawdata in reader.messages(connections=connections):
    #     msg = deserialize_cdr(rawdata, connection.msgtype)
    #     #print(msg.header.frame_id)