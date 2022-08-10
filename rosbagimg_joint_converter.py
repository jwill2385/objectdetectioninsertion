from ntpath import join
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import os
import string
import csv
import cv2
from cv_bridge import CvBridge
from decimal import Decimal #Use this import to get more  decimal place digits (double vs float)

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
    nanoseconds = int(nanoseconds) * pow(10, -9) # Multiply by 10^-9 to get correct decimal

    combined = seconds + nanoseconds

   
    return seconds, nanoseconds, combined


# create reader instance and open for reading
#STEP 1: select bag file path and image save path location
img_save_path = "/home/cvdarbeloff/Documents/Realsense/realsense_depth/roscam_images_path2"
bag_path = "/home/cvdarbeloff/workspace/bag_files/circle_path_2"
csv_path = "/home/cvdarbeloff/workspace/ros_ur_driver/src/move_arm/src/joint_storage_2.csv"
txt_path = "/home/cvdarbeloff/workspace/ros_ur_driver/src/move_arm/src/joint_times.txt"
photo_txt_path = "/home/cvdarbeloff/workspace/ros_ur_driver/src/move_arm/src/photo_times.txt"
bridge = CvBridge()
#cv2.namedWindow("Color_frame")
array = []


    
with Reader(bag_path) as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    # There are 2 message types here sensor_msgs/msg/Image, and sensor_msgs/msg/JointState
    # 2 connection topics /camera/color/image_raw, /joint_states
    num_photos = 0
    num_states = 0
    #Initialize photo time array and joint dictionary
    photo_times = []
    joint_data = {}
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/camera/color/image_raw' :
            # Read image data
            msg = deserialize_cdr(rawdata, connection.msgtype) #converts data into a readable format
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #save to bgr image format
            img_name = "auto_base_{}.jpg".format(num_photos)
            time = str(msg.header.stamp)
            # print(time)
            sec, nano, exact_time = time_parser(time)
            # store image in correct folder
            # currnet_pic = cv2.imwrite(os.path.join(img_save_path, img_name), cv_image)
            # print("{} written".format(img_name))

            num_photos += 1
            photo_times.append(exact_time)
            # print(msg.header.stamp)
        if connection.topic == '/joint_states' :
            # Read joint Data
            msg = deserialize_cdr(rawdata, connection.msgtype) #converts data into a readable format
            time = str(msg.header.stamp)
            sec, nano, exact_time = time_parser(time)
            #Now Grab position, velocity, effort data 
            p = msg.position
            vel = msg.velocity
            eff = msg.effort
            joint_data[exact_time] = [p, vel, eff]
            num_states += 1
    
  

        
    print("Photo count", num_photos)
    print('State count', num_states)


# Now That I have all joint data and times match each time to a joint
matches = 0 
# Write to txt file
# with open(txt_path, 'w') as file:
#     for key in joint_data:
#         file.write(str(key))
#         file.write('\n')
# with open(photo_txt_path, 'w') as file:
#     for t in photo_times:
#         file.write(str(t))
#         file.write('\n')

# WRITE TO CSV FILE
with open(csv_path, 'w', encoding='UTF8') as cfile:
    data_writer = csv.writer(cfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Write header row
    data_writer.writerow(['time', 'elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint',
    'wrist_2_joint', 'wrist_3_joint', 'q_dot1', 'q_dot2', 'q_dot3', 'q_dot4', 'q_dot5', 'q_dot6',
     'eforce_1', 'eforce_2', 'eforce_3', 'eforce_4', 'eforce_5', 'eforce_6'])
    for t in photo_times:
        # Loop through joint keys until first key is bigger than photo time
        for key in joint_data:
            if key >= t:
                # We have found closest joint reading
                print('Found')
                print('Photo', t)
                print('Joint', key)
                p, vel, eff = joint_data[key]
                data_writer.writerow([t, p[0], p[1], p[2], p[3], p[4], p[5], vel[0], vel[1], vel[2], 
                vel[3], vel[4], vel[5], eff[0], eff[1], eff[2], eff[3], eff[4], eff[5]])
                matches += 1
                break

print("I have {} Matches".format(matches))


