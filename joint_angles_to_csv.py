from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import os
import string
import csv

#NOTE: YOU have to import rosbags to use this file. pip install rosbags
#Rosbags documentation link "https://ternaris.gitlab.io/rosbags/index.html"

# create reader instance and open for reading
#STEP 1: Create new csv file in move_arm folder and link path below
csv_path = "/home/cvdarbeloff/workspace/ros_ur_driver/src/move_arm/src/joint_angles.csv"

with open(csv_path, 'w', encoding='UTF8') as file:
    data_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #Write header row
    data_writer.writerow(['time', 'nanosec', 'elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint',
    'wrist_2_joint', 'wrist_3_joint', 'q_dot1', 'q_dot2', 'q_dot3', 'q_dot4', 'q_dot5', 'q_dot6',
     'eforce_1', 'eforce_2', 'eforce_3', 'eforce_4', 'eforce_5', 'eforce_6'])
    
    with Reader('/home/cvdarbeloff/workspace/bag_files/joint_angles') as reader:
        # topic and msgtype information is available on .connections list
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)

        # iterate over messages
        counter = 0
        for connection, timestamp, rawdata in reader.messages():
            # if counter ==  2:
            #     break
            msg = deserialize_cdr(rawdata, connection.msgtype) #converts data into a readable format
            time = str(msg.header.stamp)
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
            #Now Grab position, velocity, effort data 
            p = msg.position
            vel = msg.velocity
            eff = msg.effort
            #save everything to csv.
            data_writer.writerow([seconds, nanoseconds, p[0], p[1], p[2], p[3], p[4], p[5], vel[0], vel[1], vel[2], 
            vel[3], vel[4], vel[5], eff[0], eff[1], eff[2], eff[3], eff[4], eff[5]])
            #msg object has postion, velocity, effort data, 
            #to get the time i need to get the header.stamp


    # # messages() accepts connection filters
    # connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
    # for connection, timestamp, rawdata in reader.messages(connections=connections):
    #     msg = deserialize_cdr(rawdata, connection.msgtype)
    #     #print(msg.header.frame_id)