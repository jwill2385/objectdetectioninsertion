import os
import cv2
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import numpy as np
import time

# define all functions up top

def stringToArr(move_string):
    #This function takes in the text string from vision2move and converts it to np array
    deltaq = np.zeros(6)
    #Split string into lists
    split_data = move_string.split(",")
    #fill delta q
    for i in range(6):
        num = float(split_data[i])
        deltaq[i] = num
    # Now return desired q change
    print("string conversion complete")
    return deltaq


#Global variables
readtxt_file = "/home/cvdarbeloff/rt_kernel_build/ur_rtde/examples/py/robot_vision_control/vision2move.txt"
writetxt_file = "/home/cvdarbeloff/rt_kernel_build/ur_rtde/examples/py/robot_vision_control/move2vision.txt"
movestate = False
move_position = []
try:

    for i in range(2):

        if movestate == False:
            #Keep reading from vision2text until movestate is true
            print("reading text file")
            with open(readtxt_file, "r+") as f:
                lines = f.readlines()
                if len(lines):
                    #This means we see some text so lets update our move position
                    for line in lines:
                        move_position = line
                    # move_position = lines
                    #clear the file
                    f.truncate(0)
                    movestate = True


        # Now that movestate is true put move logic here
        if movestate == True:
            #Move towards desired object until coordinates are reached then set it to fales
            print('cool')
            #Assume brackets are already removed from move_position string
            #get delta q value as a np array
            deltaq = stringToArr(move_position)
            
            print(deltaq)
            #This tells vision script to begin searching for object again
            with open(writetxt_file, 'w') as file:
                file.write('ready')
            movestate = False
    #break out of infinite loop if q is pressed

except KeyboardInterrupt:
    print("I escaped")
