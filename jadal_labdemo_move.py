#Lab Demo 2-16-23 robot motion script
#This script is for the anomoloy detection robot motion
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time

rtde_c = RTDEControl("192.168.1.102")
rtde_r = RTDEReceive("192.168.1.102")
init_q = rtde_r.getActualQ()
print(init_q)
velocity = .4
acceleration = .3
# starting_pos = [-0.938, -2.527, 1.023, -2.413, -1.311, 2.981]
waypoint_1 = [-1.231, -2.66, 1.302, -2.271, -1.91, 3.05]
waypoint_2 = [-1.571, -2.814, 1.624, -2.10, -2.602, 3.124]
waypoint_3 = [-.920, -2.37, 1.02, -3.21, -1.62, 2.98]
waypoint_4 = [-1.254, -2.55, 1.02, -2.88, -1.13, 3.26]
waypoint_5 = [-1.255, -2.55, 1.02, -2.40, -1.13, 3.26]
waypoint_6 = [-1.26, -2.40, 1.02, -2.54, -0.76, 3.26] # first zoom in
waypoint_7 = [-1.26, -2.13, 1.02, -2.84, -0.71, 3.45] # final zoom in
rtde_c.moveJ(waypoint_1,  velocity, acceleration, True)
time.sleep(2.0)
rtde_c.stopJ(0.5)
rtde_c.moveJ(waypoint_2,  velocity, acceleration, True)
time.sleep(5.0)
rtde_c.stopJ(0.5)
time.sleep(5.0) #give robot time to pause and render right image
# Move back to initial joint configuration
rtde_c.moveJ(init_q, velocity-.1, acceleration -.1)
# now go to waypoint 3
rtde_c.moveJ(waypoint_3,  velocity, acceleration, True)
time.sleep(5.0)
rtde_c.stopJ(0.5)
rtde_c.moveJ(waypoint_4,  velocity, acceleration, True)
time.sleep(5.0)
rtde_c.stopJ(0.5)
rtde_c.moveJ(waypoint_5,  velocity, acceleration, True)
time.sleep(5.0)
rtde_c.stopJ(0.5)
rtde_c.moveJ(waypoint_6,  velocity, acceleration, True)
time.sleep(5.0)
rtde_c.stopJ(0.5)
input("Anomoly spotted press enter to continue")
rtde_c.moveJ(waypoint_7,  velocity, acceleration, True)
time.sleep(5.0)
rtde_c.stopJ(0.5)
time.sleep(10.0)
# return to home position
rtde_c.moveJ(init_q)
# Stop the RTDE control script
rtde_c.stopScript()
