#This script will generate jacovian for my experimental setup
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time

rtde_c = RTDEControl("192.168.1.102")
rtde_r = RTDEReceive("192.168.1.102")
init_q = rtde_r.getActualQ()
print(init_q)
velocity = .4
acceleration = .3

initial_pos = [-0.651351277028219, -2.3850847683348597, 0.9308307806598108, -2.8208285770811976, -1.5902555624591272, 2.957724094390869]
rtde_c.moveJ(initial_pos, velocity, acceleration, True)
time.sleep(4.0)
# rtde_c.stopJ(0.5)
q1_shift = initial_pos.copy()
q1_shift[0] += .2

q2_shift = initial_pos.copy()
q2_shift[1] +=  0.03

q3_shift = initial_pos.copy()
q3_shift[2] +=  0.15

q4_shift = initial_pos.copy()
q4_shift[3] += .2

q5_shift = initial_pos.copy()
q5_shift[4] +=  0.2

q6_shift = initial_pos.copy()
q6_shift[5] +=  0.2

print('init')
print(initial_pos)
print('q5 iso')
print(q1_shift)
rtde_c.moveJ(q1_shift, velocity, acceleration, True)
time.sleep(4.0)
# rtde_c.stopJ(0.5)
# waypoint_1 = [-1.231, -2.66, 1.302, -2.271, -1.91, 3.05]
# rtde_c.moveJ(waypoint_1,  velocity, acceleration, True)
# time.sleep(2.0)
# rtde_c.stopJ(0.5)

# return to home position
# rtde_c.moveJ(init_q)
# Stop the RTDE control script
rtde_c.stopScript()