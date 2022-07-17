import time
import math
import logging
import argparse
from interpreter.interpreter import InterpreterHelper
import socket
import random
import os
import threading
import string
import vgamepad as vg

#Useful constants
pi = 3.1415927
deg2rad = pi/180

UR_force_data_thread = f" \
thread forceData(): \
    while (True): \
        force = to_str(get_tcp_force()) \
        force_len = str_len(force) \
        force = str_sub(force, 2, force_len - 3) \
        left_grip = get_tool_analog_in(0)*4000 \
        right_grip = get_tool_analog_in(1)*4000 \
        joints = to_str(get_actual_joint_positions())\
        joints_len = str_len(joints) \
        joints = str_sub(joints, 1, joints_len - 2) \
        socket_send_line(force + \", \" + to_str(left_grip) + \", \" + to_str(right_grip) + \", \" + joints,\"socket_to_PC\") \
        sleep(0.2) \
    end \
end\n"

#This function is used for connecting to the UR10e
def parseArgs():
    parser = argparse.ArgumentParser(description = 'Send Interpreter commands from file')
    parser.add_argument('-ip', '--ip', help='Specify the IP of the robot (required)')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('-d', '--debug', help='print debug level messages', action='store_true')
    args = parser.parse_args()

    if args.ip is None:
        args.ip = "192.168.1.102"
        print('Using default robot IP address', args.ip)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(level=logging.INFO)

    return args

# Thread to process UR10e responses
def get_reply(conn,):
        """
        read one line from the socket
        :return: text until new line
        """
        conn.settimeout(1)
        end_prog = False
        while not end_prog:
            collected = b''
            while not end_prog:
                try:
                    part = conn.recv(1)
                except socket.timeout:
                    conn.close()
                    textfile = open('force_comm.txt','w')
                    textfile.write('reconnect')
                    textfile.close()
                    end_prog = True
                    break
                        
                if part != b"\n":
                    collected += part
                elif part == b"\n":
                    break

            wrench_str = collected.decode("utf-8")
            wrench_list = wrench_str.split(', ')
            try:
                for i in range(len(wrench_list)):
                    wrench_list[i] = float(wrench_list[i])
            except:
                break #So that if the thread fails, it exits quietly
            wrench_list.insert(8, math.sqrt(wrench_list[0]**2 + wrench_list[1]**2 + wrench_list[2]**2))
            wrench_str = str(wrench_list)
            wrench_str = wrench_str[1:len(wrench_str)-1]
            
            textfile = open('force_comm.txt','a')
            textfile.write(wrench_str + '\n')
            textfile.close()
            

# Main program
if __name__ == '__main__':
    
    UR10e_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    num_msgs = 5
            
    args = parseArgs()
    interpreter = InterpreterHelper(args.ip)
    interpreter.connect()
    interpreter.execute_command("clear_interpreter()")
    interpreter.execute_command("set_analog_out(0, 0)")
    interpreter.execute_command("end_freedrive_mode()")
    default_joint_pose = [-107*deg2rad, -68*deg2rad, -111*deg2rad, -180*deg2rad, -60*deg2rad, 89.5*deg2rad]
    interpreter.execute_command("movej(" + str(default_joint_pose) + ", a=0.5, v=1, t=0, r=0)")
    while (interpreter.execute_command("stateunexecuted") != 0):
        time.sleep(0.5)
    #Put the robot arm in "zero-g" mode
    interpreter.execute_command("freedrive_mode (freeAxes=[1, 1, 1, 1, 1, 1], feature=p[0, 0, 0, 0, 0, 0])")

    # Set up a port for the UR10 to send back messages
    port = random.randint(4000,6000)
    interpreter.execute_command("socket_open(\"192.168.1.101\"," + str(port) + ", \"socket_to_PC\")")
    try:
        UR10e_socket.bind(('192.168.1.101', port))
        UR10e_socket.listen(9)
        conn, address = UR10e_socket.accept()
        print('Connected to UR10e:', address[0], address[1])
    except socket.error:
        print('Could not read data from UR10e')

    force_data_recv_thread = threading.Thread(target=get_reply, args=(conn,))
    force_data_recv_thread.daemon = True #So thread will end when program ends
    force_data_recv_thread.start()

    #Do not touch the handlebar when zeroing the force/torque sensor
    interpreter.execute_command("zero_ftsensor()")
    time.sleep(0.1)
    interpreter.execute_command(UR_force_data_thread)
    interpreter.execute_command("thrd = run forceData()")
    #To communicate between the main program and the force/pose data thread, we set up a text file
    fc_last_modified_time = os.path.getmtime("force_comm.txt")

    current_joint_pose = [0,0,0,0,0,0]
    #Emulate a gamepad (Xbox 360) to map the robot joint positions to axes
##    gamepad_1 = vg.VX360Gamepad()
##    time.sleep(1)
##    gamepad_1.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B) #A button needs to be pressed for Parsec to recognize a joystick
##    gamepad_1.update()
##    time.sleep(1)
##    gamepad_1.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)  # release the A button
##    gamepad_1.update()

    t0_prev = 0
    t1_prev = 0
    t2_prev = 0

    num_runs = 0
    
    while True:
        begin_time = time.time()
        
        if fc_last_modified_time != os.path.getmtime("force_comm.txt"): #New force/pose data recieved
            textfile = open('force_comm.txt','r+')
            wrench_str = textfile.readlines()
            textfile.truncate(0)
            textfile.close()
            fc_last_modified_time = os.path.getmtime("force_comm.txt")

            if 'reconnect' in wrench_str[0]: #reconnect to the UR10e
                while (interpreter.execute_command("stateunexecuted") != 0):
                    time.sleep(0.04)
                    num_msgs += 1
                interpreter.execute_command("socket_close(socket_name=\"socket_to_PC\")")
                port = random.randint(4000,6000)
                interpreter.execute_command("socket_open(\"192.168.1.101\"," + str(port) + ", \"socket_to_PC\")")
                UR10e_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    UR10e_socket.bind(('192.168.1.101', port))
                    UR10e_socket.listen(9)
                    conn, address = UR10e_socket.accept()
                    print('Reconnected to force/torque data port:', address[0], address[1])
                except socket.error:
                    print('Could not connect to force/torque data port.')
                    
                try:
                    interpreter.execute_command("thrd = run forceData()")
                except:
                    interpreter.execute_command(UR_force_data_thread)
                    interpreter.execute_command("thrd = run forceData()")
                                  
                force_data_recv_thread = threading.Thread(target=get_reply, args=(conn,))
                force_data_recv_thread.daemon = True #So thread will end when program ends
                force_data_recv_thread.start()
                num_msgs += 5
                
            else:
                wrench_list = wrench_str[0].split(', ')
                if (len(wrench_list) != 15):
                    continue
                
                #The robot pose is made up of the joint angles (in radians) of the joints from the base to the wrist, respectively.
                #We need to map them from [-pi, pi] to [0, 1] for the joystick axes.
                for i in range(9, 15):
                    current_joint_pose[i-9] = float(wrench_list[i])/pi + 0.5
                #print(current_joint_pose[0:3])

##                gamepad_1.left_trigger_float(value_float=0)  # value between 0.0 and 1.0
##                gamepad_1.right_trigger_float(value_float=0)  # value between 0.0 and 1.0
##                gamepad_1.left_joystick_float(x_value_float=current_joint_pose[0], y_value_float=current_joint_pose[1])  # values between -1.0 and 1.0
##                gamepad_1.right_joystick_float(x_value_float=current_joint_pose[2], y_value_float = 0)  # values between -1.0 and 1.0
##                gamepad_1.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B) #A button needs to be pressed for Parsec to recognize a joystick
##
##                gamepad_1.update()

        if num_runs > 2:
            if (current_joint_pose[0] - t0_prev) > 0.01:
                print('Base right')
            elif (current_joint_pose[0] - t0_prev) < -0.01:
                print('Base left')

            if (current_joint_pose[1] - t1_prev) > 0.01:
                print('Theta2 up')
            elif (current_joint_pose[1] - t1_prev) < -0.01:
                print('Theta2 down')

            if (current_joint_pose[2] - t2_prev) > 0.01:
                print('Theta3 up')
            elif (current_joint_pose[2] - t2_prev) < -0.01:
                print('Theta3 down')

            print('-------------------')
            
        else:
            num_runs += 1

        t0_prev = current_joint_pose[0]
        t1_prev = current_joint_pose[1]
        t2_prev = current_joint_pose[2]
                

        leftover = 0.5 - (time.time() - begin_time) #Refresh the loop at ~1 Hz
        if (leftover > 0.001):
            time.sleep(leftover)
            
##        gamepad_1.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)  # release the B button
##        gamepad_1.update()
##        time.sleep(0.1)
           
