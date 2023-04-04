import cv2
import pyrealsense2 as rs
import math
import os
import matplotlib.pyplot as plt
import numpy as np

# def initialize_camera():
#     # start the frames pipe
#     p = rs.pipeline()
#     conf = rs.config()
#     conf.enable_stream(rs.stream.accel)
#     conf.enable_stream(rs.stream.gyro)
#     prof = p.start(conf)
#     return p


# def gyro_data(gyro):
#     return np.asarray([gyro.x, gyro.y, gyro.z])


# def accel_data(accel):
#     return np.asarray([accel.x, accel.y, accel.z])

# p = initialize_camera()
# try:
#     while True:
#         f = p.wait_for_frames()
#         accel = accel_data(f[0].as_motion_frame().get_motion_data())
#         gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
#         print("accelerometer: ", accel)
#         print("gyro: ", gyro)

# finally:
#     p.stop()




theta = []
thetadot = []
def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel)
    conf.enable_stream(rs.stream.gyro)

    #Low Resolution. Works with USB 2.0
    conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = p.start(conf)
    return p

p = initialize_camera()

first = True
alpha = 0.98
totalgyroangleY = 0
flag = True
counter = 0

try:
    while flag:
        f = p.wait_for_frames()
        color_frame = f.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        
        # key = cv2.waitKey(0)
        #gather IMU data
        accel = f.first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
        gyro = f.first_or_default(rs.stream.gyro).as_motion_frame().get_motion_data()
        # accel = f[0].as_motion_frame().get_motion_data()
        # gyro = f[1].as_motion_frame().get_motion_data()

        ts = f.get_timestamp()

        #calculation for the first frame
        if (first):
            first = False
            last_ts_gyro = ts

            # accelerometer calculation
            accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
            accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
            accel_angle_y = math.degrees(math.pi)

            continue

        #calculation for the second frame onwards

        # gyrometer calculations
        dt_gyro = (ts - last_ts_gyro) / 1000
        last_ts_gyro = ts

        gyro_angle_x = gyro.x * dt_gyro
        gyro_angle_y = gyro.y * dt_gyro
        gyro_angle_z = gyro.z * dt_gyro

        dangleX = gyro_angle_x * 57.2958
        dangleY = gyro_angle_y * 57.2958
        dangleZ = gyro_angle_z * 57.2958

        totalgyroangleX = accel_angle_x + dangleX
        # totalgyroangleY = accel_angle_y + dangleY
        totalgyroangleY = accel_angle_y + dangleY + totalgyroangleY
        totalgyroangleZ = accel_angle_z + dangleZ

        #accelerometer calculation
        accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
        accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
        # accel_angle_y = math.degrees(math.pi)
        accel_angle_y = 0 #Reset accel angle y for next timestep

        #combining gyrometer and accelerometer angles
        combinedangleX = totalgyroangleX * alpha + accel_angle_x * (1-alpha)
        combinedangleZ = totalgyroangleZ * alpha + accel_angle_z * (1-alpha)
        combinedangleY = totalgyroangleY


        print("Angle -  X: " + str(round(combinedangleX,2)) + "   Y: " + str(round(combinedangleY,2)) + "   Z: " + str(round(combinedangleZ,2)))
     
        # print("Angular velocity: " + str(gyro.z))
        z_angle = round(combinedangleZ,2) 
        if(z_angle > 0):
            theta.append(180 - z_angle)
        elif (z_angle < 0):
            theta.append(-180 - z_angle)
        else:
            theta.append(0)
        thetadot.append(float(gyro.z))
        counter += 1
        if counter == 3000:
            flag = False
        cv2.imshow("Color_frame", color_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
finally:

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8, 10))
    ax[0].plot(np.array(theta))
    ax[1].plot(np.array(thetadot))
    plt.show()
    p.stop()