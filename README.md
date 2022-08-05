# objectdetectioninsertion
algorithm for peg-hole object detection

Weights and CFG file can be found on google drive. They are too big to upload directly

// Installing ROS2 for intel realsense
Start here -https://dev.intelrealsense.com/docs/ros-wrapper

C8B3A55A6F3EFCDE: ""CN = Intel(R) Intel(R) Realsense", O=Intel Corporation" not changed
gpg: Total number processed: 1
gpg:              unchanged: 1


// Here are some realsense presets https://github.com/IntelRealSense/librealsense/wiki/D400-Series-Visual-Presets
// install librealsense https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

latest release is ros2 beta - https://github.com/IntelRealSense/realsense-ros/tree/ros2-beta

Launch the realsense ros node
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30

I have to install cv_bridge to get image data
https://github.com/ros-perception/vision_opencv/tree/ros2/cv_bridge
The rostopic /camera/color/image_raw spits out a rosmsg of type sensor_msgs/image
use cv_brige to convert this message type into a opencv image which we can then easily save
