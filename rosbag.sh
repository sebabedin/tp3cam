ros2 bag info ./resource/rosbag2_2022_11_09-15_21_22/rosbag2_2022_11_09-15_21_22_0.db3
ros2 bag play -l ./resource/rosbag2_2022_11_09-15_21_22/rosbag2_2022_11_09-15_21_22_0.db3 \
--remap /stereo/left/image_raw:=/left/image_raw \
/stereo/left/camera_info:=/left/camera_info \
/stereo/right/image_raw:=/right/image_raw \
/stereo/right/camera_info:=/right/camera_info