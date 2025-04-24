# Diplom
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build
source install/setup.bash
ros2 launch gesture_control gesture_control_launch.py 
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14540@127.0.0.1:14580
