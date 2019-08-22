echo "mkdir -p ~/ros_catkin_ws/src/localisation" >&2
ssh pi@pi mkdir -p /ros_catkin_ws/src/localisation
echo "scp'ing..." >&2
scp -r ~/Documents/localisation pi@pi:ros_catkin_ws/src
