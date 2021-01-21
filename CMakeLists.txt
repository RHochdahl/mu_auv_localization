cmake_minimum_required(VERSION 2.8.3)
project(mu_auv_localization)

find_package(catkin REQUIRED COMPONENTS
  apriltag_ros
  geometry_msgs
  roscpp
  rospy
  std_msgs
)

catkin_python_setup()
