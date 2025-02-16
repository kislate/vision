cmake_minimum_required(VERSION 3.0.2)
project(opencv_cpp_yolov5)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV 4 REQUIRED)
include_directories(/usr/local/opencv-4.6.0/include/opencv4)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  image_transport
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs image_transport cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(qr_detection_node src/qr_detection_node.cpp src/qr_detector.cpp)
target_link_libraries(qr_detection_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})