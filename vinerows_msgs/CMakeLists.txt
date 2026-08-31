cmake_minimum_required(VERSION 3.5)
project(vinerows_msgs) 

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(general_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Vinerow.msg" 
  "msg/Vinerowlist.msg"
  "msg/FusionVinerow.msg"
  "msg/GroundTruthVinerow.msg"
  "msg/PerceptionTx.msg"
  "msg/PerceptionRx.msg"
  "msg/TrackedVinerow.msg"
  "msg/SmartSteerDebug.msg"
  "msg/SmartSteerDebug2.msg"
  "msg/SmartSteerSpecialtyDebug.msg"
  "msg/GPSRawUDP.msg"
  "msg/VirtualLineDebug.msg"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs general_msgs
)

ament_package()
