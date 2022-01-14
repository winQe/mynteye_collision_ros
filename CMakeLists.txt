cmake_minimum_required(VERSION 3.0.2)
project(mynteye_collision_ros)
set(CMAKE_BUILD_TYPE Debug)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)
find_package(OpenCV 4 REQUIRED) # manual specification of the OpenCVConfig.cmake path is necessary since otherwise, cmake assigns /opt/local a higher priority and ocv2.4 will be used
if(OpenCV_FOUND)
  MESSAGE ( STATUS "Found OpenCV: ${OpenCV_VERSION}" )
  MESSAGE ( STATUS "OpenCV_INCLUDE_DIRS= ${OpenCV_INCLUDE_DIRS}" )
endif(OpenCV_FOUND)

add_message_files(
  FILES
  closest_distance.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

add_definitions(-DOPENCV)
add_definitions(-O3 -g)
add_definitions (-DWITH_OPENCV) 

catkin_package(
 CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}_node 
src/main.cpp
src/CDNeuralNet.cpp
src/CDNeuralNet.hpp
)

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# link_directories(/home/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64

target_link_libraries(${PROJECT_NAME}_node
  pthread
  stdc++
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  mynteye_depth
  ${OpenCV_LIBS}
)

# add_library(LIBeSPDI SHARED IMPORTED )
# set_target_properties(LIBeSPDI PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64/libeSPDI.so)
# target_link_libraries(eSPDI SHARED LIBeSPDI )



# # use, i.e. don't skip the full RPATH for the build tree
# set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# set(CMAKE_INSTALL_RPATH "~/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64")
# # add the automatically determined parts of the RPATH
# # which point to directories outside the build tree to the install RPATH
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)



# include_directories(
#     ~/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64
# )

# set(SOURCES main.cpp ~/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64/libeSPDI.so)
# add_executable(libesPDI ${SOURCES})