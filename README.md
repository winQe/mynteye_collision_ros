# MYNT-EYE Collision ROS

## Overview

ROS package developed for object detection with MYNT EYE D RGB-D Camera. By default, the package uses **YOLO v4** pretrained model, has also been tested using **YOLO v3-tiny weights and cfg file**. Currently only detect person and chair classes, but can easily edit and change the code to include other pre-trained YOLO classes. The node has been tested on a Robot-ATV.

## Installation

### Hardware Requirements

* MYNT EYE D-1000 RGB-D Camera

### Dependencies

* ROS Melodic (Ubuntu 18.04)
* MYNT-EYE-D SDK
* OpenCV 4.1 with DNN module

### Building
In order to install, clone the repository into your catkin workspace and compile the package.

    cd catkin_ws/src
    git clone https://github.com/winQe/mynteye_collision_ros
    cd ..
    catkin_make

### Download weights
The yolov3-tiny.weights are included in this repository, however the pre-trained yolov4.weights must be downloaded manually.
    
    cd src/mynteye_collision_ros/src/darknet
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


## Usage
Before using, there are a few things to take note of 
1. [main.cpp](src/main.cpp) line 25-27 : Make sure the path to the yolo weights, config, and coco.names files are correct. Use absolute path to avoid any errors.
2. [CDNeuralNet.cpp](src/CDNeuralNet.cpp) line 74 & 104 : You can change the minimum detected confidence and the detected classes here. Class Id depends on the line number of coco.name file. Currently only detects object of classid 0 (person) and classid 56 (chair).

To run the node
    
    rosrun mynteye_collision_ros mynteye_collision_ros_node 
 
 ## Nodes
### Node : mynt_eye_collision_ros_node

### Subscribed topics
 * None

### Published topics
* **`/collision_detection`** (Message type [mynteye_collision_ros::closest_distance])
  
  Publishes the distance, coordinates, and object class id of the closest detected object, including its bounding box and confidence.

## TODO
* Recalibrate the depth scale at [main.cpp](src/main.cpp) line 106 as the distance is not very accurate in the current state