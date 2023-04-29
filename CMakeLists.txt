
cmake_minimum_required(VERSION 3.8)

add_definitions(-std=c++11)

project(NeuralNetworkDeployment VERSION 1.0)

include_directories(${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/src/)
link_directories(${PROJECT_SOURCE_DIR}/libs/)

add_library(NeuralNetworkDeployment SHARED src/main.cpp)

target_link_libraries(NeuralNetworkDeployment onnxruntime.1.10.0 opencv_core.405 opencv_imgcodecs.405 opencv_videoio.405 opencv_imgproc.405)

file(COPY assets/ DESTINATION ${PROJECT_BINARY_DIR}/assets/)
