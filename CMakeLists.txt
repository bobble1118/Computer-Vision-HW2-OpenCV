cmake_minimum_required(VERSION 3.10)

project(CV_HW1)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "C:/cpplib/opencv/build") 

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} project_hw2/include)

add_executable(cv_hw2 project_hw2/main.cpp project_hw2/func.cpp)

target_link_libraries(cv_hw2 ${OpenCV_LIBS})