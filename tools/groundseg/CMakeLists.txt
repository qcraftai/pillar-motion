cmake_minimum_required(VERSION 3.10)
project(example)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PCL)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

# add_library(linefit_ground_segmentation ground_segmentation.cc segment.cc bin.cc)
# target_link_libraries(linefit_ground_segmentation ${PCL_LIBRARIES})

# add_executable(test test.cc)
# target_link_libraries(test linefit_ground_segmentation)

find_package(pybind11 REQUIRED)
pybind11_add_module(segmentation segmentation.cc ground_segmentation.cc segment.cc bin.cc)
target_link_libraries(segmentation PRIVATE ${PCL_LIBRARIES})

# find_package(pybind11 REQUIRED)
# pybind11_add_module(example example.cpp)