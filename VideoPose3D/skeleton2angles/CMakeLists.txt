cmake_minimum_required(VERSION 2.8.3)
project(imitation)

add_compile_options(-std=c++11)
add_compile_options(-pthread)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/..)

# find_package(Boost REQUIRED COMPONENTS system)

###########
## Build ##
###########
include_directories(  "/usr/include/eigen3" )

# add_library(ske2ang SHARED ske2ang.cpp)
add_library(ske2ang SHARED s2a.cpp)
