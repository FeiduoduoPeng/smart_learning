cmake_minimum_required(VERSION 2.8.3)
project(imitation)

add_compile_options(-std=c++11)
add_compile_options(-pthread)
# find_package(Boost REQUIRED COMPONENTS system)

###########
## Build ##
###########
include_directories(  "/usr/include/eigen3" )

# add_library(ske2ang SHARED ske2ang.cpp)
add_library(ske2ang SHARED s2a.cpp)

# add_executable(main main.cpp)
# target_link_libraries(main ske2ang)