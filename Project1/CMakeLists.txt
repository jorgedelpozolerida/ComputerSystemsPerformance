cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (threading)

set(CMAKE_CXX_FLAGS "-pthread")

add_executable(independent_output independent_output.cpp)

add_executable(concurrent_output concurrent_output.cpp)