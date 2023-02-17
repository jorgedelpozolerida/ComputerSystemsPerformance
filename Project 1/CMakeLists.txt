cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (threading)

SET (CMAKE_CXX_FLAGS "-std=c++20")

add_compile_options(-pthread)

add_executable(independent_output independent_output.cpp)