cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (threading)

add_compile_options(-pthread)

add_executable(independent_output independent_output.cpp)