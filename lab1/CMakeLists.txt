cmake_minimum_required(VERSION 3.10)
project(test)

option(USE_FLOAT "Use float instead of double" OFF)
add_executable(test main.cpp)

if (USE_FLOAT)
    target_compile_definitions(test PRIVATE USE_FLOAT)
endif()

