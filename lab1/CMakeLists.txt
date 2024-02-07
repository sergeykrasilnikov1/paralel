cmake_minimum_required(VERSION 3.10)
project(test)

option(USE_FLOAT "Use float instead of double" OFF)



if (USE_FLOAT)
    target_compile_definitions(main PRIVATE USE_FLOAT)
endif()

add_executable(test main.cpp)