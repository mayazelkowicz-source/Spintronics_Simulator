cmake_minimum_required(VERSION 3.10)
project(SpintronicBNNSimulator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(bnn_sim
    main.cpp
    VirtualCrossbar.cpp
    VirtualCrossbarInference.cpp
    CsvUtils.cpp
)
target_compile_options(bnn_sim PRIVATE -Wall -Wextra -Wpedantic)
