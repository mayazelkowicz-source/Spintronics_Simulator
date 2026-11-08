cmake_minimum_required(VERSION 3.10)
project(SpintronicBNNSimulator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Simulator engine shared by the main program and the unit tests.
add_library(bnn_core STATIC
    VirtualCrossbar.cpp
    VirtualCrossbarInference.cpp
    CsvUtils.cpp
    MTJCrossbar.cpp
    NoiseInjector.cpp
    CliOptions.cpp
)
target_include_directories(bnn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(bnn_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bnn_sim main.cpp)
target_link_libraries(bnn_sim PRIVATE bnn_core)
target_compile_options(bnn_sim PRIVATE -Wall -Wextra -Wpedantic)

# Unit tests. Run from the repository root so the reference CSV files are found:
#   ctest --test-dir simulator_core/build --output-on-failure
enable_testing()
add_executable(noise_tests tests/noise_tests.cpp)
target_link_libraries(noise_tests PRIVATE bnn_core)
target_compile_options(noise_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME noise_tests COMMAND noise_tests
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
