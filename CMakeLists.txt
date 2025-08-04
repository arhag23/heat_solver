cmake_minimum_required(VERSION 3.24)

project(heat_solver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)
# set(CMAKE_BUILD_TYPE Debug)

option(USE_CUDA "Use CUDA acceleration" OFF)
option(USE_HIP "Use AMD acceleration" OFF)

add_executable(heat_solver main.cpp)
target_link_libraries(heat_solver PRIVATE init_conds solvers)

if (USE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CUDA_ARCHITECTURE "80" CACHE STRING "Specify the CUDA architecture (e.g., 80 meaning arch=compute_80,code=sm_80)")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
elseif (USE_HIP)
    find_package(HIP REQUIRED)
    enable_language(HIP)
    set(HIP_ARCHITECTURE "gfx942;gfx90A" CACHE STRING "Specify the HIP architecture (e.g., gfx942;gfx90A)")
    set_source_files_properties(main PROPERTIES LANGUAGE HIP) # this shouldn't be needed?
endif()

add_subdirectory(src)
