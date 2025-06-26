cmake_minimum_required(VERSION 3.24)

project(heat_solver)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

option(USE_CUDA "Use CUDA acceleration" OFF)
option(USE_HIP "Use AMD acceleration" OFF)

add_executable(heat_solver main.cpp)
target_link_libraries(heat_solver PRIVATE init_conds solvers)

if (USE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CUDA_ARCHITECTURE "80" CACHE STRING "Specify the CUDA architecture (e.g., 80 meaning arch=compute_80,code=sm_80)")
    target_compile_definitions(heat_solver PRIVATE "CUDA")
elseif (USE_HIP)
    find_package(HIP REQUIRED)
    enable_language(HIP)
    set(HIP_ARCHITECTURE "gfx942;gfx90A" CACHE STRING "Specify the HIP architecture (e.g., gfx942;gfx90A)")
    set_source_files_properties(main PROPERTIES LANGUAGE HIP) # this shouldn't be needed?
    target_compile_definitions(heat_solver PRIVATE "HIP")
endif()

add_subdirectory(src)