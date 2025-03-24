cmake_minimum_required(VERSION 3.30)
project(technicalsummative)

set(CMAKE_CXX_STANDARD 20)

add_executable(technicalsummative main.cpp
        sequential.cpp
        sequential.h
        openacc_cpu.cpp
        openacc_cpu.h
        openacc_gpu.cpp
        openacc_gpu.h
        openacc_gpu_mem.cpp
        openacc_gpu_mem.h)
