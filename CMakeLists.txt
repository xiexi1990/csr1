cmake_minimum_required(VERSION 3.13)

project(csr1  CXX C CUDA)

set(cnpy_lib "/usr/local/lib/libcnpy.so")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE csr1.cu)
target_link_libraries(${PROJECT_NAME} PUBLIC ${cnpy_lib})
