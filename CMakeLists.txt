cmake_minimum_required(VERSION 3.6.3 FATAL_ERROR)

project(flann-kdtree)

option(ENABLE_UNIT_TEST "Enable Unit Testing" ON)

find_package(flann REQUIRED)
find_library(LZ4_LIBRARIES NAMES liblz4.so HINTS "lz4/lib")

if(NOT flann_FOUND)
	message(FATAL_ERROR "Failed to find package flann...")
else()
	message(STATUS "Found flann: ${flann_DIR}")
endif()

if(LZ4_LIBRARIES)
	message(STATUS "Found LZ4: ${LZ4_LIBRARIES}")
else()
	message(STATUS "Not found: ${LZ4_LIBRARIES}")
	message(FATAL_ERROR "Cannot find required LZ4 libraries")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O3 -Wall")

add_executable(
	${PROJECT_NAME}
        src/main.cpp
	src/KdTreeFLANN.cpp
)

target_link_libraries(${PROJECT_NAME} flann ${LZ4_LIBRARIES})

if(ENABLE_UNIT_TEST)
	add_subdirectory(test)
endif()
