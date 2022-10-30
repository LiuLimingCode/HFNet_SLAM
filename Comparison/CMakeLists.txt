cmake_minimum_required(VERSION 2.8)
project(HFNet_SLAM_Comparison)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
MESSAGE("build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -fopenmp -march=native -pthread")

find_package(OpenCV 4.4 REQUIRED
PATHS ~/src/opencv-4.5/install
NO_DEFAULT_PATH)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

find_package(Eigen3 3.1.0 REQUIRED)

add_subdirectory(Thirdparty/DBoW2)

include_directories(
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/
${PROJECT_SOURCE_DIR}/../
${PROJECT_SOURCE_DIR}/../Thirdparty/Sophus
${PROJECT_SOURCE_DIR}/../include/
${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/ORBextractor.cc
include/ORBextractor.h
include/ORBVocabulary.h)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../lib/libHFNet_SLAM.so
)

# Build examples

#Utility examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/app)

add_executable(compare_extractors
        app/compare_extractors.cc)
target_link_libraries(compare_extractors ${PROJECT_NAME})

add_executable(compare_matchers
        app/compare_matchers.cc)
target_link_libraries(compare_matchers ${PROJECT_NAME})

add_executable(compare_matchers_euroc
        app/compare_matchers_euroc.cc)
target_link_libraries(compare_matchers_euroc ${PROJECT_NAME})

add_executable(compare_matchers_tum_rgbd
        app/compare_matchers_tum_rgbd.cc)
target_link_libraries(compare_matchers_tum_rgbd ${PROJECT_NAME})

add_executable(compare_loop_detection
        app/compare_loop_detection.cc)
target_link_libraries(compare_loop_detection ${PROJECT_NAME})
