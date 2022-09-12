add_subdirectory(Thirdparty/DBoW2)

include_directories(
${PROJECT_SOURCE_DIR}/Comparison
${PROJECT_SOURCE_DIR}/Comparison/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Comparison/lib)

add_library(Original_ORB_SLAM3 SHARED
src/ORBextractor.cc
include/ORBextractor.h
include/ORBVocabulary.h)


target_link_libraries(Original_ORB_SLAM3
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PROJECT_SOURCE_DIR}/Comparison/Thirdparty/DBoW2/lib/libDBoW2.so
)

# Build examples

#Utility examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Comparison/app)

add_executable(compare_extractors
        app/compare_extractors.cc)
target_link_libraries(compare_extractors ${PROJECT_NAME} Original_ORB_SLAM3)

add_executable(compare_matchers
        app/compare_matchers.cc)
target_link_libraries(compare_matchers ${PROJECT_NAME} Original_ORB_SLAM3)

add_executable(compare_matchers_euroc
        app/compare_matchers_euroc.cc)
target_link_libraries(compare_matchers_euroc ${PROJECT_NAME} Original_ORB_SLAM3)

add_executable(compare_matchers_tum_rgbd
        app/compare_matchers_tum_rgbd.cc)
target_link_libraries(compare_matchers_tum_rgbd ${PROJECT_NAME} Original_ORB_SLAM3)

add_executable(compare_loop_detection
        app/compare_loop_detection.cc)
target_link_libraries(compare_loop_detection ${PROJECT_NAME} Original_ORB_SLAM3)