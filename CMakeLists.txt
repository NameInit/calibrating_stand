cmake_minimum_required(VERSION 3.13)
project(StereoProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -Wall")

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(COMMON_LIBS
    ${OpenCV_LIBS}
)

#чтобы не попасть на snap
set(COMMON_LINK_OPTIONS
    -Wl,--push-state,--no-as-needed
    /usr/lib/x86_64-linux-gnu/libpthread.so.0
    -Wl,--pop-state
)

add_executable(stream_depth src/stream_depth.cpp)
add_executable(calibrate src/calibrate.cpp)
add_executable(generator_stereo_x4l2 "src/generator_stereo_x4l2.cpp")
add_executable(test_life_stereo_x4l2 "test/test_life_stereo_x4l2.cpp")
add_executable(test_opencv_version "test/test_opencv_version.cpp")
add_executable(sandbox "sandbox/sandbox.cpp")

target_link_libraries(stream_depth PRIVATE ${COMMON_LIBS})
target_link_libraries(calibrate PRIVATE ${COMMON_LIBS})
target_link_libraries(generator_stereo_x4l2 PRIVATE ${COMMON_LIBS})
target_link_libraries(test_life_stereo_x4l2 PRIVATE ${COMMON_LIBS})
target_link_libraries(test_opencv_version PRIVATE ${COMMON_LIBS})
target_link_libraries("sandbox" PRIVATE ${COMMON_LIBS})

target_link_options(stream_depth PRIVATE ${COMMON_LINK_OPTIONS})
target_link_options(generator_stereo_x4l2 PRIVATE ${COMMON_LINK_OPTIONS})
target_link_options(test_life_stereo_x4l2 PRIVATE ${COMMON_LINK_OPTIONS})
target_link_options(test_opencv_version PRIVATE ${COMMON_LINK_OPTIONS})