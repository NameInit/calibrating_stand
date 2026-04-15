cmake_minimum_required(VERSION 3.13)
project(OakStereoProject)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -Wall")

set(depthai_DIR "/home/ilya/Desktop/depthai-core/build") #тут потом менять придётся

find_package(depthai CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(COMMON_LIBS 
    depthai::core 
    ${OpenCV_LIBS}
)

set(COMMON_LINK_OPTIONS
    -Wl,--push-state,--no-as-needed
    /usr/lib/x86_64-linux-gnu/libpthread.so.0
    -Wl,--pop-state
)

add_executable(stream_depth src/stream_depth.cpp)
add_executable(generator_oak src/generator_oak.cpp)
add_executable(generator_x4l src/generator_x4l.cpp)
add_executable(calibrate src/calibrate.cpp)

target_link_libraries(stream_depth PRIVATE ${COMMON_LIBS})
target_link_libraries(generator_oak PRIVATE ${COMMON_LIBS})
target_link_libraries(generator_x4l PRIVATE ${COMMON_LIBS})
target_link_libraries(calibrate PRIVATE ${OpenCV_LIBS})

target_link_options(stream_depth PRIVATE ${COMMON_LINK_OPTIONS})
target_link_options(generator_oak PRIVATE ${COMMON_LINK_OPTIONS})
target_link_options(generator_x4l PRIVATE ${COMMON_LINK_OPTIONS})