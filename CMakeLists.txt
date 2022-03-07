cmake_minimum_required(VERSION 3.10)
project(TensorRTInference)

option (DEBUG_MODE              "Set switch to build at Debug mode"             ON)
option (USE_NPP                 "Set switch to build Preprocess mode"           OFF)

enable_language(CXX)
enable_language(CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE Debug)
    message("\n------------------ Debug Mode activated! -------------------\n")
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

if(USE_NPP)
    add_definitions(-DUSE_NPP)
endif()

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)


# TARGETS --------------------------------------------------------------------------------------------------------------
file(GLOB cpp_sources "*.cpp" "*.h" ".hpp")
file(GLOB cuda_sources "*.cu")
add_executable(trt_infer ${cpp_sources} ${cuda_sources})
# add_executable(trt_infer "test.cpp")
target_include_directories(trt_infer PUBLIC ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(trt_infer PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} -lnppc -lnppig -lnppicc -lnppidei -lnppist -lnppisu)
