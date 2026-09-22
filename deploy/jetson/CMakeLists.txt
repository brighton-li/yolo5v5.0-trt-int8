cmake_minimum_required(VERSION 3.16)
project(yolo_trt CXX)

# ============================================================================
#  Jetson / x86 Linux + TensorRT 10 C++ 部署
#
#  在 Jetson 上:
#     mkdir build && cd build && cmake .. && make -j$(nproc)
#     ./yolo_trt ../best_int8_minmax.engine ../val ../out
#
#  在 x86 Linux (同样可行, 用于先在开发机上验证代码逻辑):
#     cmake -DTRT_ROOT=/path/to/TensorRT-10.x .. && make -j
# ============================================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ---------------------------------------------------------------- OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV      : ${OpenCV_VERSION}  (${OpenCV_INCLUDE_DIRS})")

# ---------------------------------------------------------------- CUDA
find_library(CUDART_LIB NAMES cudart
  PATHS
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/local/cuda/targets/x86_64-linux/lib
    $ENV{CUDA_HOME}/lib64
)
if(NOT CUDART_LIB)
  message(FATAL_ERROR "找不到 libcudart —— 检查 CUDA 是否安装 (/usr/local/cuda)")
endif()
message(STATUS "libcudart   : ${CUDART_LIB}")

find_path(CUDA_INC cuda_runtime_api.h
  PATHS /usr/local/cuda/include $ENV{CUDA_HOME}/include)
if(NOT CUDA_INC)
  message(FATAL_ERROR "找不到 cuda_runtime_api.h")
endif()

# ---------------------------------------------------------------- TensorRT
# 允许用 -DTRT_ROOT=... 覆盖 (x86 上通常需要; Jetson 上 JetPack 装在 /usr)
if(TRT_ROOT)
  set(_trt_inc_hint ${TRT_ROOT}/include)
  set(_trt_lib_hint ${TRT_ROOT}/lib)
else()
  set(_trt_inc_hint "")
  set(_trt_lib_hint "")
endif()

find_path(TRT_INC NvInfer.h
  PATHS
    ${_trt_inc_hint}
    /usr/include/aarch64-linux-gnu     # JetPack 的头文件在这里!
    /usr/include/x86_64-linux-gnu
    /usr/include
)
find_library(TRT_LIB NAMES nvinfer
  PATHS
    ${_trt_lib_hint}
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT TRT_INC OR NOT TRT_LIB)
  message(FATAL_ERROR
    "找不到 TensorRT。Jetson 上应已随 JetPack 安装 (NvInfer.h 在 "
    "/usr/include/aarch64-linux-gnu)。x86 上请加 -DTRT_ROOT=/path/to/TensorRT-10.x")
endif()
message(STATUS "NvInfer.h   : ${TRT_INC}")
message(STATUS "libnvinfer  : ${TRT_LIB}")

# ---------------------------------------------------------------- 目标
add_executable(yolo_trt yolo_trt.cpp)
target_include_directories(yolo_trt PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${TRT_INC}
  ${CUDA_INC}
)
target_link_libraries(yolo_trt PRIVATE
  ${OpenCV_LIBS}
  ${TRT_LIB}
  ${CUDART_LIB}
  pthread
  dl
)
