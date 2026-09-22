cmake_minimum_required(VERSION 3.16)
project(rknn_yolo CXX)

# ============================================================================
#  RK3588 NPU C++ 部署
#
#  依赖 rknpu2:
#    头文件  rknn_api.h    (rknpu2/runtime/RK3588/Linux/librknn_api/include)
#    动态库  librknnrt.so  (rknpu2/runtime/RK3588/Linux/librknn_api/aarch64)
#
#  板子上一般已预装: 头文件 /usr/include/rknn_api.h, 库 /usr/lib/librknnrt.so
#  若没有, 从 https://github.com/airockchip/rknn-toolkit2 的 rknpu2 分支取。
#
#  编译 (在板子上, 或 aarch64 交叉编译):
#     mkdir build && cd build
#     cmake .. && make -j$(nproc)
#     ./rknn_yolo ../best_rk3588.rknn test.jpg out.jpg
#
#  若 rknpu2 不在默认位置:
#     cmake -DRKNPU_ROOT=/path/to/rknpu2/runtime/RK3588/Linux ..
# ============================================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ---------------------------------------------------------------- OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV      : ${OpenCV_VERSION}")

# ---------------------------------------------------------------- rknpu2
set(RKNPU_ROOT "$ENV{RKNPU_ROOT}" CACHE PATH "rknpu2 根目录 (含 librknn_api)")

find_path(RKNN_INC rknn_api.h
  PATHS
    ${RKNPU_ROOT}/librknn_api/include
    ${RKNPU_ROOT}/include
    /usr/include
    /usr/local/include
)
find_library(RKNN_LIB NAMES rknnrt
  PATHS
    ${RKNPU_ROOT}/librknn_api/aarch64
    ${RKNPU_ROOT}/librknn_api/armhf
    ${RKNPU_ROOT}/lib
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
)
if(NOT RKNN_INC OR NOT RKNN_LIB)
  message(FATAL_ERROR
    "找不到 rknn_api.h / librknnrt.so。\n"
    "  板子上一般预装在 /usr/include 与 /usr/lib 下;\n"
    "  若放在别处, 请用 -DRKNPU_ROOT=<rknpu2/runtime/RK3588/Linux> 指定")
endif()
message(STATUS "rknn_api.h  : ${RKNN_INC}")
message(STATUS "librknnrt  : ${RKNN_LIB}")

# ---------------------------------------------------------------- 目标
add_executable(rknn_yolo rknn_yolo.cpp)
target_include_directories(rknn_yolo PRIVATE ${OpenCV_INCLUDE_DIRS} ${RKNN_INC})
target_link_libraries(rknn_yolo PRIVATE ${OpenCV_LIBS} ${RKNN_LIB} pthread dl)

# 板端部署时把库一起带上 (可选)
install(TARGETS rknn_yolo RUNTIME DESTINATION bin)
