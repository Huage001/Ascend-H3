cmake_minimum_required(VERSION 3.16)
project(spark_joint_v4_ext LANGUAGES C CXX)

set(LINUX TRUE)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(SOC_VERSION "Ascend910B1" CACHE STRING "Ascend SoC used to compile kernels")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH
        "CANN installation directory" FORCE)
endif()

if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
else()
    message(FATAL_ERROR "Cannot locate ascendc_kernel_cmake; source CANN set_env.sh first")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

set(ALL_HOST_SRCS "")
set(ALL_KERNEL_SRCS "")
foreach(OP_DIR spark_joint_v4)
    unset(OP_NAME)
    unset(OP_HOST_SRCS)
    unset(OP_KERNEL_SRCS)
    include("${PROJECT_SOURCE_DIR}/csrc/${OP_DIR}/CMakeLists.txt")
    list(APPEND ALL_HOST_SRCS ${OP_HOST_SRCS})
    list(APPEND ALL_KERNEL_SRCS ${OP_KERNEL_SRCS})
    message(STATUS "Adding ${OP_NAME}")
endforeach()

add_compile_options(-Wno-deprecated-declarations -fPIC)

ascendc_library(spark_joint_v4_kernels STATIC ${ALL_KERNEL_SRCS})
ascendc_compile_definitions(spark_joint_v4_kernels PRIVATE
    -DHAVE_WORKSPACE -DHAVE_TILING)

add_library(spark_joint_v4_ext SHARED ${ALL_HOST_SRCS})
set_target_properties(spark_joint_v4_ext PROPERTIES
    OUTPUT_NAME spark_joint_v4
    POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(spark_joint_v4_ext PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
target_include_directories(spark_joint_v4_ext PRIVATE
    "${TORCH_NPU_PATH}/include"
    "${TORCH_PATH}/include"
    "${TORCH_PATH}/include/torch/csrc/api/include"
    "${PROJECT_SOURCE_DIR}/csrc")
target_link_directories(spark_joint_v4_ext PRIVATE
    "${TORCH_PATH}/lib"
    "${TORCH_NPU_PATH}/lib")
target_link_libraries(spark_joint_v4_ext PRIVATE
    spark_joint_v4_kernels
    torch_npu
    ascendcl
    tiling_api
    register
    platform
    ascendalog
    dl)
