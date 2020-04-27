cmake_minimum_required(VERSION 3.9)

# set debug
set(CMAKE_VERBOSE_MAKEFILE ON)

# set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# collect all .h/.cpp files
file(GLOB CUDA_LITE_EMUL_SRCS *.cpp)
file(GLOB CUDA_LITE_EMUL_HEADERS *.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# collect kernel files from TOP/hammerblade
set(HB_KERNEL_PATH "../../../hammerblade/torch/kernel/")
file(GLOB KERNEL_SRCS ${HB_KERNEL_PATH}/*.cpp)
file(GLOB KERNEL_HEADERS ${HB_KERNEL_PATH}/*.hpp)
include_directories(${HB_KERNEL_PATH})
include_directories(${HB_KERNEL_PATH}/include/)

# generate library
add_library(bsg_manycore_runtime.1.0 SHARED ${CUDA_LITE_EMUL_HEADERS} ${CUDA_LITE_EMUL_SRCS} ${KERNEL_HEADERS} ${KERNEL_SRCS})

# compile options
target_compile_options(bsg_manycore_runtime.1.0 PRIVATE "-Wno-unused-variable")
target_compile_options(bsg_manycore_runtime.1.0 PRIVATE "-Wno-unused-function")
target_compile_options(bsg_manycore_runtime.1.0 PRIVATE "-Wno-unused-but-set-variable")

# notify the kernel that we are emulating
target_compile_definitions(bsg_manycore_runtime.1.0 PRIVATE "-DHB_EMUL")

# Include the nlohmann_json as a header-only library
target_include_directories(bsg_manycore_runtime.1.0
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}/third_party/include)

# Kernel call logging
if(HB_ENABLE_KERNEL_LOG)
  target_compile_definitions(bsg_manycore_runtime.1.0
                             PRIVATE
                             "-DHB_ENABLE_KERNEL_LOG")
endif()

# installation
install(TARGETS bsg_manycore_runtime.1.0 EXPORT Caffe2Targets DESTINATION lib)
