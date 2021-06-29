cmake_minimum_required(VERSION 3.14)
include_directories(./)

file(GLOB SRC_FILES "./ctcBeamSearchDecoderCustom.cpp")

add_compile_options(-std=c++11)
add_library(ctc_beam_search_decoder SHARED ${SRC_FILES})

target_include_directories(ctc_beam_search_decoder PUBLIC
    /usr/include/x86_64-linux-gnu
    /build/python-tensorrt/plugin/common
    /usr/local/cuda-11.2/targets/x86_64-linux/include
    /usr/local/cuda/include)

target_link_directories(ctc_beam_search_decoder PUBLIC /usr/local/cuda-11.2/targets/x86_64-linux/lib)

target_link_libraries(ctc_beam_search_decoder dl pthread rt)
target_link_libraries(ctc_beam_search_decoder
    nvparsers_static
    nvinfer_static
    nvinfer_plugin_static
    stdc++
    cudnn_static
    culibos
    cudart_static
    cublas_static)
