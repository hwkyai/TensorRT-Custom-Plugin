cmake_minimum_required(VERSION 3.14)
include_directories(./)

file(GLOB SRC_FILES "./CtcBeamSearchDecoderCustom.cpp")

add_compile_options(-std=c++11)
add_library(ctc_beam_search_decoder SHARED ${SRC_FILES})

target_include_directories(ctc_beam_search_decoder PUBLIC /usr/include/x86_64-linux-gnu)
target_include_directories(ctc_beam_search_decoder PUBLIC /build/python-tensorrt/plugin/common)
target_include_directories(ctc_beam_search_decoder PUBLIC /usr/local/cuda-11.2/targets/x86_64-linux/include)
target_include_directories(ctc_beam_search_decoder PUBLIC /usr/local/cuda/include)

target_link_libraries(ctc_beam_search_decoder -ldl -lpthread -lrt)
target_link_libraries(ctc_beam_search_decoder /usr/lib/x86_64-linux-gnu/libnvparsers_static.a)
target_link_libraries(ctc_beam_search_decoder /usr/lib/x86_64-linux-gnu/libnvinfer_static.a)
target_link_libraries(ctc_beam_search_decoder /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a)
target_link_libraries(ctc_beam_search_decoder /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
target_link_libraries(ctc_beam_search_decoder /usr/lib/x86_64-linux-gnu/libcudnn_static.a /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas_static.a)
target_link_libraries(ctc_beam_search_decoder /usr/local/cuda-11.2/targets/x86_64-linux/lib/libculibos.a)
target_link_libraries(ctc_beam_search_decoder /usr/lib/x86_64-linux-gnu/libcudart.a)

