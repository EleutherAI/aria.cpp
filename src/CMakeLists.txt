
add_executable(aria main.cpp)
target_link_libraries(aria PRIVATE ggml common common-ggml)

add_executable(aria-quantize quantize.cpp)
target_link_libraries(aria-quantize PRIVATE ggml common common-ggml)
