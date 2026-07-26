set(TARGET qnn-u16-core-debug)

add_executable(${TARGET} qnn_u16_core_debug.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/nlohmann)
target_link_libraries(${TARGET} PRIVATE llama ggml-cpu ggml-base)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
