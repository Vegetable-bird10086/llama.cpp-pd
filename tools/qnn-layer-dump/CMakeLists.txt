set(TARGET llama-qnn-layer-dump)

add_executable(${TARGET} qnn_layer_dump.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
