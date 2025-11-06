cmake_minimum_required(VERSION 3.15...4.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(adlet VERSION 1.0
    LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_library(adlet_lib
    src/graph.cpp
    src/node.cpp
    src/tensor.cpp
    src/einsum.cpp
    src/utils.cpp
    src/viz.cpp
)

target_include_directories(adlet_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ../taco/include
)


add_executable(adlet src/main.cpp include/)
add_executable(tests src/tests.cpp include/)
add_executable(benchmark
    benchmarks/benchmark.cpp
    benchmarks/benchmark_graph.cpp
    benchmarks/benchmark_einsum.cpp
    benchmarks/benchmark_format.cpp
    benchmarks/benchmark_proptime.cpp
)

target_include_directories(adlet_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(adlet PRIVATE adlet_lib taco)
target_link_libraries(tests PRIVATE adlet_lib taco)
target_link_libraries(benchmark PRIVATE adlet_lib taco)


target_include_directories(adlet PRIVATE ../taco/include)
target_include_directories(tests PRIVATE ../taco/include)
target_include_directories(benchmark PRIVATE ../taco/include)

target_link_directories(adlet PRIVATE ../taco/build/lib)
target_link_directories(tests PRIVATE ../taco/build/lib)
target_link_directories(benchmark PRIVATE ../taco/build/lib)

set(SIZE_MACRO "2048" CACHE STRING "Macro to set bit_set size to")
add_compile_definitions(SIZE_MACRO=${SIZE_MACRO})
