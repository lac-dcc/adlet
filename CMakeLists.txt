cmake_minimum_required(VERSION 3.15...4.0)

project(SparseInference VERSION 1.0
    LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_executable(sparseinf src/main.cpp src/graph.hpp)
add_executable(tests src/tests.cpp src/graph.hpp)
target_link_directories(sparseinf PRIVATE ../taco/build/lib)
target_include_directories(sparseinf PRIVATE ../taco/include)
target_link_directories(tests PRIVATE ../taco/build/lib)
target_include_directories(tests PRIVATE ../taco/include)

target_link_libraries(sparseinf taco)
target_link_libraries(tests taco)
