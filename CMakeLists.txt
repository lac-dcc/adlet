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

add_executable(adlet src/main.cpp src/graph.hpp)
add_executable(tests src/tests.cpp src/graph.hpp)
add_executable(benchmark benchmarks/main.cpp src/graph.hpp)

target_link_directories(benchmark PRIVATE ../taco/build/lib)
target_include_directories(benchmark PRIVATE ../taco/include)
target_link_directories(benchmark PRIVATE ../papi/src/install/lib)
target_include_directories(benchmark PRIVATE ../papi/src/install/include)

target_link_directories(adlet PRIVATE ../taco/build/lib)
target_include_directories(adlet PRIVATE ../taco/include)
target_link_directories(adlet PRIVATE ../papi/src/install/lib)
target_include_directories(adlet PRIVATE ../papi/src/install/include)

target_link_directories(tests PRIVATE ../taco/build/lib)
target_include_directories(tests PRIVATE ../taco/include)
target_link_directories(tests PRIVATE ../papi/src/install/lib)
target_include_directories(tests PRIVATE ../papi/src/install/include)

target_link_libraries(adlet taco)
target_link_libraries(tests taco)
target_link_libraries(benchmark taco)
target_link_libraries(adlet papi)
target_link_libraries(tests papi)
target_link_libraries(benchmark papi)
