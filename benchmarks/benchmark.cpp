#include "benchmark_einsum.hpp"
#include "benchmark_format.hpp"
#include "benchmark_graph.hpp"
#include "benchmark_proptime.hpp"
#include <iostream>

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <benchmark name> " << std::endl;
    return 1;
  }

  std::string benchmark = argv[1];

  if (benchmark == "graph") {
    return benchmark_graph(argc, argv);
  } else if (benchmark == "format") {
    return parseArguments(argc, argv);
  } else if (benchmark == "einsum") {
    return benchmark_einsum(argc, argv);
  } else if (benchmark == "proptime") {
    benchmark_proptime();
  } else {
    std::cerr << "Error: unknown benchmark" << std::endl;
  }

  return 0;
}
