#include "taco.h"
#include "graph.hpp"
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

int main() {
  auto X = std::make_shared<Tensor>(8, 4, 0.5, 0.5, "X");
  auto W1 = std::make_shared<Tensor>(4, 4, 0.0, 0.0, "W1");
  auto O1 = std::make_shared<Tensor>(8, 4,"O1");
  auto W2 = std::make_shared<Tensor>(4, 4, 0.0, 0.0, "W2");
  auto O2 = std::make_shared<Tensor>(8, 4, "O2");
  auto Y = std::make_shared<Tensor>(8, 4, "Y");
  auto Y_T = std::make_shared<Tensor>(4, 8,"Y_T");

  auto g = Graph::build_graph(X, Y_T,
                              {std::make_shared<MatMul>(X, W1, O1),
                               std::make_shared<MatMul>(O1, W2, O2),
                               std::make_shared<Relu>(O2, Y),
                               std::make_shared<Transpose>(Y, Y_T)});

  run_graph_with_logging(g);
}
