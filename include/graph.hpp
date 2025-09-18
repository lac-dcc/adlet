#pragma once

#include "../include/node.hpp"

class Graph {

public:
  std::vector<OpNodePtr> nodes;
  std::vector<TensorPtr> inputs;
  TensorPtr output;
  static Graph build_graph(std::vector<TensorPtr> inputs, TensorPtr out,
                           const std::vector<OpNodePtr> &ops);

  ~Graph() = default;

  void run_propagation();
  void run_propagation(Direction dir);

  void assemble_expressions();
  void compile();
  TensorPtr compute();
  void print();
  void print_sparsity();
  float get_sparsity_ratio();
  void get_tensor_sizes();
};
