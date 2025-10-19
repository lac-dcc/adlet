#pragma once

#include "../include/utils.hpp"
#include "taco.h"

class OpNode;
using OpNodePtr = std::shared_ptr<OpNode>;

class Tensor {
public:
  std::shared_ptr<taco::Tensor<float>> data;
  int numDims{};
  std::vector<SparsityVector> sparsities;
  const std::string name;
  std::vector<int> sizes;
  int numOps{0}; // number of operators this tensor belongs to as an operand
  bool outputTensor = false;

  std::vector<OpNodePtr> inputOps; // ops where this tensor is an input
  OpNodePtr outputOp;              // ops where this tensor is an input

  // constructor from sparsity vector (doesn't initialize tensor)
  Tensor(std::vector<int> sizes, std::vector<SparsityVector> sparsities,
         const std::string &n = "", const bool outputTensor = false);
  // constructor for empty output tensors
  Tensor(std::vector<int> sizes, const std::string &n = "");
  Tensor(std::vector<int> sizes, const std::string &n, taco::Format format);

  Tensor(std::vector<int> sizes, std::vector<float> sparsityRatios,
         const std::string &n = "",
         taco::Format format = {taco::Dense, taco::Dense});

  void create_data(const double threshold = 0.5);

  void create_data(taco::Format format);
  void initialize_data();
  void print_matrix();
  void print_full_sparsity();
  float get_sparsity_ratio();
  size_t get_nnz();
  size_t compute_size_in_bytes();
  void print_shape();
};
using TensorPtr = std::shared_ptr<Tensor>;
