#pragma once

#include "../include/tensor.hpp"
#include <vector>

class OpNode {
public:
  std::vector<TensorPtr> inputs;
  TensorPtr output;
  virtual void set_expression() = 0;
  virtual void propagate(Direction dir) = 0;
  virtual void print() = 0;
  virtual void print_sparsity() = 0;
  virtual std::string op_type() const = 0;
  virtual void compute() = 0;
};

class Add : public OpNode {
public:
  Add(std::vector<TensorPtr> inputs, TensorPtr &Out);
  std::vector<std::shared_ptr<bitset>> get_input_bitsets(int inputDim);
  void set_expression() override;
  void propagate(Direction dir) override;
  void print() override;
  void print_sparsity() override;
  std::string op_type() const override;
  void compute() override;
};

using OpNodePtr = std::shared_ptr<OpNode>;

class Einsum : public OpNode {
public:
  std::string expression;
  std::string outputInds;
  std::vector<std::string> tensorIndicesVector;
  std::unordered_map<char, std::vector<std::pair<int, int>>> outputDims;
  std::unordered_map<char, std::vector<std::pair<int, int>>> reductionDims;

  Einsum(std::vector<TensorPtr> inputs, TensorPtr Out, std::string expression);

  std::vector<std::shared_ptr<bitset>> get_reduction_bitsets(char indexVar);

  std::vector<std::shared_ptr<bitset>> get_output_bitsets(char indexVar);
  char get_tensor_ind_var(TensorPtr tensor, int indDimension);

  int get_tensor_char_ind(TensorPtr tensor, char indexVar);
  void propagate_forward();
  bitset or_all_operands_add(Add *op, int inputInd, int inputDim);

  bitset and_all_operands_einsum(Einsum *einsumOp, int inputInd, int inputDim);
  bitset op_output_sparsity_einsum(Einsum *einsumOp, int inputInd,
                                   int inputDim);

  bitset propagate_intra_multiop(OpNodePtr op, int inputInd, int inputDim);

  bitset propagate_intra_dimension(int inputInd, int inputDim, char indexChar);
  void propagate_intra();
  bitset compute_multiop_einsum_sparsity(Einsum *opPtr, int inputInd,
                                         int inputDim);

  bitset compute_multiop_add_sparsity(Add *opPtr, int inputInd, int inputDim);
  bitset compute_multiop_sparsity(OpNode *opPtr, int inputInd, int inputDim);
  void propagate_backward();
  void set_expression() override;
  void propagate(Direction dir) override;
  void print() override;
  void print_sparsity() override;
  std::string op_type() const override;
  void compute() override;
};
