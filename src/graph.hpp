#pragma once

#include "taco.h"
#include "taco/parser/einsum_parser.h"
#include "taco/format.h"
#include <bitset>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

constexpr int size = 2048;

using bitset = std::bitset<size>;

enum Direction { FORWARD, INTRA, BACKWARD };

class OpNode;
using OpNodePtr = std::shared_ptr<OpNode>;

int count_bits(bitset A, int pos) {
  int high_bits_to_eliminate = (size - 1) - (pos - 1);
  A <<= (high_bits_to_eliminate & (size - 1));
  return (A[size - 1] ? ~0ULL : 0) & A.count();
}

bitset generate_sparsity_vector(double sparsity, int size) {
  bitset sparsityVector;
  sparsityVector.set();

  int numZeros = static_cast<int>(size * sparsity);

  std::vector<int> indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(),
          std::mt19937{std::random_device{}()});

  for (int i = 0; i < numZeros; ++i)
    sparsityVector.set(indices[i], 0);

  return sparsityVector;
}

// should be used for creating non-adlet tensors for comparison
void fill_tensor(taco::Tensor<float> &tensor, double rowSparsityRatio,
                 double colSparsityRatio, int rows, int cols) {
  int zeroRowCount = static_cast<int>(rows * rowSparsityRatio);
  int zeroColCount = static_cast<int>(cols * colSparsityRatio);

  std::bitset<size> rowSparsity;
  std::bitset<size> colSparsity;
  rowSparsity.set();
  colSparsity.set();

  std::vector<int> rowIndices(rows), colIndices(cols);
  std::iota(rowIndices.begin(), rowIndices.end(), 0);
  std::iota(colIndices.begin(), colIndices.end(), 0);

  std::shuffle(rowIndices.begin(), rowIndices.end(),
               std::mt19937{std::random_device{}()});
  std::shuffle(colIndices.begin(), colIndices.end(),
               std::mt19937{std::random_device{}()});

  for (int i = 0; i < zeroRowCount; ++i)
    rowSparsity.set(rowIndices[i], 0);

  for (int j = 0; j < zeroColCount; ++j)
    colSparsity.set(colIndices[j], 0);

  for (int i = 0; i < rows; ++i) {
    if (!rowSparsity.test(i))
      continue; // skip zeroed row
    for (int j = 0; j < cols; ++j) {
      if (!colSparsity.test(j))
        continue; // skip zeroed col

      float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      tensor.insert({i, j}, val);
    }
  }

  tensor.pack();
}


class Tensor {
public:
  std::shared_ptr<taco::Tensor<float>> data;
  int numDims{ };
  std::vector<bitset> sparsities;
  const std::string name;
  std::vector<int> sizes;
  int numOps{ 0 }; // number of operators this tensor belongs to as an operand
  bool outputTensor = false;

  std::vector<OpNodePtr> inputOps; // ops where this tensor is an input

  // constructor from sparsity vector (doesn't initialize tensor)
  Tensor(std::vector<int> sizes, std::vector<bitset> sparsities, const std::string &n = "")
      : name(n), sizes(sizes), sparsities(sparsities) {
    numDims = sizes.size();
  }
  // constructor for empty output tensors
  Tensor(std::vector<int> sizes, const std::string &n = "")
      : name(n), sizes(sizes) {
    numDims = sizes.size();
    for (int i = 0; i < numDims; ++i) {
      sparsities.push_back(bitset());
      sparsities[i].set();
    }
  }

  Tensor(std::vector<int> sizes, const std::string &n, taco::Format format)
      : data(std::make_shared<taco::Tensor<float>>(n, sizes, format)),
        name(n), sizes(sizes) {
    numDims = sizes.size();
    for (int i = 0; i < numDims; ++i) {
      sparsities.push_back(bitset());
      sparsities[i].set();
    }
  }

  Tensor(std::vector<int> sizes, std::vector<float> sparsityRatios,
         const std::string &n = "",
         taco::Format format = {taco::Dense, taco::Dense})
      : data(std::make_shared<taco::Tensor<float>>(n, sizes, format)), 
        name(n), sizes(sizes) {
    numDims = sizes.size();
    // Initialize sparsity bitsets to 1 (active)
    for (int i = 0; i < numDims; ++i) {
      sparsities.push_back(bitset());
      sparsities[i].set();
    }

    // number of dimensions can vary: compute indices for each one
    for (int i = 0; i < numDims; ++i) {
      int zeroCount = static_cast<int>(sizes[i] * sparsityRatios[i]);

      std::vector<int> indices(sizes[i]);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(),
                   std::mt19937{std::random_device{}()});

      for (int j = 0; j < zeroCount; ++j)
        sparsities[i].set(indices[j], 0);
    }

    initialize_data();
  }

  void create_data(taco::Format format) {
    data = std::make_shared<taco::Tensor<float>>(taco::Tensor<float>(name, sizes, format));
  }

  std::vector<int> get_indices(std::vector<int> &dimSizes, int numElement) {
    int numDims = dimSizes.size();
    std::vector<int> indices(numDims);
    std::vector<int> cumulativeSize(numDims);
    cumulativeSize[0] = 1;

    for (int i = 1; i < numDims; ++i)
      cumulativeSize[i] = cumulativeSize[i - 1] * dimSizes[i - 1];

    for (int i = 0; i < numDims; ++i) {
      if (numElement < cumulativeSize[numDims - 1 - i])
        continue;
      indices[i] = numElement / cumulativeSize[numDims - 1 - i];
      numElement %= cumulativeSize[numDims - 1 - i];
    }

    return indices;
  }

  void initialize_data() {
    // number of dimensions can vary so compute num elements
    int numElements = 1;
    for (auto size : sizes)
      numElements *= size;

    for (int numElement = 0; numElement < numElements; ++numElement) {
      auto indices = get_indices(sizes, numElement);
      bool isZero = false;

      for (int i = 0; i < numDims; ++i) {
        if (sparsities[i][indices[i]] == 0) {
          isZero = true;
          break;
        }
      }
      if (isZero)
        continue;

      float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      data->insert(indices, val);
    }

    data->pack();
  }

  void print_matrix() {
    assert(numDims == 2 && "Tensor must be a matrix to call this method");
    std::vector<std::vector<float>> tmp(sizes[0], std::vector<float>(sizes[1], 0.0));
    for (auto entry : *data) {
      tmp[entry.first[0]][entry.first[1]] = entry.second;
    }
    for (int i = 0; i < sizes[0]; ++i) {
      for (int j = 0; j < sizes[1]; ++j) {
        std::cout << tmp[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  void print_full_sparsity() {
    for (int i = 0; i < numDims; ++i) {
      std::cout << "dim " << i << std::endl;
      for (int j = 0; j < sizes[i]; j++) {
        if (sparsities[i][j] == 0)
          std::cout << '0';
        else
          std::cout << '1';
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  float get_sparsity_ratio(int ind) {
    int total = sizes[ind];
    return (float)(total - static_cast<float>(count_bits(sparsities[ind], sizes[ind])) / total);
  }

  int get_nnz() {
    int nnz = 0;

    int numElements = 1;
    for (auto size : sizes)
      numElements *= size;

    for (int i = 0; i < numElements; ++i) {
      std::vector<int> index(numDims);
      bool zero = false;
      for (int j = 0; j < numDims; ++j) {
        // check if this index is sparse, skip if it is
        if (sparsities[j][i % sizes[j]] == 0) {
          zero = true;
          break;
        }
      }
      if (zero)
        continue;
      nnz++; // not sparse so increment
    }
    return nnz;
  }
};

using TensorPtr = std::shared_ptr<Tensor>;

class OpNode {
public:
  std::vector<TensorPtr> inputs;
  TensorPtr output;
  virtual void set_expression() = 0;
  virtual void propagate(Direction dir) = 0;
  virtual void print() = 0;
  virtual void print_sparsity() = 0;
  virtual std::string op_type() const = 0;
};

class MatMul : public OpNode {
public:
  MatMul(std::vector<TensorPtr> inputs, TensorPtr Out) {
    assert(inputs.size() == 2 && "MatMul requires 2 inputs!");
    assert(inputs[0]->numDims == 2 && inputs[1]->numDims == 2 && "Inputs must be matrices!");
    for (auto input : inputs) {
      input->numOps++;
    }
    this->inputs = inputs;
    output = Out;
    output->outputTensor = true;
  }

  void set_expression() override {
    taco::IndexVar i, j, k;
    (*output->data)(i, j) = (*inputs[0]->data)(i, k) * (*inputs[1]->data)(k, j);
  }

  void propagate(Direction dir) override {
    switch (dir) {
    case FORWARD:
      output->sparsities[0] &= inputs[0]->sparsities[0];
      output->sparsities[1] &= inputs[1]->sparsities[1];
      break;
    case INTRA:
      if (inputs[1]->numOps == 1)
        inputs[1]->sparsities[0] &= inputs[0]->sparsities[1];
      if (inputs[0]->numOps == 1)
        inputs[0]->sparsities[1] &= inputs[1]->sparsities[0];
      break;
    case BACKWARD:
      if (inputs[1]->numOps == 1)
        inputs[1]->sparsities[1] &= output->sparsities[1];
      if (inputs[0]->numOps == 1)
        inputs[0]->sparsities[0] &= output->sparsities[0];
      break;
    }
  }

  std::string op_type() const override { return "MatMul"; }

  void print() override {
    std::cout << "->Mul(" << inputs[0]->name << "," << inputs[1]->name
              << ",out=" << output->name << ")";
  }
  void print_sparsity() override {
    inputs[0]->print_full_sparsity();
    std::cout << "*" << std::endl;
    inputs[1]->print_full_sparsity();
    std::cout << " = " << std::endl;
    output->print_full_sparsity();
    std::cout << std::endl;
  }
};

class Add : public OpNode {
public:
  Add(std::vector<TensorPtr> &inputs, TensorPtr &Out) {
    this->inputs = inputs;
    output = Out;
    for (auto &input : inputs)
      input->numOps++;
  }

  void set_expression() override {
    std::vector<taco::IndexVar> inds(output->numDims);
    for (auto &input : inputs)
      (*output->data)(inds) += (*input->data)(inds);
  }

  void propagate(Direction dir) override {
    if (dir == FORWARD) {
      for (int dim = 0; dim < output->numDims; ++dim) {
        bitset inputSparsity;
        for (auto input : inputs)
           inputSparsity |= input->sparsities[dim];

        output->sparsities[dim] &= inputSparsity;
      }
    }
  }

  void print() override {
    std::cout << "->Add(";
    for (int i = 0; i < inputs.size(); ++i) {
      std::cout << inputs[i]->name;
      if (i != inputs.size() - 1)
          std::cout << ", ";
    }
    std::cout << ", out=" << output->name << ")";
  }

  void print_sparsity() override {
    for (int i = 0; i < inputs.size(); ++i) {
      inputs[i]->print_full_sparsity();
      if (i != inputs.size() - 1)
          std::cout << "+";
    }
    std::cout << " = " << std::endl;
    output->print_full_sparsity();
    std::cout << std::endl;
  }
  std::string op_type() const override { return "Add"; }
};

class Einsum : public OpNode {
public:
  std::string expression;
  std::string outputInds;
  std::vector<std::string> tensorIndicesVector;
  std::unordered_map<char, std::vector<std::pair<int, int>>> outputDims;
  std::unordered_map<char, std::vector<std::pair<int, int>>> reductionDims;
  Einsum(std::vector<TensorPtr> inputs, TensorPtr Out, std::string expression) {
    this->inputs = inputs;
    for (auto &input : inputs)
      input->numOps++;
    this->expression = expression;
    output = Out;

    int arrowPos = expression.find("->");
    std::string lhs = expression.substr(0, arrowPos);
    outputInds = expression.substr(arrowPos + 2);

    std::stringstream ss(lhs);
    std::string token;

    while (std::getline(ss, token, ','))
      tensorIndicesVector.push_back(token);

    for (char c : outputInds) {
      for (int i = 0; i < tensorIndicesVector.size(); ++i) {
        int pos = tensorIndicesVector[i].find(c);
        if (pos != std::string::npos)
          outputDims[c].push_back(std::make_pair(i, pos));
      }
    }

    for (int i = 0; i < tensorIndicesVector.size(); ++i) {
      for (int j = 0; j < tensorIndicesVector[i].size(); ++j) {
        char c = tensorIndicesVector[i][j];
        
        if (outputDims.find(c) != outputDims.end())
          continue;
        if (reductionDims.find(c) != reductionDims.end())
          continue;
        
        reductionDims[c].push_back(std::make_pair(i, j));
        
        for (int k = i + 1; k < tensorIndicesVector.size(); ++k) {
          int pos = tensorIndicesVector[k].find(c);
          if (pos != std::string::npos) {
            reductionDims[c].push_back(std::make_pair(k, pos));
          }
        }
      }
    }
  }

  void set_expression() override {
    std::vector<taco::TensorBase> tensors;
    for (auto input : inputs)
      tensors.push_back(*input->data);
    taco::Format format{ output->data->getStorage().getFormat() };
    taco::parser::EinsumParser parser(expression, tensors, format, taco::Datatype::Float32);
    parser.parse();
    output->data = std::make_shared<taco::Tensor<float>>(parser.getResultTensor());
  }

  void propagate_forward() {
    for (int i = 0; i < outputInds.length(); ++i) {
      bitset inputBitset;
      inputBitset.set();

      char c = outputInds[i];
      for (auto p : outputDims[c]) {
        int inputInd = p.first;       // which of the inputs
        int inputDim = p.second;      // which dimension
        inputBitset &= inputs[inputInd]->sparsities[inputDim];
      }
      output->sparsities[i] &= inputBitset;
    }
  }

  // op: pointer to Add
  // inputInd: the location of the input propagating to in THIS Einsum
  // inputDim: the dim of the input propagating to in THIS Einsum
  bitset or_all_operands_add(OpNodePtr op, int inputInd, int inputDim) {
    bitset inputBitset;
    for (auto input : op->inputs) { // go through ops in the addition and skip the current one
      if (input.get() == inputs[inputInd].get())
        continue;
      inputBitset |= input->sparsities[inputDim]; // all operands have the same dimensionality
    }
    return inputBitset;
  }

  // op: pointer to Einsum
  // inputInd: the location of the input propagating to in THIS Einsum
  // inputDim: the dim of the input propagating to in THIS Einsum
  bitset or_all_operands_einsum(OpNodePtr op, int inputInd, int inputDim) {
    bitset inputBitset;
    int currInd{ };
    char currChar{ };
    auto einsumOp = dynamic_cast<Einsum*> (op.get());
    // find the index of this operand in einsumOp, save into currInd
    // use currInd to get the char IndexVar of this tensor, save to currChar
    for (int i =0; i < op->inputs.size(); ++i) {
      if (op->inputs[i].get() != inputs[inputInd].get())
        continue;
      currInd = i;
      currChar = einsumOp->tensorIndicesVector[currInd][inputDim];
      break;
    }
    // std::cout << "currChar " << currChar << std::endl;
    // iterate all reduction vars corresponding to this char
    for (auto loc : einsumOp->reductionDims[currChar]) {
      if (einsumOp->inputs[loc.first].get() == inputs[inputInd].get())
        continue;
      // std::cout << loc.first << ", " << loc.second << "(" << einsumOp->inputs[loc.first]->name << ")" << std::endl;
      inputBitset |= einsumOp->inputs[loc.first]->sparsities[loc.second];
    }
    return inputBitset;
  }

  // returns op output sparsity for the corresponding dimension or empty set if not in output
  bitset op_output_sparsity_einsum(OpNodePtr op, int inputInd, int inputDim) {
    auto einsumOp = dynamic_cast<Einsum*> (op.get());
    char outputChar = '?';
    for (int i = 0; i < einsumOp->inputs.size(); ++i) {
      auto einsumInputTensor = einsumOp->inputs[i];
      if (einsumInputTensor.get() != inputs[inputInd].get())
        continue;
      outputChar = einsumOp->tensorIndicesVector[i][inputDim];
    }
    assert(outputChar != '?');
    int outputInd = -1;
    for (int i = 0; i < einsumOp->outputInds.length(); ++i) {
      if (einsumOp->outputInds[i] == outputChar) {
        outputInd = i;
        break;
      }
    }

    if (outputInd == -1)
      return bitset(); // not in the output: return empty bitset
    return einsumOp->output->sparsities[outputInd];
  }

  bitset propagate_intra_multiop(OpNodePtr op, int inputInd, int inputDim) {
    bitset inputBitset; // start off 0
    OpNode* opType = op.get();
    if (typeid(*opType) == typeid(Add)) {
      inputBitset |= or_all_operands_add(op, inputInd, inputDim);
    } else if (typeid(*opType) == typeid(Einsum)) {
      // std::cout << inputBitset[0] << inputBitset[1] << std::endl;
      // std::cout << "updating " << inputs[inputInd]->name << " dim " << inputDim << std::endl;
      inputBitset |= or_all_operands_einsum(op, inputInd, inputDim);
      // std::cout << inputBitset[0] << inputBitset[1] << std::endl;
      inputBitset |= op_output_sparsity_einsum(op, inputInd, inputDim);
    }
    return inputBitset;
  }

  bitset propagate_intra_dimension(char indexChar) {
    bitset inputBitset;
    inputBitset.set();

    for (auto inputLocation : reductionDims[indexChar]) {
      int inputInd = inputLocation.first;       // which of the inputs
      int inputDim = inputLocation.second;      // which dimension
      inputBitset &= inputs[inputInd]->sparsities[inputDim]; // handle the main case

      for (auto op : inputs[inputInd]->inputOps) {
        if (op.get() == this)
          continue;
        inputBitset |= propagate_intra_multiop(op, inputInd, inputDim);
      }
    }
    return inputBitset;
  }

  void propagate_intra() {
    for (auto kv : reductionDims) { // iterate over character: pair(inputInd, inputDim) map.
      bitset inputBitset = propagate_intra_dimension(kv.first);
      for (auto p : kv.second) {
        int inputInd = p.first;       // which of the inputs
        int inputDim = p.second;      // which dimension
        inputs[inputInd]->sparsities[inputDim] &= inputBitset;
      }
    }
  }

  void propagate_backward_dimension(int outputInd) {
    char indexChar = outputInds[outputInd];
    for (auto p : outputDims[indexChar]) {
      int inputInd = p.first;       // which of the inputs
      int inputDim = p.second;      // which dimension
      bitset inputBitset;
      for (auto op : inputs[inputInd]->inputOps) {
        if (op.get() == this)
          continue;
        OpNode* opType = op.get();
        if (typeid(*opType) == typeid(Add))
          inputBitset |= or_all_operands_add(op, inputInd, inputDim);
        else if (typeid(*opType) == typeid(Einsum)) {
          inputBitset |= or_all_operands_einsum(op, inputInd, inputDim);
          inputBitset |= op_output_sparsity_einsum(op, inputInd, inputDim);
        }
      }
      inputBitset |= output->sparsities[outputInd];
      inputs[inputInd]->sparsities[inputDim] &= inputBitset;
    }
  }

  void propagate_backward() {
    for (int i = 0; i < outputInds.length(); ++i) {
      propagate_backward_dimension(i);
    }
  }

  void propagate(Direction dir) override {
    switch (dir) {
    case FORWARD:
      propagate_forward();
      break;
    case INTRA:
      if (inputs.size() >= 2)
        propagate_intra();
      break;
    case BACKWARD:
      propagate_backward();
      break;
    }
  }

  void print() override {
    std::cout << "->Einsum[" << expression << "](";
    for (int i = 0; i < inputs.size(); ++i) {
      std::cout << inputs[i]->name;
      if (i != inputs.size() - 1)
          std::cout << ", ";
    }
    std::cout << ", out=" << output->name << ")";
  }

  void print_sparsity() override {
    for (int i = 0; i < inputs.size(); ++i) {
      inputs[i]->print_full_sparsity();
      if (i != inputs.size() - 1)
          std::cout << ",";
    }
    std::cout << " = " << std::endl;
    output->print_full_sparsity();
    std::cout << std::endl;
  }
  std::string op_type() const override { return "Einsum"; }
};

class Graph {
  std::vector<OpNodePtr> nodes;

public:
  std::vector<TensorPtr> inputs;
  TensorPtr output;
  static Graph build_graph(std::vector<TensorPtr> inputs, TensorPtr out,
                           const std::vector<OpNodePtr> &ops) {
    Graph g;
    g.inputs = inputs;
    g.output = out;
    g.nodes = ops;
    for (auto op : ops) {
      for (auto input : op->inputs) {
        input->inputOps.push_back(op);
      }
    }
    return g;
  }

  ~Graph() {}

  void run_propagation() {
    for (auto &op : nodes)
      op->propagate(Direction::FORWARD);
    for (auto &op : nodes)
      op->propagate(Direction::INTRA);
    for (auto &op : nodes)
      op->propagate(Direction::BACKWARD);
  }

  void assemble_expressions() {
    for (auto &op : nodes)
      op->set_expression();
  }

  void compile() {
    assemble_expressions();
    output->data->compile();
    output->data->assemble();
  }

  TensorPtr compute() {
    output->data->compute();
    return output;
  }

  void print() {
    for (auto &input : inputs) {
        std::cout << input->name << ",";
    }
    for (auto &op : nodes) {
      op->print();
    }
    std::cout << "->" << output->name << std::endl;
  }

  void print_sparsity() {
    for (auto &op : nodes) {
      op->print_sparsity();
    }
  }
};
