#pragma once

#include "taco.h"
#include "taco/format.h"
#include "taco/parser/einsum_parser.h"
#include "utils.hpp"
#include <bitset>
#include <cassert>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using bitset = std::bitset<size>;

enum Direction { FORWARD, INTRA, BACKWARD };

class OpNode;

using OpNodePtr = std::shared_ptr<OpNode>;

class Tensor {
public:
  std::shared_ptr<taco::Tensor<float>> data;
  int numDims{};
  std::vector<bitset> sparsities;
  const std::string name;
  std::vector<int> sizes;
  int numOps{0}; // number of operators this tensor belongs to as an operand
  bool outputTensor = false;

  std::vector<OpNodePtr> inputOps; // ops where this tensor is an input
  OpNodePtr outputOp;              // ops where this tensor is an input

  // constructor from sparsity vector (doesn't initialize tensor)
  Tensor(std::vector<int> sizes, std::vector<bitset> sparsities,
         const std::string &n = "")
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
      : data(std::make_shared<taco::Tensor<float>>(n, sizes, format)), name(n),
        sizes(sizes) {
    numDims = sizes.size();
    for (int i = 0; i < numDims; ++i) {
      sparsities.push_back(bitset());
      sparsities[i].set();
    }
  }

  Tensor(std::vector<int> sizes, std::vector<float> sparsityRatios,
         const std::string &n = "",
         taco::Format format = {taco::Dense, taco::Dense})
      : data(std::make_shared<taco::Tensor<float>>(n, sizes, format)), name(n),
        sizes(sizes) {
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
    this->data = std::make_shared<taco::Tensor<float>>(
        taco::Tensor<float>(this->name, this->sizes, format));
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
    std::vector<std::vector<float>> tmp(sizes[0],
                                        std::vector<float>(sizes[1], 0.0));
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

  float get_sparsity_ratio() {
    int total = 1;
    int nnz = 1;
    for (int dim = 0; dim < this->numDims; dim++) {
      int dimSize = this->sizes[dim];
      total *= dimSize;
      nnz *= count_bits(this->sparsities[dim], dimSize);
    }
    int zero_elements = total - nnz;
    return static_cast<float>(zero_elements) / total;
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
        if (!sparsities[j].test(i % sizes[j])) {
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
  virtual void compute() = 0;
};

class MatMul : public OpNode {
public:
  MatMul(std::vector<TensorPtr> inputs, TensorPtr Out) {
    assert(inputs.size() == 2 && "MatMul requires 2 inputs!");
    assert(inputs[0]->numDims == 2 && inputs[1]->numDims == 2 &&
           "Inputs must be matrices!");
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
    this->output->data->compile();
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

  void compute() override {
    for (auto &input : this->inputs) {
      std::cout << input->data->getName() << " ";
    }
    std::cout << std::endl;

    const auto startAssemble{std::chrono::steady_clock::now()};
    this->output->data->assemble();
    const auto startCompilation{std::chrono::steady_clock::now()};
    this->output->data->compute();
    const auto startRuntime{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> assembleSecs{startCompilation -
                                                     startAssemble};
    const std::chrono::duration<double> runtimeSecs{startRuntime -
                                                    startCompilation};
    std::cout << "assemble= " << assembleSecs.count() << std::endl;
    std::cout << "compute= " << runtimeSecs.count() << std::endl;
  }
};

class Add : public OpNode {
public:
  Add(std::vector<TensorPtr> inputs, TensorPtr &Out) {
    this->inputs = inputs;
    this->output = Out;
    for (auto &input : inputs)
      input->numOps++;
  }

  void set_expression() override {
    std::vector<taco::IndexVar> inds(output->numDims);
    for (auto &input : inputs)
      (*output->data)(inds) += (*input->data)(inds);
    this->output->data->compile();
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

  std::vector<std::shared_ptr<bitset>> get_input_bitsets(int inputDim) {
    std::vector<std::shared_ptr<bitset>> ret;
    for (auto input : inputs)
      ret.push_back(std::make_shared<bitset>(input->sparsities[inputDim]));
    return ret;
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

  void compute() override {
    for (auto &input : this->inputs) {
      std::cout << input->data->getName() << " ";
    }
    std::cout << std::endl;

    const auto startAssemble{std::chrono::steady_clock::now()};
    this->output->data->assemble();
    const auto startCompilation{std::chrono::steady_clock::now()};
    this->output->data->compute();
    const auto startRuntime{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> assembleSecs{startCompilation -
                                                     startAssemble};
    const std::chrono::duration<double> runtimeSecs{startRuntime -
                                                    startCompilation};
    std::cout << "assemble= " << assembleSecs.count() << std::endl;
    std::cout << "compute= " << runtimeSecs.count() << std::endl;
  }
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

  std::vector<std::shared_ptr<bitset>> get_reduction_bitsets(char indexVar) {
    std::vector<std::shared_ptr<bitset>> ret;
    for (auto tensorLoc : reductionDims[indexVar])
      ret.push_back(std::make_shared<bitset>(
          inputs[tensorLoc.first]->sparsities[tensorLoc.second]));

    return ret;
  }

  std::vector<std::shared_ptr<bitset>> get_output_bitsets(char indexVar) {
    std::vector<std::shared_ptr<bitset>> ret;
    for (auto tensorLoc : outputDims[indexVar])
      ret.push_back(std::make_shared<bitset>(
          inputs[tensorLoc.first]->sparsities[tensorLoc.second]));

    return ret;
  }

  char get_tensor_ind_var(TensorPtr tensor, int indDimension) {
    char ret{'?'};
    for (int i = 0; i < inputs.size(); ++i) {
      if (tensor.get() == inputs[i].get()) {
        ret = tensorIndicesVector[i][indDimension];
        break;
      }
    }
    assert(ret != '?' && "Tensor has to be an input to use this function!");
    return ret;
  }

  int get_tensor_char_ind(TensorPtr tensor, char indexVar) {
    int ind = -1;
    for (int i = 0; i < outputInds.length(); ++i) {
      if (outputInds[i] == indexVar) {
        ind = i;
        break;
      }
    }
    return ind;
  }

  void set_expression() override {
    std::vector<taco::TensorBase> tensors;
    for (auto input : inputs)
      tensors.push_back(*input->data);
    taco::Format format{output->data->getStorage().getFormat()};
    taco::parser::EinsumParser parser(expression, tensors, format,
                                      taco::Datatype::Float32);
    parser.parse();
    std::string name = output->data->getName();
    output->data =
        std::make_shared<taco::Tensor<float>>(parser.getResultTensor());
    output->data->setName(name);
    this->output->data->compile();
  }

  void propagate_forward() {
    for (int i = 0; i < outputInds.length(); ++i) {
      bitset inputBitset;
      inputBitset.set();

      char c = outputInds[i];
      for (auto p : outputDims[c]) {
        int inputInd = p.first;  // which of the inputs
        int inputDim = p.second; // which dimension
        inputBitset &= inputs[inputInd]->sparsities[inputDim];
      }
      output->sparsities[i] &= inputBitset;
    }
  }

  // op: pointer to Add
  // inputInd: the location of the input propagating to in THIS Einsum
  // inputDim: the dim of the input propagating to in THIS Einsum
  bitset or_all_operands_add(Add *op, int inputInd, int inputDim) {
    bitset inputBitset;
    for (auto input : op->inputs) { // go through ops in the addition and skip
                                    // the current one
      if (input.get() == inputs[inputInd].get())
        continue;
      inputBitset |= input->sparsities[inputDim]; // all operands have the
                                                  // same dimensionality
    }
    return inputBitset;
  }

  // inputInd: the location of the input propagating to in THIS Einsum
  // inputDim: the dim of the input propagating to in THIS Einsum
  bitset and_all_operands_einsum(Einsum *einsumOp, int inputInd, int inputDim) {
    bitset inputBitset;
    inputBitset.set();
    int currInd{};
    char currChar{};
    // find the index of this operand in einsumOp, save into currInd
    // use currInd to get the char IndexVar of this tensor, save to currChar
    for (int i = 0; i < einsumOp->inputs.size(); ++i) {
      if (einsumOp->inputs[i].get() != inputs[inputInd].get())
        continue;
      currInd = i;
      currChar = einsumOp->tensorIndicesVector[currInd][inputDim];
      break;
    }
    // iterate all reduction vars corresponding to this char
    for (auto loc : einsumOp->reductionDims[currChar]) {
      if (einsumOp->inputs[loc.first].get() == inputs[inputInd].get())
        continue;
      // einsumOp->inputs[loc.first]->name << ")" << std::endl;
      inputBitset &= einsumOp->inputs[loc.first]->sparsities[loc.second];
    }
    return inputBitset;
  }

  // returns op output sparsity for the corresponding dimension or empty set
  // if not in output
  bitset op_output_sparsity_einsum(Einsum *einsumOp, int inputInd,
                                   int inputDim) {
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
    OpNode *opPtr = op.get();
    if (typeid(*opPtr) == typeid(Add)) {
      Add *addPtr = dynamic_cast<Add *>(opPtr);
      auto addBitsets = addPtr->get_input_bitsets(inputDim);
      for (int i = 0; i < addBitsets.size(); ++i) {
        if (i == inputInd)
          continue;
        inputBitset |= *addBitsets[i];
      }
    } else if (typeid(*opPtr) == typeid(Einsum)) {
      Einsum *einsumOp = dynamic_cast<Einsum *>(op.get());
      inputBitset |= and_all_operands_einsum(einsumOp, inputInd, inputDim);
      inputBitset |= op_output_sparsity_einsum(einsumOp, inputInd, inputDim);
    }
    return inputBitset;
  }

  bitset propagate_intra_dimension(int inputInd, int inputDim, char indexChar) {
    bitset inputBitset;

    for (auto op : inputs[inputInd]->inputOps) {
      auto opPtr = op.get();
      inputBitset |= compute_multiop_sparsity(opPtr, inputInd, inputDim);
      ;
    }

    return inputBitset;
  }

  void propagate_intra() {
    for (auto kv : reductionDims) { // iterate over character: pair(inputInd,
                                    // inputDim) map.
      for (auto p : kv.second) {
        int inputInd = p.first;  // which of the inputs
        int inputDim = p.second; // which dimension
        inputs[inputInd]->sparsities[inputDim] &=
            propagate_intra_dimension(inputInd, inputDim, kv.first);
      }
    }
  }

  bitset compute_multiop_einsum_sparsity(Einsum *opPtr, int inputInd,
                                         int inputDim) {
    char indexVar = opPtr->tensorIndicesVector[inputInd][inputDim];
    bitset inputBitset;

    if (opPtr->outputDims.find(indexVar) != opPtr->outputDims.end()) {
      int ind = get_tensor_char_ind(opPtr->output, indexVar);
      inputBitset = opPtr->output->sparsities[ind];
    } else {
      assert(opPtr->reductionDims.find(indexVar) != opPtr->reductionDims.end());
      auto pairs = opPtr->reductionDims[indexVar];
      inputBitset.set();
      for (auto p : pairs) {
        int otherInputInd = p.first;  // which of the inputs
        int otherInputDim = p.second; // which dimension
        inputBitset &= opPtr->inputs[otherInputInd]->sparsities[otherInputDim];
      }
    }

    return inputBitset;
  }

  bitset compute_multiop_add_sparsity(Add *opPtr, int inputInd, int inputDim) {
    return opPtr->output
        ->sparsities[inputDim]; // addition can only have output dimension
  }

  bitset compute_multiop_sparsity(OpNode *opPtr, int inputInd, int inputDim) {
    bitset inputBitset;

    if (typeid(*opPtr) == typeid(Add)) {
      Add *addPtr = dynamic_cast<Add *>(opPtr);
      inputBitset = compute_multiop_add_sparsity(addPtr, inputInd, inputDim);
    } else if (typeid(*opPtr) == typeid(Einsum)) {
      Einsum *einPtr = dynamic_cast<Einsum *>(opPtr);
      inputBitset = compute_multiop_einsum_sparsity(einPtr, inputInd, inputDim);
    }

    return inputBitset;
  }

  void propagate_backward_dimension(int outputInd) {
    // char indexChar = outputInds[outputInd];
    // for (auto p : outputDims[indexChar]) {
    //   int inputInd = p.first;  // which of the inputs
    //   int inputDim = p.second; // which dimension
    //   bitset inputBitset;
    //   for (auto op : inputs[inputInd]->inputOps) {
    //     OpNode *opPtr = op.get();
    //     if (opPtr == this)
    //       continue;
    //     // now handle the case where this operand is used in other ops.
    //     // if any other corresponding reduction or output dims are nonsparse
    //     // we OR it
    //     inputBitset |= get_multiop_sparsity(opPtr, inputInd, inputDim);
    //   }
    //   inputBitset |= output->sparsities[outputInd];
    //   inputs[inputInd]->sparsities[inputDim] &= inputBitset;
    // }
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

  void compute() override {
    for (auto &input : this->inputs) {
      std::cout << input->data->getName() << "(";
      std::cout << count_bits(input->sparsities[0], size) << ","
                << count_bits(input->sparsities[1], size) << ")("
                << input->get_sparsity_ratio() << ")";
    }
    std::cout << std::endl;

    const auto startAssemble{std::chrono::steady_clock::now()};
    this->output->data->assemble();
    const auto startCompilation{std::chrono::steady_clock::now()};
    this->output->data->compute();
    const auto startRuntime{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> assembleSecs{startCompilation -
                                                     startAssemble};
    const std::chrono::duration<double> runtimeSecs{startRuntime -
                                                    startCompilation};
    std::cout << "assemble= " << assembleSecs.count() << std::endl;
    std::cout << "compute= " << runtimeSecs.count() << std::endl;
    std::cout << std::endl;
  }
};

class Graph {

public:
  std::vector<OpNodePtr> nodes;
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
      op->output->outputOp = op;
    }
    return g;
  }

  ~Graph() {}

  void run_propagation() {
    run_propagation(Direction::FORWARD);
    run_propagation(Direction::INTRA);
    run_propagation(Direction::INTRA);
    // run_propagation(Direction::BACKWARD);
  }

  void run_propagation(Direction dir) {
    if (dir == INTRA || dir == BACKWARD) {
      std::vector<OpNodePtr> intraStack{output->outputOp};
      std::unordered_map<TensorPtr, bool> doneProp;
      while (intraStack.size() > 0) {
        auto op = intraStack.back();
        intraStack.pop_back();
        op->propagate(dir);
        for (auto input : op->inputs) {
          doneProp[input] = true;

          if (!input->outputOp)
            continue;
          bool allDone{false};
          for (auto otherInput : input->outputOp->inputs) {
            for (auto otherOp : otherInput->inputOps)
              allDone |= doneProp[otherOp->output];
          }
          if (allDone)
            intraStack.push_back(input->outputOp);
        }
      }
    } else {
      for (auto &op : nodes)
        op->propagate(dir);
    }
  }

  void assemble_expressions() {
    for (auto &op : nodes)
      op->set_expression();
  }

  void compile() {
    assemble_expressions();
    output->data->compile();
  }

  TensorPtr compute() {
    for (auto &op : nodes)
      op->compute();
    /*output->data->assemble();*/
    /*output->data->compute();*/
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

  float get_sparsity_ratio() {
    int count = 0;
    float total_ratio = 0;
    for (auto &ops : this->nodes) {
      for (auto &input : ops->inputs) {
        count++;
        total_ratio += input->get_sparsity_ratio();
      }
    }
    total_ratio += this->output->get_sparsity_ratio();
    count++;
    return total_ratio / count;
  }
};
