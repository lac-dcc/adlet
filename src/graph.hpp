#pragma once

#include "taco.h"
#include "taco/format.h"
#include <bitset>
#include <cassert>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

constexpr int size = 2048;

using bitset = std::bitset<size>;

enum Direction { FORWARD, INTRA, BACKWARD };

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

void fill_tensor(taco::Tensor<float> &tensor, bitset rowSparsityVector,
                 bitset colSparsityVector, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        if (rowSparsityVector[i] == 0)
            continue;
        for (int j = 0; j < cols; ++j) {
            if (colSparsityVector[j] == 0)
                continue;
            float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            tensor.insert({i, j}, val);
        }
    }
    tensor.pack();
}

class Tensor {
public:
  std::shared_ptr<taco::Tensor<float>> data;
  bitset rowSparsity;
  bitset colSparsity;
  const std::string name;
  const int rows;
  const int cols;
  int numOps = 0; // number of operators this tensor belongs to as an operand
  bool outputTensor = false;

  // constructor from sparsity vector (doesn't initialize tensor)
  Tensor(int rows, int cols, bitset rowSparsity, bitset colSparsity, const std::string &n = "")
      : name(n), rows(rows), cols(cols), rowSparsity(rowSparsity), colSparsity(colSparsity) { }
  // constructor for empty output tensors
  Tensor(int rows, int cols, const std::string &n = "")
      : name(n), rows(rows), cols(cols) {
    rowSparsity.set(); // all rows initially active
    colSparsity.set(); // all cols initially active
  }

  Tensor(int rows, int cols, const std::string &n, taco::Format format)
      : data(std::make_shared<taco::Tensor<float>>(
            n, std::vector<int>{rows, cols}, format)),
        name(n), rows(rows), cols(cols) {
    rowSparsity.set();
    colSparsity.set();
  }

  Tensor(int rows, int cols, float rowSparsityRatio, float colSparsityRatio,
         const std::string &n = "",
         taco::Format format = {taco::Dense, taco::Dense})
      : data(std::make_shared<taco::Tensor<float>>(
            n, std::vector<int>{rows, cols}, format)),
        name(n), rows(rows), cols(cols) {
    // Initialize sparsity bitsets to 1 (active)
    rowSparsity.set(); // all rows initially active
    colSparsity.set(); // all cols initially active

    // Randomly deactivate some rows
    int zeroRowCount = static_cast<int>(rows * rowSparsityRatio);
    int zeroColCount = static_cast<int>(cols * colSparsityRatio);

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
        data->insert({i, j}, val);
      }
    }

    data->pack();
  }

  void create_data(taco::Format format) {
    data = std::make_shared<taco::Tensor<float>>(taco::Tensor<float>(name, {rows, cols}, format));
  }

  void print_tensor() {
    std::vector<std::vector<float>> tmp(rows, std::vector<float>(cols, 0.0));
    for (auto entry : *data) {
      tmp[entry.first[0]][entry.first[1]] = entry.second;
    }
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        std::cout << tmp[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  void print_full_sparsity() {
    for (int i = 0; i < rows; i++)
      if (rowSparsity[i] == 0)
        std::cout << std::string(cols, '0') << std::endl;
      else {
        for (int j = 0; j < cols; j++) {
          if (colSparsity[j] == 0)
            std::cout << '0';
          else
            std::cout << '1';
        }
        std::cout << std::endl;
      }
    std::cout << std::endl;
  }

  float get_row_sparsity_ratio() {
    int total = rows;
    return (float)(total -
                   static_cast<float>(count_bits(rowSparsity, rows)) / total);
  }

  float get_col_sparsity_ratio() {
    int total = cols;
    return (float)(total -
                   static_cast<float>(count_bits(colSparsity, cols)) / total);
  }

  int get_nnz() {
    int nnz = 0;
    for (int i = 0; i < rows; i++) {
      if (rowSparsity.test(i)) {
        for (int j = 0; j < cols; j++) {
          if (colSparsity.test(j)) {
            nnz += 1;
          }
        }
      }
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

using OpNodePtr = std::shared_ptr<OpNode>;

class MatMul : public OpNode {
public:
  MatMul(TensorPtr A, TensorPtr B, TensorPtr Out) {
    A->numOps++;
    B->numOps++;
    inputs = {A, B};
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
      output->rowSparsity &= inputs[0]->rowSparsity;
      output->colSparsity &= inputs[1]->colSparsity;
      break;
    case INTRA:
      if (inputs[1]->numOps == 1)
        inputs[1]->rowSparsity &= inputs[0]->colSparsity;
      if (inputs[0]->numOps == 1)
        inputs[0]->colSparsity &= inputs[1]->rowSparsity;
      break;
    case BACKWARD:
      if (inputs[1]->numOps == 1)
        inputs[1]->colSparsity &= output->colSparsity;
      if (inputs[0]->numOps == 1)
        inputs[0]->rowSparsity &= output->rowSparsity;
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
    inputs = std::move(inputs);
    output = Out;
    for (auto &input : inputs)
      input->numOps++;
  }

  void set_expression() override {
    taco::IndexVar i, j;
    for (auto &input : inputs)
      (*output->data)(i, j) += (*input->data)(i, j);
  }

  void propagate(Direction dir) override {
    if (dir == FORWARD) {
      for (auto input : inputs) {
        output->rowSparsity = input->rowSparsity;
        output->colSparsity = input->colSparsity;
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

class Transpose : public OpNode {
public:
  Transpose(TensorPtr In, TensorPtr Out) {
    In->numOps++;
    inputs = {In};
    output = Out;
  }

  void set_expression() override {
    taco::IndexVar i, j;
    (*output->data)(i, j) = (*inputs[0]->data)(j, i);
  }

  void propagate(Direction dir) override {
    if (dir == FORWARD) {
      output->rowSparsity &= inputs[0]->colSparsity;
      output->colSparsity &= inputs[0]->rowSparsity;
    } else if (dir == BACKWARD) {
      if (inputs[0]->numOps == 1) {
        inputs[0]->rowSparsity &= output->colSparsity;
        inputs[0]->colSparsity &= output->rowSparsity;
      }
    }
  }

  void print() override {
    std::cout << "->Transpose(" << inputs[0]->name << ",out=" << output->name
              << ")";
  }

  void print_sparsity() override {
    std::cout << "Transpose" << std::endl;
    inputs[0]->print_full_sparsity();
    std::cout << " = " << std::endl;
    output->print_full_sparsity();
    std::cout << std::endl;
  }
  std::string op_type() const override { return "Transpose"; }
};

class Graph {
  std::vector<OpNodePtr> nodes;

public:
  TensorPtr input, output;
  static Graph build_graph(TensorPtr in, TensorPtr out,
                           const std::vector<OpNodePtr> &ops) {
    Graph g;
    g.input = in;
    g.output = out;
    g.nodes = ops;
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
    std::cout << input->name;
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
