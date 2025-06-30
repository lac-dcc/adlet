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

#ifndef GRAPH
#define GRAPH

#define MAX_SIZE 2048

typedef std::bitset<MAX_SIZE> bitset;

enum Direction { FORWARD, INTRA, BACKWARD };

int count_bits(bitset A, int pos) {
  int high_bits_to_eliminate = (MAX_SIZE - 1) - (pos - 1);
  A <<= (high_bits_to_eliminate & (MAX_SIZE - 1));
  return (A[MAX_SIZE - 1] ? ~0ULL : 0) & A.count();
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
  bool pruneRow = false;
  bool pruneCol = false;
  bool outputTensor = false;

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
    *data = taco::Tensor<float>(name, {rows, cols}, format);
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

  float get_sparsity_ratio() {
    int total = cols * rows;
    return (float)(total - get_nnz()) / total;
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

  void prune() {
    bool prunedRows = false, prunedCols = false;
    if (!outputTensor) {
      if (pruneRow) {
        for (int i = 0; i < rows; i++)
          if (!rowSparsity.test(i))
            for (int col = 0; col < cols; col++)
              data->insert({i, col}, -data->at({i, col}));
        prunedRows = true;
      }
      if (pruneCol) {
        for (int i = 0; i < cols; i++)
          if (!colSparsity.test(i))
            for (int row = 0; row < rows; row++)
              data->insert({row, i}, -data->at({row, i}));
        prunedCols = true;
      }
    }
    // taco::Format format = getFormat();
    taco::Format sparse({taco::Dense, taco::Sparse});

    if (prunedCols || prunedRows) {
      auto newTensor = data->removeExplicitZeros(sparse);
      newTensor.setName(data->getName() + "_pruned");
      newTensor.pack();
      *data = newTensor;
    } else if (outputTensor && (pruneCol || pruneRow)) {
      auto newTensor = taco::Tensor<float>({rows, cols}, sparse);
      newTensor.setName(data->getName() + "_pruned");
      newTensor.pack();
      *data = newTensor;
    }
    pruneRow = false;
    pruneCol = false;
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
  virtual float get_sparsity_ratio() = 0;
  virtual std::string op_type() const = 0;
  virtual void prune() = 0;
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

  float get_sparsity_ratio() override {
    float a_zeros = inputs[0]->get_sparsity_ratio();
    float b_zeros = inputs[1]->get_sparsity_ratio();
    return (a_zeros + b_zeros) / 2;
  }

  void propagate(Direction dir) override {
    auto oldOutRow = output->rowSparsity;
    auto oldOutCol = output->colSparsity;
    auto oldLeftRow = inputs[0]->rowSparsity;
    auto oldLeftCol = inputs[0]->colSparsity;
    auto oldRightRow = inputs[1]->rowSparsity;
    auto oldRightCol = inputs[1]->colSparsity;

    auto detectChange = [](auto &before, auto &after, bool &pruneFlag,
                           Direction &dir) {
      if ((before ^ after).any()) {
        pruneFlag = true & !(dir == FORWARD);
      }
    };

    switch (dir) {
    case FORWARD:
      output->rowSparsity &= inputs[0]->rowSparsity;
      output->colSparsity &= inputs[1]->colSparsity;
      detectChange(oldOutCol, output->colSparsity, output->pruneCol, dir);
      detectChange(oldOutRow, output->rowSparsity, output->pruneRow, dir);
      break;

    case INTRA:
      if (inputs[1]->numOps == 1)
        inputs[1]->rowSparsity &= inputs[0]->colSparsity;
      if (inputs[0]->numOps == 1)
        inputs[0]->colSparsity &= inputs[1]->rowSparsity;
      detectChange(oldLeftCol, inputs[0]->colSparsity, inputs[0]->pruneCol,
                   dir);
      detectChange(oldRightRow, inputs[1]->rowSparsity, inputs[1]->pruneRow,
                   dir);
      break;

    case BACKWARD:
      if (inputs[1]->numOps == 1)
        inputs[1]->colSparsity &= output->colSparsity;
      if (inputs[0]->numOps == 1)
        inputs[0]->rowSparsity &= output->rowSparsity;
      detectChange(oldLeftRow, inputs[0]->rowSparsity, inputs[0]->pruneRow,
                   dir);
      detectChange(oldRightCol, inputs[1]->colSparsity, inputs[1]->pruneCol,
                   dir);
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
  void prune() override {
    inputs[0]->prune();
    inputs[1]->prune();
    output->prune();
  }
};

class Add : public OpNode {
public:
  Add(TensorPtr A, TensorPtr B, TensorPtr Out) {
    A->numOps++;
    B->numOps++;
    inputs = {A, B};
    output = Out;
  }

  void set_expression() override {
    taco::IndexVar i, j;
    (*output->data)(i, j) = (*inputs[0]->data)(i, j) + (*inputs[1]->data)(i, j);
  }

  float get_sparsity_ratio() override {
    float a_ratio = inputs[0]->get_sparsity_ratio();
    float b_ratio = inputs[1]->get_sparsity_ratio();
    return (a_ratio + b_ratio) / 2;
  }

  void propagate(Direction dir) override {
    auto initRowSparsity = output->rowSparsity;
    auto initColSparsity = output->colSparsity;
    if (dir == FORWARD) {
      output->rowSparsity = inputs[0]->rowSparsity | inputs[1]->rowSparsity;
      output->colSparsity = inputs[0]->colSparsity | inputs[1]->colSparsity;
    }
    auto colChange = initColSparsity ^ output->colSparsity;
    auto rowChange = initRowSparsity ^ output->rowSparsity;
  }

  void print() override {
    std::cout << "->Add(" << inputs[0]->name << "," << inputs[1]->name
              << ",out=" << output->name << ")";
  }

  void print_sparsity() override {
    inputs[0]->print_full_sparsity();
    std::cout << "+" << std::endl;
    inputs[1]->print_full_sparsity();
    std::cout << " = " << std::endl;
    output->print_full_sparsity();
    std::cout << std::endl;
  }
  std::string op_type() const override { return "Add"; }

  void prune() override {
    inputs[0]->prune();
    inputs[1]->prune();
    output->prune();
  }
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

  float get_sparsity_ratio() override {
    return inputs[0]->get_sparsity_ratio();
  }

  void propagate(Direction dir) override {
    auto initRowSparsity = output->rowSparsity;
    auto initColSparsity = output->colSparsity;
    if (dir == FORWARD) {
      output->rowSparsity &= inputs[0]->colSparsity;
      output->colSparsity &= inputs[0]->rowSparsity;
    } else if (dir == BACKWARD) {
      if (inputs[0]->numOps == 1) {
        inputs[0]->rowSparsity &= output->colSparsity;
        inputs[0]->colSparsity &= output->rowSparsity;
      }
    }
    auto colChange = initColSparsity ^ output->colSparsity;
    auto rowChange = initRowSparsity ^ output->rowSparsity;
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

  void prune() override {
    inputs[0]->prune();
    output->prune();
  }
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

  float get_sparsity_ratio() {
    float sparsity = 0.0;
    for (auto &op : nodes) {
      sparsity += op->get_sparsity_ratio();
    }
    return (float)(sparsity / nodes.size());
  }

  void run_analysis() {
    for (auto &op : nodes)
      op->propagate(Direction::FORWARD);
    for (auto &op : nodes)
      op->propagate(Direction::INTRA);
    for (auto &op : nodes)
      op->propagate(Direction::BACKWARD);
  }

  void run_propagation() {
    for (auto &op : nodes)
      op->prune();
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
#endif
