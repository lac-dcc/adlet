#include "taco.h"
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#define MAX_DIM 1024

enum Direction { FORWARD, BACKWARD };

class Tensor {
public:
  taco::Tensor<float> data;
  std::bitset<MAX_DIM> rowSparsity;
  std::bitset<MAX_DIM> colSparsity;
  const std::string name;
  const unsigned int rows;
  const unsigned int cols;

  Tensor(int rows, int cols, bool random = false, const std::string &n = "")
      : data(n, {rows, cols}, taco::Format{taco::Dense, taco::Dense}), name(n),
        rows(rows), cols(cols) {
    if (random) {
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          data.insert({i, j}, static_cast<float>(rand()) /
                                  static_cast<float>(RAND_MAX));
      data.pack();
    }
  }

  Tensor(int rows, int cols, float rowSparsityRatio, float colSparsityRatio,
         const std::string &n = "")
      : data(n, {rows, cols}, taco::Format{taco::Dense, taco::Dense}), name(n),
        rows(rows), cols(cols) {

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
        data.insert({i, j}, val);
      }
    }

    data.pack();
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
    return (float) (total - get_nnz())/total;
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
  virtual void compute() = 0;
  virtual void propagate(Direction dir) = 0;
  virtual void print() = 0;
  virtual void print_sparsity() = 0;
  virtual float get_sparsity_ratio() = 0;
  virtual std::string op_type() const = 0;
};

using OpNodePtr = std::shared_ptr<OpNode>;

class MatMul : public OpNode {
public:
  MatMul(TensorPtr A, TensorPtr B, TensorPtr Out) {
    inputs = {A, B};
    output = Out;
  }

  void compute() override {
    taco::IndexVar i, j, k;
    output->data(i, j) = inputs[0]->data(i, k) * inputs[1]->data(k, j);
    output->data.compile();
    output->data.assemble();
    output->data.compute();
  }

  float get_sparsity_ratio() override {
    float a_zeros = inputs[0]->get_sparsity_ratio();
    float b_zeros = inputs[1]->get_sparsity_ratio();
    return (a_zeros + b_zeros) / 2;
  }

  void propagate(Direction dir) override {
    if (dir == FORWARD) {
      ////////BETWEEN OPERANDS
      inputs[1]->rowSparsity &= inputs[0]->colSparsity;
      inputs[0]->colSparsity &= inputs[1]->rowSparsity;
      ////////TO THE OUTPUT
      output->rowSparsity &= inputs[0]->rowSparsity;
      output->colSparsity &= inputs[1]->colSparsity;
    } else {
      inputs[1]->colSparsity &= output->colSparsity;
      inputs[0]->rowSparsity &= output->rowSparsity;
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
  Add(TensorPtr A, TensorPtr B, TensorPtr Out) {
    inputs = {A, B};
    output = Out;
  }

  void compute() override {
    taco::IndexVar i, j;
    output->data(i, j) = inputs[0]->data(i, j) + inputs[1]->data(i, j);
    output->data.compile();
    output->data.assemble();
    output->data.compute();
  }


  float get_sparsity_ratio() override {
    float a_ratio = inputs[0]->get_sparsity_ratio();
    float b_ratio = inputs[1]->get_sparsity_ratio();
    return (a_ratio + b_ratio) / 2;
  }

  void propagate(Direction dir) override {
    if (dir == FORWARD) {
      output->rowSparsity = inputs[0]->rowSparsity | inputs[1]->rowSparsity;
      output->colSparsity = inputs[0]->colSparsity | inputs[1]->colSparsity;
    }
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
};

class Relu : public OpNode {
public:
  Relu(TensorPtr In, TensorPtr Out) {
    inputs = {In};
    output = Out;
  }

  void compute() override {
    for (auto &val : inputs[0]->data)
      output->data.insert(val.first.toVector(), val.second);

    output->data.pack();
  }


  float get_sparsity_ratio() override {
    return inputs[0]->get_sparsity_ratio();
  }

  void propagate(Direction dir) override {
    if (dir == FORWARD) {
      output->rowSparsity = inputs[0]->rowSparsity;
      output->colSparsity = inputs[0]->colSparsity;
    }
  }

  void print() override {
    std::cout << "->ReLU(" << inputs[0]->name << ",out=" << output->name << ")";
  }

  void print_sparsity() override {
    std::cout << "Relu" << std::endl;
    inputs[0]->print_full_sparsity();
    std::cout << " = " << std::endl;
    output->print_full_sparsity();
    std::cout << std::endl;
  }
  std::string op_type() const override { return "Relu"; }
};

class Graph {
  std::vector<OpNodePtr> nodes;
  TensorPtr input, output;

public:
  static Graph build_graph(TensorPtr in, TensorPtr out,
                           const std::vector<OpNodePtr> &ops) {
    Graph g;
    g.input = in;
    g.output = out;
    g.nodes = ops;
    return g;
  }

  float get_sparsity_ratio() {
    float sparsity = 0.0;
    for(auto &op: nodes){
      sparsity += op->get_sparsity_ratio();
    }
    return (float)(sparsity/nodes.size()) + output->get_sparsity_ratio() + 1;

  }

  void propagate(Direction dir = FORWARD) {
    std::cout << ">Propagation pass" << std::endl;
    for (auto &op : nodes) {
      op->propagate(dir);
    }
  }

  void set_formats() {
    std::cout << "Inferring formats (placeholder)...\n";
    // Example: auto-set sparse/dense formats
  }

  TensorPtr compute() {
    for (auto &op : nodes)
      op->compute();
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

int main() {
  auto X = std::make_shared<Tensor>(4, 4, 0.0, 0.0, "X");
  auto W1 = std::make_shared<Tensor>(4, 4, 0.5, 0.5, "W1");
  auto O1 = std::make_shared<Tensor>(4, 4, 0.0, 0.0, "O1");
  auto W2 = std::make_shared<Tensor>(4, 4, 0.5, 0.5, "W2");
  auto O2 = std::make_shared<Tensor>(4, 4, 0.0, 0.0, "O2");
  auto Y = std::make_shared<Tensor>(4, 4, 0.0, 0.0, "Y");

  auto g = Graph::build_graph(X, Y,
                              {std::make_shared<MatMul>(X, W1, O1),
                               std::make_shared<MatMul>(O1, W2, O2),
                               std::make_shared<Relu>(O2, Y)});

  float sparsity = g.get_sparsity_ratio();
  std::cout << "before = " << sparsity << std::endl;
  g.print();
  g.print_sparsity();
  g.propagate(FORWARD);
  g.propagate(BACKWARD);
  g.print_sparsity();
  sparsity = g.get_sparsity_ratio();
  std::cout << "after = " << sparsity << std::endl;
  auto result = g.compute();
}
