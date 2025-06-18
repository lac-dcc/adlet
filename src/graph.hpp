#include "taco.h"
#include <iostream>
#include <memory>
#include <cassert>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#ifndef GRAPH
#define GRAPH

#define MAX_DIM 1024

enum Direction { FORWARD, INTRA, BACKWARD };

bool first_n_equal(std::bitset<MAX_DIM> &a, std::bitset<MAX_DIM> &b, int n) {
  assert(n <= MAX_DIM);
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i])
      return false;

  return true;
}

class Tensor {
public:
  taco::Tensor<float> data {};
  std::bitset<MAX_DIM> rowSparsity;
  std::bitset<MAX_DIM> colSparsity;
  const std::string name;
  const int rows;
  const int cols;
  int numOps = 0; // number of operators this tensor belongs to as an operand
  
  // constructor for empty output tensors
  Tensor(int rows, int cols, const std::string &n = "")
      : name(n), rows(rows), cols(cols) {
    rowSparsity.set(); // all rows initially active
    colSparsity.set(); // all cols initially active
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

  void create_data(taco::Format format) {
    data = taco::Tensor<float>(name, {rows, cols}, format);
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

  // how many bits are set in the first n bits 
  int count_bits(std::bitset<MAX_DIM> &sparsityVec, int n) {
    int numSet = 0;
    for (int i = 0; i < n; ++i) {
      if (sparsityVec[i])
        numSet += 1;
    }
    return numSet;
  }

  float get_row_sparsity_ratio() {
    int total = rows;
    return (float) (total - count_bits(rowSparsity, rows)/total);
  }

  float get_col_sparsity_ratio() {
    int total = cols;
    return (float) (total - count_bits(colSparsity, cols)/total);
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
  virtual bool propagate(Direction dir) = 0;
  virtual void print() = 0;
  virtual void print_sparsity() = 0;
  virtual float get_sparsity_ratio() = 0;
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

  bool propagate(Direction dir) override {
    bool changed = false;
    if (dir == FORWARD) {
      auto initRowSparsity = output->rowSparsity;
      auto initColSparsity = output->colSparsity;
      output->rowSparsity &= inputs[0]->rowSparsity;
      output->colSparsity &= inputs[1]->colSparsity;
      changed |= !first_n_equal(initColSparsity, output->colSparsity, output->cols);
      changed |= !first_n_equal(initRowSparsity, output->rowSparsity, output->rows);
    } else if (dir == INTRA) {
      if (inputs[1]->numOps == 1) {
        auto initRowSparsity = inputs[1]->rowSparsity;
        inputs[1]->rowSparsity &= inputs[0]->colSparsity;
        changed |= !first_n_equal(initRowSparsity, inputs[1]->rowSparsity, inputs[1]->rows);
      }
      if (inputs[0]->numOps == 1) {
        auto initColSparsity = inputs[0]->colSparsity;
        inputs[0]->colSparsity &= inputs[1]->rowSparsity;
        changed |= !first_n_equal(initColSparsity, inputs[0]->colSparsity, inputs[0]->cols);
      }
    } else if (dir == BACKWARD) {
      if (inputs[1]->numOps == 1) {
        auto initColSparsity = inputs[1]->colSparsity;
        inputs[1]->colSparsity &= output->colSparsity;
        changed |= !first_n_equal(initColSparsity, inputs[1]->colSparsity, inputs[1]->cols);
      }
      if (inputs[0]->numOps == 1) {
        auto initRowSparsity = inputs[0]->rowSparsity;
        inputs[0]->rowSparsity &= output->rowSparsity;
        changed |= !first_n_equal(initRowSparsity, inputs[0]->rowSparsity, inputs[0]->rows);
      }
    }
    return changed;
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
    A->numOps++;
    B->numOps++;
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

  bool propagate(Direction dir) override {
    bool changed = false;
    if (dir == FORWARD) {
      auto initRowSparsity = output->rowSparsity;
      auto initColSparsity = output->colSparsity;
      output->rowSparsity = inputs[0]->rowSparsity | inputs[1]->rowSparsity;
      output->colSparsity = inputs[0]->colSparsity | inputs[1]->colSparsity;
      changed |= !first_n_equal(initColSparsity, output->colSparsity, output->cols);
      changed |= !first_n_equal(initRowSparsity, output->rowSparsity, output->rows);
    }
    return changed;
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
    In->numOps++;
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

  bool propagate(Direction dir) override {
    bool changed = false;
    if (dir == FORWARD) {
      auto initRowSparsity = output->rowSparsity;
      auto initColSparsity = output->colSparsity;
      output->rowSparsity &= inputs[0]->rowSparsity;
      output->colSparsity &= inputs[0]->colSparsity;     
      changed |= !first_n_equal(initColSparsity, output->colSparsity, output->cols);
      changed |= !first_n_equal(initRowSparsity, output->rowSparsity, output->rows);
    }
    if (dir == BACKWARD) {
      if (inputs[0]->numOps == 1) {
        auto initRowSparsity = output->rowSparsity;
        auto initColSparsity = output->colSparsity;
        inputs[0]->rowSparsity &= output->rowSparsity;
        inputs[0]->colSparsity &= output->colSparsity;     
        changed |= !first_n_equal(initColSparsity, output->colSparsity, output->cols);
        changed |= !first_n_equal(initRowSparsity, output->rowSparsity, output->rows);
      }
    }
    return changed;
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

class Transpose : public OpNode {
public:
  Transpose(TensorPtr In, TensorPtr Out) {
    In->numOps++;
    inputs = {In};
    output = Out;
  }

  void compute() override {
    output->data = inputs[0]->data.transpose({1, 0});
  }


  float get_sparsity_ratio() override {
    return inputs[0]->get_sparsity_ratio();
  }

  bool propagate(Direction dir) override {
    bool changed = false;
    if (dir == FORWARD) {
      auto initRowSparsity = output->rowSparsity;
      auto initColSparsity = output->colSparsity;
      output->rowSparsity &= inputs[0]->colSparsity;
      output->colSparsity &= inputs[0]->rowSparsity;
      changed |= !first_n_equal(initColSparsity, output->colSparsity, output->cols);
      changed |= !first_n_equal(initRowSparsity, output->rowSparsity, output->rows);
    } else if (dir == BACKWARD) {
      if (inputs[0]->numOps == 1) {
        auto initRowSparsity = output->rowSparsity;
        auto initColSparsity = output->colSparsity;
        inputs[0]->rowSparsity &= output->colSparsity;
        inputs[0]->colSparsity &= output->rowSparsity;
        changed |= !first_n_equal(initColSparsity, output->colSparsity, output->cols);
        changed |= !first_n_equal(initRowSparsity, output->rowSparsity, output->rows);
      }
    }
    return changed;
  }

  void print() override {
    std::cout << "->Transpose(" << inputs[0]->name << ",out=" << output->name << ")";
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
  TensorPtr input, output;
  bool fixed = false;

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
    return (float)(sparsity/nodes.size());

  }

  void propagate(Direction dir = FORWARD) {
    std::cout << ">Propagation pass" << std::endl;
    for (auto &op : nodes) {
      op->propagate(dir);
    }
  }

  void propagate_full() {
    while (!fixed) {
      bool changed = false;
      for (auto &op : nodes) {
        changed |= op->propagate(Direction::FORWARD);
        changed |= op->propagate(Direction::INTRA);
        changed |= op->propagate(Direction::BACKWARD);
      }
      if (!changed)
          fixed = true;
    }
  }

  void set_output_formats(float threshold) {
    for (auto &op : nodes) {
      if (op->output->get_row_sparsity_ratio() > threshold) {
        op->output->create_data(taco::Format({taco::Sparse, taco::Dense}));
      } else if (op->output->get_col_sparsity_ratio() > threshold) {
        op->output->create_data(taco::Format({taco::Sparse, taco::Dense}, {1, 0}));
      }

    }
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

void run_graph_with_logging(Graph& g) {
    float sparsity = g.get_sparsity_ratio();
    g.print();
    g.print_sparsity();
    std::cout << "before = " << sparsity << std::endl;
    
    const auto startProp{std::chrono::steady_clock::now()};
    g.print_sparsity();
    std::cout << "PROPAGATE ALL" << std::endl;
    g.propagate_full();
    g.print_sparsity();
    std::cout << "SETFORMATS" << std::endl;
    g.set_output_formats(0.4);
    const auto finishProp{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> propSecs{finishProp - startProp};
    
    sparsity = g.get_sparsity_ratio();
    std::cout << "after = " << sparsity << std::endl;
    
    const auto startRuntime{std::chrono::steady_clock::now()};
    auto result = g.compute();
    const auto finishRuntime{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> runtimeSecs{finishRuntime - startRuntime};
    
    std::cout << "inference = " << propSecs.count() << std::endl;
    std::cout << "runtime = " << runtimeSecs.count() << std::endl;
}

#endif
