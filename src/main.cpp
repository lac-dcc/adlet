#include "graph.hpp"
#include "taco/format.h"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

constexpr int size = 200;

void fill_tensor(taco::Tensor<double> &tensor, double rowSparsityRatio,
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

      double val = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      tensor.insert({i, j}, val);
    }
  }

  tensor.pack();
}

void benchmark(bool propagate, float sparsity) {
  const auto startAllocate{std::chrono::steady_clock::now()};
  taco::Format dense({taco::Dense, taco::Dense});
  auto X = std::make_shared<Tensor>(size, size, sparsity, sparsity, "X", dense);
  auto W1 = std::make_shared<Tensor>(size, size, "W1", dense);
  auto W2 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W2", dense);
  auto W3 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W3", dense);
  auto W4 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W4", dense);
  auto O1 = std::make_shared<Tensor>(size, size, "O1", dense);
  auto O2 = std::make_shared<Tensor>(size, size, "O2", dense);
  auto O2_T = std::make_shared<Tensor>(size, size, "O2_T", dense);
  auto O3 = std::make_shared<Tensor>(size, size, "O2", dense);
  auto Y = std::make_shared<Tensor>(size, size, "Y", dense);
  auto Y_T = std::make_shared<Tensor>(size, size, "Y_T", dense);

  auto g = Graph::build_graph(X, Y_T,
                              {std::make_shared<MatMul>(X, W1, O1),
                               std::make_shared<MatMul>(O1, W2, O2),
                               std::make_shared<Transpose>(O2, O2_T),
                               std::make_shared<MatMul>(W3, O2_T, O3),
                               std::make_shared<MatMul>(W4, O3, Y),
                               std::make_shared<Transpose>(Y, Y_T)});

  const auto finishAllocate{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocateSecs{finishAllocate -
                                                   startAllocate};

  std::cout << "allocate = " << allocateSecs.count() << std::endl;
  const auto startCompile{std::chrono::steady_clock::now()};
  g.compile();
  const auto endCompile{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> compileSecs{endCompile - startCompile};
  std::cout << "compile = " << compileSecs.count() << std::endl;

  if (propagate) {
    const auto startAnalysis{std::chrono::steady_clock::now()};
    g.run_analysis();
    const auto startPropagation{std::chrono::steady_clock::now()};
    g.run_propagation();
    const auto endPropagation{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> analysisSecs{startPropagation -
                                                     startAnalysis};
    const std::chrono::duration<double> propagationSecs{endPropagation -
                                                        startPropagation};
    std::cout << "analysis = " << analysisSecs.count() << std::endl;
    std::cout << "pruning = " << propagationSecs.count() << std::endl;
  } else {
    std::cout << "analysis = " << 0 << std::endl;
    std::cout << "pruning = " << 0 << std::endl;
  }

  const auto startRuntime{std::chrono::steady_clock::now()};
  auto result = g.compute();
  const auto finishRuntime{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> runtimeSecs{finishRuntime - startRuntime};

  std::cout << "runtime = " << runtimeSecs.count() << std::endl;
}

int main(int argc, char **argv) {
  assert(argc == 3);
  double sparsity = std::stod(argv[1]);
  bool useProp = std::stoi(argv[2]);
  benchmark(useProp, sparsity);
}
