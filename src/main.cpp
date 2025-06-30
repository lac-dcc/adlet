#include "graph.hpp"
#include "taco/format.h"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

constexpr int size = 2048;

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

void prop(float sparsity) {
  const auto startAllocate{std::chrono::steady_clock::now()};
  taco::Format dense({taco::Dense, taco::Dense});
  taco::Format dense_sparse({taco::Dense, taco::Sparse});
  auto X = std::make_shared<Tensor>(size, size, 0.9, 0.9, "X", dense_sparse);
  auto W1 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W1", dense);
  auto W2 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W2", dense);
  auto W3 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W3", dense);
  auto W4 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W4", dense);
  auto O1 = std::make_shared<Tensor>(size, size, "O1", dense);
  auto O2 = std::make_shared<Tensor>(size, size, "O2", dense);
  auto O2_T = std::make_shared<Tensor>(size, size, "O2_T", dense);
  auto O3 = std::make_shared<Tensor>(size, size, "O2", dense);
  auto Y = std::make_shared<Tensor>(size, size, "Y", dense);
  auto Y_T = std::make_shared<Tensor>(size, size, "Y_T", dense);

  auto g = Graph::build_graph(X, O1,
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
  const auto startRuntime{std::chrono::steady_clock::now()};
  auto result = g.compute();
  const auto finishRuntime{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> runtimeSecs{finishRuntime - startRuntime};
  const std::chrono::duration<double> compileSecs{startRuntime - startCompile};
  std::cout << "compile = " << compileSecs.count() << std::endl;
  std::cout << "runtime = " << runtimeSecs.count() << std::endl;
}

void noprop() {
  const auto startAllocate{std::chrono::steady_clock::now()};
  taco::Format format({taco::Dense, taco::Dense});
  auto X = std::make_shared<Tensor>(size, size, 0.0, 0.0, "X", format);
  auto W1 = std::make_shared<Tensor>(size, size, "W1", format);
  auto W2 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W2", format);
  auto W3 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W3", format);
  auto W4 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W4", format);
  auto O1 = std::make_shared<Tensor>(size, size, "O1", format);
  auto O2 = std::make_shared<Tensor>(size, size, "O2", format);
  auto O2_T = std::make_shared<Tensor>(size, size, "O2_T", format);
  auto O3 = std::make_shared<Tensor>(size, size, "O2", format);
  auto Y = std::make_shared<Tensor>(size, size, "Y", format);
  auto Y_T = std::make_shared<Tensor>(size, size, "Y_T", format);

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
  const auto startAnalysis{std::chrono::steady_clock::now()};
  g.run_analysis();
  const auto startPropagation{std::chrono::steady_clock::now()};
  g.run_propagation();
  const auto startRuntime{std::chrono::steady_clock::now()};
  auto result = g.compute();
  const auto finishRuntime{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> runtimeSecs{finishRuntime - startRuntime};
  const std::chrono::duration<double> compileSecs{startAnalysis - startCompile};
  const std::chrono::duration<double> propagationSecs{startRuntime -
                                                      startPropagation};
  const std::chrono::duration<double> analysisSecs{startPropagation -
                                                   startAnalysis};
  std::cout << "compile = " << compileSecs.count() << std::endl;
  std::cout << "analysis = " << analysisSecs.count() << std::endl;
  std::cout << "pruning = " << propagationSecs.count() << std::endl;
  std::cout << "runtime = " << runtimeSecs.count() << std::endl;
}

int main(int argc, char **argv) {
  assert(argc == 3);
  double sparsity = std::stod(argv[1]);
  bool useProp = std::stoi(argv[2]);

  if (!useProp) {
    noprop();
    return 0;
  }
  prop(sparsity);
}
