#include "taco.h"
#include "graph.hpp"
#include <iostream>
#include <memory>
#include <cassert>
#include <algorithm>
#include <ostream>
#include <random>
#include <bitset>
#include <string>
#include <vector>

constexpr int size = 4096;

void fill_tensor(taco::Tensor<double> &tensor, double rowSparsityRatio, double colSparsityRatio, int rows, int cols) {
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

void noprop(double sparsity) {
    std::cout << sparsity << std::endl;
    const auto startAllocate{std::chrono::steady_clock::now()};
    taco::Tensor<double> X { {size, size}, {taco::Sparse, taco::Dense} };
    fill_tensor(X, sparsity, sparsity, size, size);
    taco::Tensor<double> W1 { {size, size}, {taco::Dense, taco::Dense} };
    fill_tensor(W1, 0.0, 0.0, size, size);
    taco::Tensor<double> O1 { {size, size}, {taco::Dense, taco::Dense} };
    taco::IndexVar i, j, k;
    const auto startCompile{std::chrono::steady_clock::now()};
    O1(i, j) = X(i, k) * W1(k, j);
    O1.compile();
    O1.assemble();
    const auto startRuntime{std::chrono::steady_clock::now()};
    O1.evaluate();
    const auto finishRuntime{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> runtimeSecs{finishRuntime - startRuntime};
    const std::chrono::duration<double> allocateSecs{startCompile - startAllocate};
    const std::chrono::duration<double> compileSecs{startRuntime - startCompile};
    std::cout << "allocate = " << allocateSecs.count() << std::endl;
    std::cout << "compile = " << compileSecs.count() << std::endl;
    std::cout << "runtime = " << runtimeSecs.count() << std::endl;
}

int main(int argc, char** argv) {
  assert(argc == 3);
  double sparsity = std::stod(argv[1]);
  bool useProp = std::stoi(argv[2]);

  if (!useProp) {
    noprop(sparsity);
    return 0;
  }

  const auto startAllocate{std::chrono::steady_clock::now()};
  auto X = std::make_shared<Tensor>(size, size, sparsity, sparsity, "X");
  auto W1 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W1");
  auto O1 = std::make_shared<Tensor>(size, size,"O1");

  auto g = Graph::build_graph(X, O1,
                              {std::make_shared<MatMul>(X, W1, O1)});

  const auto finishAllocate{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocateSecs{finishAllocate - startAllocate};
  std::cout << "allocate = " << allocateSecs.count() << std::endl;
  run_graph_with_logging(g);
  /*std::cout << X->name << std::endl;*/
  /*X->print_tensor();*/
  /*std::cout << W1->name << std::endl;*/
  /*W1->print_tensor();*/
  /*std::cout << O1->name << std::endl;*/
  /*O1->print_tensor();*/
}
