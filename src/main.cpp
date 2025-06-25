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

constexpr int size = 1024;

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
    const auto startAllocate{std::chrono::steady_clock::now()};
    taco::Tensor<double> X { {size, size}, {taco::Sparse, taco::Dense} };
    fill_tensor(X, sparsity, sparsity, size, size);
    taco::Tensor<double> W1 { {size, size}, {taco::Dense, taco::Dense} };
    fill_tensor(W1, 0.0, 0.0, size, size);
    taco::Tensor<double> W2 { {size, size}, {taco::Dense, taco::Dense} };
    fill_tensor(W2, 0.0, 0.0, size, size);
    taco::Tensor<double> W3 { {size, size}, {taco::Dense, taco::Dense} };
    fill_tensor(W3, 0.0, 0.0, size, size);
    taco::Tensor<double> W4 { {size, size}, {taco::Dense, taco::Dense} };
    fill_tensor(W4, 0.0, 0.0, size, size);
    taco::Tensor<double> O1 { {size, size}, {taco::Dense, taco::Dense} };
    taco::Tensor<double> O2 { {size, size}, {taco::Dense, taco::Dense} };
    taco::Tensor<double> O2_T { {size, size}, {taco::Dense, taco::Dense} };
    taco::Tensor<double> O3 { {size, size}, {taco::Dense, taco::Dense} };
    taco::Tensor<double> Y { {size, size}, {taco::Dense, taco::Dense} };
    taco::Tensor<double> Y_T { {size, size}, {taco::Dense, taco::Dense} };

    taco::IndexVar i, j, k;
    const auto startCompile{std::chrono::steady_clock::now()};
    O1(i, j) = X(i, k) * W1(k, j);
    O2(i, j) = O1(i, k) * W2(k, j);
    O2_T(i, j) = O2(j, i);
    O3(i, j) = W3(i, k) * O2_T(k, j);
    Y(i, j) = W4(i, k) * O3(k, j);
    Y_T(i, j) = Y(j, i);

    Y_T.compile();
    Y_T.assemble();
    const auto startRuntime{std::chrono::steady_clock::now()};
    Y_T.evaluate();
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
  auto W2 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W2");
  auto W3 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W3");
  auto W4 = std::make_shared<Tensor>(size, size, 0.0, 0.0, "W4");
  auto O1 = std::make_shared<Tensor>(size, size,"O1");
  auto O2 = std::make_shared<Tensor>(size, size,"O2");
  auto O2_T = std::make_shared<Tensor>(size, size,"O2_T");
  auto O3 = std::make_shared<Tensor>(size, size,"O2");
  auto Y = std::make_shared<Tensor>(size, size,"Y");
  auto Y_T = std::make_shared<Tensor>(size, size,"Y_T");

  auto g = Graph::build_graph(X, Y_T,
                              {std::make_shared<MatMul>(X, W1, O1),
                               std::make_shared<MatMul>(O1, W2, O2),
                               std::make_shared<Transpose>(O2, O2_T),
                               std::make_shared<MatMul>(W3, O2_T, O3),
                               std::make_shared<MatMul>(W4, O3, Y),
                               std::make_shared<Transpose>(Y, Y_T)});

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
