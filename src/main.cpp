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


void benchmark(bool propagate, float sparsity) {
  const auto startAllocate1{std::chrono::steady_clock::now()};
  taco::Format dense({taco::Dense, taco::Dense});

  auto rowSparsityVector = generate_sparsity_vector(sparsity, size);
  auto colSparsityVector = generate_sparsity_vector(sparsity, size);
  auto denseSparsityVector = generate_sparsity_vector(0.0, size);

  std::cout << denseSparsityVector << std::endl;

  auto X = std::make_shared<Tensor>(size, size, rowSparsityVector, colSparsityVector, "X");
  auto W1 = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "W1");
  auto W2 = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "W2");
  auto W3 = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "W3");
  auto W4 = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "W4");
  auto O1 = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "O1");
  auto O2 = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "O2");
  auto O2_T = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "O2_T");
  auto O3 = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "O3");
  auto Y = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "Y");
  auto Y_T = std::make_shared<Tensor>(size, size, denseSparsityVector, denseSparsityVector, "Y_T");

  auto g = Graph::build_graph(X, Y_T,
                              {std::make_shared<MatMul>(X, W1, O1),
                               std::make_shared<MatMul>(O1, W2, O2),
                               std::make_shared<Transpose>(O2, O2_T),
                               std::make_shared<MatMul>(W3, O2_T, O3),
                               std::make_shared<MatMul>(W4, O3, Y),
                               std::make_shared<Transpose>(Y, Y_T)});

  const auto finishAllocate1{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate1Secs{finishAllocate1 -
                                                   startAllocate1};

  std::cout << "allocate1 = " << allocate1Secs.count() << std::endl;
  if (propagate) {
    const auto startPropagation{std::chrono::steady_clock::now()};
    g.run_propagation();
    const auto endPropagation{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> propagationSecs{endPropagation -
                                                        startPropagation};
    std::cout << "propagation = " << propagationSecs.count() << std::endl;
  } else {
    std::cout << "propagation = " << 0 << std::endl;
    std::cout << "pruning = " << 0 << std::endl;
  }
  const auto startAllocate2{std::chrono::steady_clock::now()};

  W1->create_data({ taco::Sparse, taco::Dense });
  W2->create_data({ taco::Sparse, taco::Dense });
  W3->create_data({ taco::Sparse, taco::Dense });
  W4->create_data({ taco::Sparse, taco::Dense });
  X->create_data({ taco::Sparse, taco::Dense });
  O1->create_data({ taco::Sparse, taco::Dense });
  O2->create_data({ taco::Sparse, taco::Dense });
  O2_T->create_data({{ taco::Sparse, taco::Dense }, {1, 0}});
  O3->create_data({ taco::Sparse, taco::Dense });
  Y->create_data({ taco::Sparse, taco::Dense });
  Y_T->create_data({{ taco::Sparse, taco::Dense }, {1, 0}});

  fill_tensor(*W1->data, W1->rowSparsity, W1->colSparsity, W1->rows, W1->cols);
  fill_tensor(*W2->data, W2->rowSparsity, W2->colSparsity, W2->rows, W2->cols);
  fill_tensor(*W3->data, W3->rowSparsity, W3->colSparsity, W3->rows, W3->cols);
  fill_tensor(*W4->data, W4->rowSparsity, W4->colSparsity, W4->rows, W4->cols);
  fill_tensor(*X->data, X->rowSparsity, X->colSparsity, X->rows, X->cols);
  const auto finishAllocate2{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate2Secs{finishAllocate2 -
                                                   startAllocate2};

  std::cout << "allocate2 = " << allocate2Secs.count() << std::endl;
  const auto startRuntime{std::chrono::steady_clock::now()};
  g.compile();
  auto result = g.compute();
  const auto finishRuntime{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> runtimeSecs{finishRuntime - startRuntime};
  std::cout << "runtime+compile = " << runtimeSecs.count() << std::endl;
}

int main(int argc, char **argv) {
  assert(argc == 3);
  double sparsity = std::stod(argv[1]);
  bool useProp = std::stoi(argv[2]);
  benchmark(useProp, sparsity);
}
