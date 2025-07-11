#include "graph.hpp"
#include "taco/format.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

void benchmark(bool propagate, float sparsity) {
  const auto startAllocate1{std::chrono::steady_clock::now()};
  taco::Format dense({taco::Dense, taco::Dense});

  auto rowSparsityVector = generate_sparsity_vector(sparsity, size);
  auto colSparsityVector = generate_sparsity_vector(sparsity, size);
  auto denseSparsityVector = generate_sparsity_vector(0.0, size);

  auto X = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "X");
  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{rowSparsityVector, denseSparsityVector}, "W1");
  auto W2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{rowSparsityVector, denseSparsityVector}, "W2");
  auto W3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{rowSparsityVector, denseSparsityVector}, "W3");
  auto W4 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{rowSparsityVector, denseSparsityVector}, "W4");
  auto W5 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{rowSparsityVector, denseSparsityVector}, "W5");
  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O1");
  auto O2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O2");
  auto O3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O4");
  auto O4 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O4");
  auto O5 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O5");

  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W1, X}, O1, "ik,kj->ij");
  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W2, O1}, O2, "ik,kj->ij");

  auto addition = std::make_shared<Add>(std::vector<TensorPtr>{O2, W3}, O3);
  auto matmul3 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W4, O3}, O4, "ik,kj->ij");
  auto transpose =
      std::make_shared<Einsum>(std::vector<TensorPtr>{O4}, O5, "ij->ji");

  auto g = Graph::build_graph({X, W1}, O5,
                              {matmul1, matmul2, addition, matmul3, transpose});

  const auto finishAllocate1{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate1Secs{finishAllocate1 -
                                                    startAllocate1};

  std::cout << "graph definition = " << allocate1Secs.count() << std::endl;
  if (propagate) {
    const auto startPropagation{std::chrono::steady_clock::now()};
    // g.run_propagation();
    const auto endPropagation{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> propagationSecs{endPropagation -
                                                        startPropagation};
    std::cout << "propagation = " << propagationSecs.count() << std::endl;
  } else {
    std::cout << "propagation = " << 0 << std::endl;
  }
  const auto startAllocate2{std::chrono::steady_clock::now()};

  taco::Format sparseDense({taco::Sparse, taco::Dense});
  taco::Format CSR({taco::Dense, taco::Sparse});
  taco::Format CSC({taco::Dense, taco::Sparse}, {1, 0});

  W1->create_data({sparseDense});
  W2->create_data({sparseDense});
  W3->create_data({sparseDense});
  W4->create_data({sparseDense});
  W5->create_data({sparseDense});
  X->create_data({sparseDense});

  O1->create_data({dense});
  O2->create_data({dense});
  O3->create_data({dense});
  O4->create_data({dense});
  O5->create_data({{taco::Sparse, taco::Dense}, {1, 0}});

  W1->initialize_data();
  W2->initialize_data();
  W3->initialize_data();
  W4->initialize_data();
  W5->initialize_data();

  X->initialize_data();

  const auto finishAllocate2{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate2Secs{finishAllocate2 -
                                                    startAllocate2};

  std::cout << "allocate = " << allocate2Secs.count() << std::endl;
  const auto startCompilation{std::chrono::steady_clock::now()};
  g.compile();
  const auto startRuntime{std::chrono::steady_clock::now()};
  auto result = g.compute();
  const auto finishRuntime{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> compilationSecs{startRuntime -
                                                      startCompilation};
  const std::chrono::duration<double> runtimeSecs{finishRuntime - startRuntime};
  std::cout << "compilation = " << compilationSecs.count() << std::endl;
  std::cout << "runtime = " << runtimeSecs.count() << std::endl;
}

int main(int argc, char **argv) {
  assert(argc == 3);
  double sparsity = std::stod(argv[1]);
  bool useProp = std::stoi(argv[2]);
  benchmark(useProp, sparsity);
}
