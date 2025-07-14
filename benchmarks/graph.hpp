#include "../src/graph.hpp"
#include "taco.h"
#include "taco/format.h"

void run(taco::Format format, bool propagate, float row_sparsity,
         float col_sparsity) {
  const auto startAllocate1{std::chrono::steady_clock::now()};

  auto rowSparsityVector = generate_sparsity_vector(row_sparsity, size);
  auto colSparsityVector = generate_sparsity_vector(col_sparsity, size);
  auto denseSparsityVector = generate_sparsity_vector(0.0, size);

  auto X = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, size),
                          generate_sparsity_vector(col_sparsity, size)},
      "X");
  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, size),
                          denseSparsityVector},
      "W1");
  auto W2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, size),
                          denseSparsityVector},
      "W2");
  auto W3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, size),
                          denseSparsityVector},
      "W3");
  auto W4 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, size),
                          denseSparsityVector},
      "W4");
  // outputs
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

  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X, W1}, O1, "ik,kj->ij");

  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W2, O1}, O2, "ik,kj->ij");

  auto matmul3 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W3, O2}, O3, "ik,kj->ij");

  auto matmul4 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W4, O3}, O4, "ik,kj->ij");

  auto g =
      Graph::build_graph({X, W1}, O4, {matmul1, matmul2, matmul3, matmul4});
  /*{matmul1, matmul2, matmul3, addition, transpose});*/

  const auto finishAllocate1{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate1Secs{finishAllocate1 -
                                                    startAllocate1};

  /*std::cout << "W1 rows " << count_bits(W1->sparsities[0], size) << " W1 cols
   * "*/
  /*          << count_bits(W1->sparsities[1], size) << std::endl;*/
  /*std::cout << "W2 rows " << count_bits(W2->sparsities[0], size) << " W2 cols
   * "*/
  /*          << count_bits(W2->sparsities[1], size) << std::endl;*/
  /*std::cout << "W3 rows " << count_bits(W3->sparsities[0], size) << " W3 cols
   * "*/
  /*          << count_bits(W3->sparsities[1], size) << std::endl;*/
  std::cout << "ratio before " << g.get_sparsity_ratio() << std::endl;

  if (propagate) {
    const auto startPropagation{std::chrono::steady_clock::now()};
    g.run_propagation();
    const auto endPropagation{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> propagationSecs{endPropagation -
                                                        startPropagation};
    std::cout << "analysis = " << propagationSecs.count() << std::endl;
  } else {
    std::cout << "analysis = " << 0 << std::endl;
  }
  std::cout << "ratio after " << g.get_sparsity_ratio() << std::endl;
  const auto startAllocate2{std::chrono::steady_clock::now()};

  /*std::cout << "W1 rows " << count_bits(W1->sparsities[0], size) << " W1 cols
   * "*/
  /*          << count_bits(W1->sparsities[1], size) << std::endl;*/
  /*std::cout << "W2 rows " << count_bits(W2->sparsities[0], size) << " W2 cols
   * "*/
  /*          << count_bits(W2->sparsities[1], size) << std::endl;*/
  /*std::cout << "W3 rows " << count_bits(W3->sparsities[0], size) << " W3 cols
   * "*/
  /*          << count_bits(W3->sparsities[1], size) << std::endl;*/

  X->create_data(format);
  W1->create_data(format);
  W2->create_data(format);
  W3->create_data(format);
  W4->create_data(format);

  /*O1->create_data({taco::Dense, taco::Sparse});*/
  /*O2->create_data({taco::Dense, taco::Sparse});*/
  /*O3->create_data({taco::Dense, taco::Sparse});*/

  O1->create_data({taco::Dense, taco::Dense});
  O2->create_data({taco::Dense, taco::Dense});
  O3->create_data({taco::Dense, taco::Dense});
  O4->create_data({taco::Dense, taco::Dense});

  X->initialize_data();
  W1->initialize_data();
  W2->initialize_data();
  W3->initialize_data();
  W4->initialize_data();

  const auto finishAllocate2{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate2Secs{finishAllocate2 -
                                                    startAllocate2};

  std::cout << "load graph = " << allocate2Secs.count() << std::endl;
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
  print_memory_usage();
}

int benchmark_graph(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " graph <row sparsity> <col sparsity> <format> <propagate> \n";
    return 1;
  }
  int param = 1;
  double row_sparsity = std::stod(argv[++param]);
  double col_sparsity = std::stod(argv[++param]);
  std::string format = argv[++param];
  bool propagate = std::stoi(argv[++param]);
  run(getFormat(format), propagate, row_sparsity, col_sparsity);

  return 0;
}
