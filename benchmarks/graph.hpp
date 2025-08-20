#include "../src/dot.hpp"
#include "../src/graph.hpp"
#include "../src/utils.hpp"
#include "taco.h"
#include "taco/format.h"
#include <memory>
#include <string>

void deepFM(taco::Format format, bool propagate, float row_sparsity,
            float col_sparsity) {

  const int size = 2048;
  std::cout << "running deepfm-like benchmark" << std::endl;
  const auto startAllocate1{std::chrono::steady_clock::now()};

  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{32, 100},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 100)},
      "X1");
  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{100, 256},
      std::vector<bitset>{generate_sparsity_vector(0, 100),
                          generate_sparsity_vector(col_sparsity, 256)},
      "W1");

  auto W2 = std::make_shared<Tensor>(
      std::vector<int>{100, 8},
      std::vector<bitset>{generate_sparsity_vector(0, 100),
                          generate_sparsity_vector(col_sparsity, 8)},
      "W2");

  auto W3 = std::make_shared<Tensor>(
      std::vector<int>{100, 8},
      std::vector<bitset>{generate_sparsity_vector(0, 100),
                          generate_sparsity_vector(col_sparsity, 8)},
      "W3");

  auto W4 = std::make_shared<Tensor>(
      std::vector<int>{8, 1},
      std::vector<bitset>{generate_sparsity_vector(0, 8),
                          generate_sparsity_vector(0, 1)},
      "W4");

  auto W5 = std::make_shared<Tensor>(
      std::vector<int>{100, 1},
      std::vector<bitset>{generate_sparsity_vector(0, 100),
                          generate_sparsity_vector(0, 1)},
      "W5");

  auto W6 = std::make_shared<Tensor>(
      std::vector<int>{256, 128},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, 256),
                          generate_sparsity_vector(col_sparsity, 128)},
      "W6");

  auto W7 = std::make_shared<Tensor>(
      std::vector<int>{128, 64},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, 128),
                          generate_sparsity_vector(col_sparsity, 64)},
      "W7");

  auto W8 = std::make_shared<Tensor>(
      std::vector<int>{64, 1},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, 64),
                          generate_sparsity_vector(0, 1)},
      "W8");

  auto outputs = std::vector<TensorPtr>(11);
  outputs[0] = std::make_shared<Tensor>(
      std::vector<int>{32, 256},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 256)},
      "O1");
  outputs[1] = std::make_shared<Tensor>(
      std::vector<int>{32, 8},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 8)},
      "O2");
  outputs[2] = std::make_shared<Tensor>(
      std::vector<int>{32, 8},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 8)},
      "O3");
  outputs[3] = std::make_shared<Tensor>(
      std::vector<int>{32, 8},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 8)},
      "O4");
  outputs[4] = std::make_shared<Tensor>(
      std::vector<int>{32, 1},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 1)},
      "O5");
  outputs[5] = std::make_shared<Tensor>(
      std::vector<int>{32, 1},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 1)},
      "O6");
  outputs[6] = std::make_shared<Tensor>(
      std::vector<int>{32, 1},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 1)},
      "O7");
  outputs[7] = std::make_shared<Tensor>(
      std::vector<int>{32, 128},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 128)},
      "O8");
  outputs[8] = std::make_shared<Tensor>(
      std::vector<int>{32, 64},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 64)},
      "O9");
  outputs[9] = std::make_shared<Tensor>(
      std::vector<int>{32, 1},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 1)},
      "O10");
  outputs[10] = std::make_shared<Tensor>(
      std::vector<int>{32, 1},
      std::vector<bitset>{generate_sparsity_vector(0, 32),
                          generate_sparsity_vector(0, 1)},
      "O11");

  auto linear1 = std::make_shared<Einsum>(std::vector<TensorPtr>{X1, W1},
                                          outputs[0], "ik,kj->ij");
  auto matmul1 = std::make_shared<Einsum>(std::vector<TensorPtr>{X1, W2},
                                          outputs[1], "ik,kj->ij");
  auto matmul2 = std::make_shared<Einsum>(std::vector<TensorPtr>{X1, W3},
                                          outputs[2], "ik,kj->ij");
  auto sub1 = std::make_shared<Add>(
      std::vector<TensorPtr>{outputs[1], outputs[2]}, outputs[3]);
  auto sum1 = std::make_shared<Einsum>(std::vector<TensorPtr>{outputs[3], W4},
                                       outputs[4], "ik,kj->ij");
  auto linear2 = std::make_shared<Einsum>(std::vector<TensorPtr>{X1, W5},
                                          outputs[5], "ik,kj->ij");
  auto add1 = std::make_shared<Add>(
      std::vector<TensorPtr>{outputs[4], outputs[5]}, outputs[6]);
  auto linear3 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[0], W6}, outputs[7], "ik,kj->ij");
  auto linear4 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[7], W7}, outputs[8], "ik,kj->ij");
  auto linear5 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[8], W8}, outputs[9], "ik,kj->ij");
  auto add2 = std::make_shared<Add>(
      std::vector<TensorPtr>{outputs[6], outputs[9]}, outputs[10]);

  /*auto g = Graph::build_graph({X1, W1, W2, W3, W4, W5, W6, W7, W8},
   * outputs[10],*/
  /*                            {linear1, linear2, linear3, linear4, linear5,*/
  /*                             matmul1, matmul2, sum1, sub1, add1, add2});*/
  auto g = Graph::build_graph(
      {X1, W1, W2, W3, W4, W5, W6, W7, W8}, outputs[6],
      {linear1, linear2, matmul1, matmul2, sum1, sub1, add1});

  const auto finishAllocate1{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate1Secs{finishAllocate1 -
                                                    startAllocate1};

  g.run_propagation(FORWARD);
  // print_dot(g, "befora.dot");
  std::cout << "ratio before = " << g.get_sparsity_ratio() << std::endl;

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
  std::cout << "ratio after = " << g.get_sparsity_ratio() << std::endl;
  const auto startAllocate2{std::chrono::steady_clock::now()};

  X1->create_data(format);
  W1->create_data(format);
  W2->create_data(format);
  W3->create_data(format);
  W4->create_data(format);
  W5->create_data(format);
  W6->create_data(format);
  W7->create_data(format);
  W8->create_data(format);

  for (int i = 0; i < 11; i++) {
    outputs[i]->create_data(format);
  }

  X1->initialize_data();
  W1->initialize_data();
  W2->initialize_data();
  W3->initialize_data();
  W4->initialize_data();
  W5->initialize_data();
  W6->initialize_data();
  W7->initialize_data();
  W8->initialize_data();

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
  // print_dot(g);
}

void bert(taco::Format format, bool propagate, float row_sparsity,
          float col_sparsity) {

  const int size = 2048;
  std::cout << "running bert-like benchmark" << std::endl;
  const auto startAllocate1{std::chrono::steady_clock::now()};

  auto rowSparsityVector = generate_sparsity_vector(row_sparsity, size);
  auto colSparsityVector = generate_sparsity_vector(col_sparsity, size);
  auto denseSparsityVector = generate_sparsity_vector(0.0, size);

  auto input = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "input");

  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W1");

  auto W2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W2");

  auto W3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W3");

  auto W4 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W4");

  auto W5 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{generate_sparsity_vector(row_sparsity, size),
                          generate_sparsity_vector(col_sparsity, size)},
      "W5");

  auto W6 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{generate_sparsity_vector(col_sparsity, size),
                          denseSparsityVector},
      "W6");

  auto outputs = std::vector<TensorPtr>(10);
  for (int i = 0; i < 10; i++) {
    outputs[i] = std::make_shared<Tensor>(
        std::vector<int>{size, size},
        std::vector<bitset>{denseSparsityVector, denseSparsityVector},
        "O" + std::to_string(i));
  }

  auto matmul1 = std::make_shared<Einsum>(std::vector<TensorPtr>{input, W1},
                                          outputs[0], "ik,kj->ij");
  auto matmul2 = std::make_shared<Einsum>(std::vector<TensorPtr>{input, W2},
                                          outputs[1], "ik,kj->ij");
  auto matmul3 = std::make_shared<Einsum>(std::vector<TensorPtr>{input, W3},
                                          outputs[2], "ik,kj->ij");

  auto matmul4 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[1], outputs[2]}, outputs[3], "ik,kj->ij");

  auto matmul5 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[3], outputs[0]}, outputs[4], "ik,kj->ij");

  auto matmul6 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[4], W4}, outputs[5], "ik,kj->ij");

  auto add1 = std::make_shared<Add>(std::vector<TensorPtr>{input, outputs[5]},
                                    outputs[6]);

  auto matmul7 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[6], W5}, outputs[7], "ik,kj->ij");

  auto matmul8 = std::make_shared<Einsum>(
      std::vector<TensorPtr>{outputs[7], W6}, outputs[8], "ik,kj->ij");

  auto add2 = std::make_shared<Add>(
      std::vector<TensorPtr>{outputs[8], outputs[6]}, outputs[9]);

  auto g = Graph::build_graph({input, W1, W2, W3, W4, W5, W6}, outputs[9],
                              {matmul1, matmul2, matmul3, matmul4, matmul5,
                               matmul6, add1, matmul7, matmul8, add2});

  const auto finishAllocate1{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate1Secs{finishAllocate1 -
                                                    startAllocate1};

  g.run_propagation(FORWARD);
  // print_dot(g, "befora.dot");
  std::cout << "ratio before = " << g.get_sparsity_ratio() << std::endl;

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
  std::cout << "ratio after = " << g.get_sparsity_ratio() << std::endl;
  const auto startAllocate2{std::chrono::steady_clock::now()};

  input->create_data(format);
  W1->create_data(format);
  W2->create_data(format);
  W3->create_data(format);
  W4->create_data(format);
  W5->create_data(format);
  W6->create_data(format);

  for (int i = 0; i < 10; i++) {
    /*outputs[i]->create_data({taco::Dense, taco::Dense});*/
    outputs[i]->create_data(format);
  }

  input->initialize_data();
  W1->initialize_data();
  W2->initialize_data();
  W3->initialize_data();
  W4->initialize_data();
  W5->initialize_data();
  W6->initialize_data();

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
  g.get_tensor_sizes();
  // print_dot(g);
}

void run(taco::Format format, bool propagate, float row_sparsity,
         float col_sparsity) {
  std::cout << "running small-graph benchmark" << std::endl;
  const int size = 2048;
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
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W1");
  auto W2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W2");
  auto W3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W3");
  auto W4 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
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
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O3");
  auto O4 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O4");

  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W1, X}, O1, "ik,kj->ij");

  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W2, O1}, O2, "ik,kj->ij");

  auto matmul3 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W3, O2}, O3, "ik,kj->ij");

  auto matmul4 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W4, O3}, O4, "ik,kj->ij");

  auto g =
      Graph::build_graph({X, W1}, O4, {matmul1, matmul2, matmul3, matmul4});

  const auto finishAllocate1{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate1Secs{finishAllocate1 -
                                                    startAllocate1};
  g.run_propagation(FORWARD);
  std::cout << "ratio before = " << g.get_sparsity_ratio() << std::endl;
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
  std::cout << "ratio after = " << g.get_sparsity_ratio() << std::endl;
  const auto startAllocate2{std::chrono::steady_clock::now()};

  X->create_data(format);
  W1->create_data(format);
  W2->create_data(format);
  W3->create_data(format);
  W4->create_data(format);

  O1->create_data(format);
  O2->create_data(format);
  O3->create_data(format);
  O4->create_data(format);

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
  print_dot(g);
}

void memtest(taco::Format format, bool propagate, float row_sparsity,
             float col_sparsity) {
  const int size = 2048;
  std::cout << "memory usage on start is ";
  print_memory_usage();
  std::cout << "running memtest-graph benchmark" << std::endl;
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
      std::vector<bitset>{denseSparsityVector,
                          generate_sparsity_vector(col_sparsity, size)},
      "W1");
  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{denseSparsityVector, denseSparsityVector}, "O1");

  std::cout << "memory usage after Tensor load is ";
  print_memory_usage();

  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W1, X}, O1, "ik,kj->ij");

  std::cout << "memory usage after op load is ";
  print_memory_usage();

  auto g = Graph::build_graph({X, W1}, O1, {matmul1});

  std::cout << "memory usage after graph load is ";
  print_memory_usage();

  const auto finishAllocate1{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> allocate1Secs{finishAllocate1 -
                                                    startAllocate1};
  g.run_propagation(FORWARD);
  std::cout << "ratio before = " << g.get_sparsity_ratio() << std::endl;
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
  std::cout << "memory usage after prop is ";
  print_memory_usage();
  std::cout << "ratio after = " << g.get_sparsity_ratio() << std::endl;
  const auto startAllocate2{std::chrono::steady_clock::now()};

  X->create_data(format);
  W1->create_data(format);
  O1->create_data(format);
  X->initialize_data();
  W1->initialize_data();

  std::cout << "memory usage after data init is ";
  print_memory_usage();

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
  print_tensor_memory_usage(*O1->data, O1->name);
  print_tensor_memory_usage(*X->data, X->name);
  print_tensor_memory_usage(*W1->data, W1->name);
  print_dot(g);
  write_kernel("memtest.c", *O1->data);
}

int benchmark_graph(int argc, char *argv[]) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0]
              << " graph <graph_name> <row sparsity> <col sparsity> <format> "
                 "<propagate> <seed> \n ";
    return 1;
  }
  int param = 1;
  std::string graph_name = argv[++param];
  double row_sparsity = std::stod(argv[++param]);
  double col_sparsity = std::stod(argv[++param]);
  std::string format = argv[++param];
  bool propagate = std::stoi(argv[++param]);
  SEED = std::stoi(argv[++param]);

  if (graph_name == "bert") {
    bert(getFormat(format), propagate, row_sparsity, col_sparsity);
  } else if (graph_name == "deepfm") {
    deepFM(getFormat(format), propagate, row_sparsity, col_sparsity);
  } else if (graph_name == "mem_test") {
    memtest(getFormat(format), propagate, row_sparsity, col_sparsity);
  } else {
    run(getFormat(format), propagate, row_sparsity, col_sparsity);
  }
  return 0;
}
