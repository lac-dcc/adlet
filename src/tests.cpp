#include "einsum.hpp"
#include "graph.hpp"
#include "taco.h"
#include "taco/format.h"
#include <cassert>
#include <memory>
#include <vector>

void print_matrix(taco::Tensor<float> &tensor, std::vector<int> sizes) {
  assert(sizes.size() == 2 && "Tensor must be a matrix to call this method");
  std::vector<std::vector<float>> tmp(sizes[0],
                                      std::vector<float>(sizes[1], 0.0));
  for (auto entry : tensor) {
    tmp[entry.first[0]][entry.first[1]] = entry.second;
  }
  for (int i = 0; i < sizes[0]; ++i) {
    for (int j = 0; j < sizes[1]; ++j) {
      std::cout << tmp[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

bool is_same(taco::Tensor<float> &a, taco::Tensor<float> &b,
             std::vector<int> sizes) {
  int numElements = 1;
  for (int size : sizes)
    numElements *= size;

  for (int i = 0; i < numElements; ++i) {
    auto index = get_indices(sizes, i);
    auto diff = a.at(index) - b.at(index);
    diff = diff < 0 ? diff * -1 : diff;
    if (diff > 1e-5) {
      // std::cout << index[0] << ", " << index[1] << ": " << diff << std::endl;
      return false;
    }
  }
  return true;
}

void test_propagation() {
  int size = 2;

  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("11")}, "X1");
  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("10")}, "W1");
  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X1, W1}, O1, "ik,kj->ij");

  auto X2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto W2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "W2");
  auto O2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O2");
  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X2, W2}, O2, "ik,kj->ij");

  auto O3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O3");
  auto matmul3 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{O1, O2}, O3, "ik,kj->ij");

  auto g =
      Graph::build_graph({X1, W1, X2, W2}, O3, {matmul1, matmul2, matmul3});

  g.run_propagation();

  assert(X1->sparsities[0][0] == 1 && "X1 sparsity shouldn't change!");
  assert(X1->sparsities[0][1] == 0 && "X1 sparsity shouldn't change!");
  assert(X1->sparsities[1][0] == 1 && "X1 sparsity shouldn't change!");
  assert(X1->sparsities[1][1] == 1 && "X1 sparsity shouldn't change!");

  assert(W1->sparsities[0][0] == 1 && "W1 sparsity shouldn't change!");
  assert(W1->sparsities[0][1] == 1 && "W1 sparsity shouldn't change!");
  assert(W1->sparsities[1][0] == 0 && "W1 sparsity shouldn't change!");
  assert(W1->sparsities[1][1] == 1 && "W1 sparsity shouldn't change!");

  assert(O1->sparsities[1][0] == 0 && "Forward propagation failed!");
  assert(O1->sparsities[0][1] == 0 && "Forward propagation failed!");
  assert(O1->sparsities[1][1] == 1 && "Forward propagation failed!");
  assert(O1->sparsities[0][0] == 1 && "Forward propagation failed!");

  X1->create_data({taco::Sparse, taco::Dense});
  X2->create_data({taco::Sparse, taco::Dense});
  O1->create_data({taco::Sparse, taco::Dense});
  O2->create_data({taco::Sparse, taco::Dense});
  O3->create_data({taco::Sparse, taco::Dense});
  W1->create_data({taco::Sparse, taco::Dense});
  W2->create_data({taco::Sparse, taco::Dense});

  X1->initialize_data();
  X2->initialize_data();
  W1->initialize_data();
  W2->initialize_data();

  g.compile();
  g.compute();

  assert(O3->data->at({1, 0}) == 0 && O3->data->at({1, 1}) == 0 &&
         "Values expected to be sparse aren't!");
  assert(O3->data->at({0, 0}) != 0 && O3->data->at({0, 1}) != 0 &&
         "Values expected to be dense are sparse!");
  std::cout << "test_propagation() OK " << std::endl;
}

void test_addition() {
  const int size = 2;
  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("01")}, "X1");
  auto X2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("10")}, "X2");
  auto X3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("01")}, "X3");

  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O1");

  std::vector<TensorPtr> inputs{X1, X2, X3};

  auto add1 = std::make_shared<Add>(inputs, O1);

  auto g = Graph::build_graph({X1, X2, X3}, O1, {add1});
  g.run_propagation();

  assert(O1->sparsities[0][0] == 1 && "Add: Forward propagation failed!");
  assert(O1->sparsities[0][1] == 0 && "Add: Forward propagation failed!");
  assert(O1->sparsities[1][0] == 1 && "Add: Forward propagation failed!");
  assert(O1->sparsities[1][1] == 1 && "Add: Forward propagation failed!");
  std::cout << "test_addition() OK " << std::endl;
}

void test_einsum() {
  const int size = 2;
  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("01")}, "X1");
  auto X2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  std::vector<TensorPtr> inputs1{X1, X2};
  auto einsum1 =
      std::make_shared<Einsum>(inputs1, O1, std::string{"ik,kj->ij"});

  auto X3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto O2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O2");
  std::vector<TensorPtr> inputs2{O1, X3};
  auto einsum2 =
      std::make_shared<Einsum>(inputs2, O2, std::string{"ik,kj->ij"});

  assert(einsum1->reductionDims['i'].size() == 0 &&
         "Only 'k' should be a reduction dim!");
  assert(einsum1->reductionDims['j'].size() == 0 &&
         "Only 'k' should be a reduction dim!");
  assert(einsum1->reductionDims['k'].size() == 2 && "'k' should appear twice!");
  assert(einsum1->outputDims['i'].size() == 1 && "'i' should appear once!");
  assert(einsum1->outputDims['j'].size() == 1 && "'j' should appear once!");
  assert(einsum1->outputDims['k'].size() == 0 &&
         "'k' shouldn't appear as an output dim!");

  assert(einsum1->reductionDims['k'][1].first == 1 &&
         einsum1->reductionDims['k'][1].second == 0 &&
         "Should be the pair (1, 0)!");

  auto g = Graph::build_graph({X1, X2, X3}, O2, {einsum1, einsum2});
  g.run_propagation();

  X1->create_data({taco::Sparse, taco::Dense});
  X2->create_data({taco::Sparse, taco::Dense});
  X3->create_data({taco::Sparse, taco::Dense});
  O1->create_data({taco::Sparse, taco::Dense});
  O2->create_data({taco::Sparse, taco::Dense});

  X1->initialize_data();
  X2->initialize_data();
  X3->initialize_data();

  assert(O2->sparsities[0][1] == 0 && "Forward propagation failed!");
  assert(O2->data->at({1, 0}) == 0 && O1->data->at({1, 1}) == 0 &&
         "Computation not sparse!");
  assert(O1->outputTensor == true);
  assert(O2->outputTensor == true);
  assert(X1->outputTensor == false);
  assert(X2->outputTensor == false);
  assert(X3->outputTensor == false);

  g.compile();
  g.compute();
  std::cout << "test_einsum() OK " << std::endl;
}

void test_einsum_transpose() {
  int size = 2;

  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("10")}, "X1");
  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  auto transpose =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X1}, O1, "ij->ji");

  auto g = Graph::build_graph({X1}, O1, {transpose});
  g.run_propagation();

  assert(X1->sparsities[0][0] == 1 && "X1 sparsity shouldn't change!");
  assert(X1->sparsities[0][1] == 0 && "X1 sparsity shouldn't change!");
  assert(X1->sparsities[1][0] == 0 && "X1 sparsity shouldn't change!");
  assert(X1->sparsities[1][1] == 1 && "X1 sparsity shouldn't change!");

  assert(O1->sparsities[0][0] == 0 && "Forward propagation failed!");
  assert(O1->sparsities[0][1] == 1 && "Forward propagation failed!");
  assert(O1->sparsities[1][0] == 1 && "Forward propagation failed!");
  assert(O1->sparsities[1][1] == 0 && "Forward propagation failed!");

  X1->create_data({taco::Sparse, taco::Dense});
  O1->create_data({{taco::Sparse, taco::Dense}, {1, 0}});

  X1->initialize_data();

  g.compile();
  g.compute();

  assert(X1->data->at({0, 1}) == O1->data->at({1, 0}));
  std::cout << "test_einsum_transpose() OK " << std::endl;
}

void test_einsum_multiop_1() {
  int size = 2;

  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("01")}, "X1");
  auto X2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("01")}, "W1");

  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X1, X2}, O1, "ik,kj->ij");
  auto O2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O2");
  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{W1, X2}, O2, "ik,kj->ij");

  auto O3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O3");
  auto matmul3 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{O1, O2}, O3, "ik,kj->ij");

  auto g = Graph::build_graph({X1, X2, W1}, O1, {matmul1, matmul2, matmul3});
  g.run_propagation();

  X1->create_data({taco::Sparse, taco::Dense});
  X2->create_data({taco::Sparse, taco::Dense});
  W1->create_data({taco::Sparse, taco::Dense});
  O1->create_data({taco::Sparse, taco::Dense});
  O2->create_data({taco::Sparse, taco::Dense});
  O3->create_data({taco::Sparse, taco::Dense});

  X1->initialize_data();
  X2->initialize_data();
  W1->initialize_data();

  g.compile();
  g.compute();

  assert(X2->data->at({0, 0}) != 0 && X2->data->at({0, 1}) != 0 &&
         X2->data->at({1, 0}) == 0 && X2->data->at({1, 1}) == 0);
  assert(O3->data->at({0, 0}) != 0 && O3->data->at({0, 1}) != 0 &&
         O3->data->at({1, 0}) == 0 && O3->data->at({1, 1}) == 0);
  std::cout << "test_einsum_multiop_1() OK " << std::endl;
}

void test_einsum_multiop_2() {
  int size = 2;

  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("01"), bitset("01")}, "X1");
  auto X2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("01")}, "W1");

  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X1, X2}, O1, "ik,kj->ij");
  auto O2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O2");
  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X2, W1}, O2, "ik,kj->ij");

  auto O3 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("11"), bitset("11")}, "O3");
  auto matmul3 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{O1, O2}, O3, "ik,kj->ij");

  auto g = Graph::build_graph({X1, X2, W1}, O3, {matmul1, matmul2, matmul3});
  g.run_propagation();

  X1->create_data({taco::Sparse, taco::Dense});
  X2->create_data({taco::Sparse, taco::Dense});
  W1->create_data({taco::Sparse, taco::Dense});
  O1->create_data({taco::Sparse, taco::Dense});
  O2->create_data({taco::Sparse, taco::Dense});
  O3->create_data({taco::Sparse, taco::Dense});

  X1->initialize_data();
  X2->initialize_data();
  W1->initialize_data();

  g.compile();
  g.compute();

  assert(X2->data->at({0, 0}) != 0 && X2->data->at({0, 1}) != 0 &&
         X2->data->at({1, 0}) != 0 && X2->data->at({1, 1}) != 0);
  assert(O3->data->at({0, 0}) != 0 && O3->data->at({0, 1}) == 0 &&
         O3->data->at({1, 0}) == 0 && O3->data->at({1, 1}) == 0);
  std::cout << "test_einsum_multiop_2() OK " << std::endl;
}

void compare_taco_matmul() {
  int size = 3;

  bitset X1Vec1 = generate_sparsity_vector(0.5, size);
  bitset X1Vec2 = generate_sparsity_vector(0.5, size);
  bitset denseVec = generate_sparsity_vector(0, size);
  bitset X2Vec1 = generate_sparsity_vector(0.5, size);
  bitset X2Vec2 = generate_sparsity_vector(0.5, size);
  auto X1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("110"), bitset("111")}, "X1");
  auto W1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("011"), bitset("111")}, "W1");
  auto O1 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("111"), bitset("111")}, "O1");
  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X1, W1}, O1, "ik,kj->ij");

  auto X2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("101"), bitset("111")}, "X2");
  auto O2 = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<bitset>{bitset("111"), bitset("111")}, "O2");
  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X2, O1}, O2, "ik,kj->ij");

  auto g = Graph::build_graph({X1, W1, X2}, O2, {matmul1, matmul2});
  g.run_propagation();

  X1->create_data({taco::Sparse, taco::Dense});
  X2->create_data({taco::Sparse, taco::Dense});
  W1->create_data({taco::Sparse, taco::Dense});
  O1->create_data({taco::Sparse, taco::Dense});
  O2->create_data({taco::Sparse, taco::Dense});

  X1->data->insert({0, 0}, 1.0f);
  X1->data->insert({0, 1}, 1.0f);
  X1->data->insert({0, 2}, 1.0f);

  X1->data->insert({1, 0}, 2.0f);
  X1->data->insert({1, 1}, 2.0f);
  X1->data->insert({1, 2}, 2.0f);

  W1->data->insert({1, 0}, 1.0f);
  W1->data->insert({1, 1}, 1.0f);
  W1->data->insert({1, 2}, 1.0f);

  W1->data->insert({2, 0}, 2.0f);
  W1->data->insert({2, 1}, 2.0f);
  W1->data->insert({2, 2}, 2.0f);

  X2->data->insert({0, 0}, 1.0f);
  X2->data->insert({0, 1}, 1.0f);
  X2->data->insert({0, 2}, 1.0f);

  X2->data->insert({2, 0}, 2.0f);
  X2->data->insert({2, 1}, 2.0f);
  X2->data->insert({2, 2}, 2.0f);

  g.compile();
  g.compute();

  taco::Tensor<float> X1Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> X2Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> W1Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> O1Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> O2Taco({size, size}, {taco::Sparse, taco::Dense});

  X1Taco = *X1->data;
  X2Taco = *X2->data;
  W1Taco = *W1->data;

  taco::IndexVar i, j, k;
  O1Taco(i, j) = X1Taco(i, k) * W1Taco(k, j);
  O2Taco(i, j) = X2Taco(i, k) * O1Taco(k, j);
  O2Taco.evaluate();

  assert(is_same(O2Taco, *O2->data, {size, size}) &&
         "Resulting tensors are different!");

  assert(O2->data->at({0, 0}) == 9);
  assert(O2->data->at({0, 1}) == 9);
  assert(O2->data->at({0, 2}) == 9);

  assert(O2->data->at({1, 0}) == 0);
  assert(O2->data->at({1, 1}) == 0);
  assert(O2->data->at({1, 2}) == 0);

  assert(O2->data->at({2, 0}) == 18);
  assert(O2->data->at({2, 1}) == 18);
  assert(O2->data->at({2, 2}) == 18);

  std::cout << "compare_taco_matmul() OK " << std::endl;
}

void compare_taco_einsum() {
  int size = 10;

  bitset X1Vec1 = generate_sparsity_vector(0.5, size);
  bitset X1Vec2 = generate_sparsity_vector(0.5, size);
  bitset X2Vec1 = generate_sparsity_vector(0.5, size);
  bitset X2Vec2 = generate_sparsity_vector(0.5, size);

  bitset denseVec = generate_sparsity_vector(0, size);

  auto X1 = std::make_shared<Tensor>(std::vector<int>{size, size},
                                     std::vector<bitset>{X1Vec1, X1Vec2}, "X1");
  auto X2 = std::make_shared<Tensor>(std::vector<int>{size, size},
                                     std::vector<bitset>{X2Vec1, X2Vec2}, "X2");
  auto W1 =
      std::make_shared<Tensor>(std::vector<int>{size, size},
                               std::vector<bitset>{denseVec, denseVec}, "W1");

  auto O1 =
      std::make_shared<Tensor>(std::vector<int>{size, size},
                               std::vector<bitset>{denseVec, denseVec}, "O1");
  auto matmul1 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X1, X2}, O1, "ik,kj->ij");
  auto O2 =
      std::make_shared<Tensor>(std::vector<int>{size, size},
                               std::vector<bitset>{denseVec, denseVec}, "O2");
  auto O2_T =
      std::make_shared<Tensor>(std::vector<int>{size, size},
                               std::vector<bitset>{denseVec, denseVec}, "O2_T");
  auto matmul2 =
      std::make_shared<Einsum>(std::vector<TensorPtr>{X2, W1}, O2, "ik,kj->ij");
  auto transpose =
      std::make_shared<Einsum>(std::vector<TensorPtr>{O2}, O2_T, "ij->ji");
  auto O3 =
      std::make_shared<Tensor>(std::vector<int>{size, size},
                               std::vector<bitset>{denseVec, denseVec}, "O3");
  auto matmul3 = std::make_shared<Einsum>(std::vector<TensorPtr>{O1, O2_T}, O3,
                                          "ik,kj->ij");

  auto g = Graph::build_graph({X1, X2, W1}, O1,
                              {matmul1, matmul2, transpose, matmul3});
  g.run_propagation();

  X1->create_data({taco::Sparse, taco::Dense});
  X2->create_data({taco::Sparse, taco::Dense});
  W1->create_data({taco::Sparse, taco::Dense});
  O1->create_data({taco::Sparse, taco::Dense});
  O2->create_data({taco::Sparse, taco::Dense});
  O2_T->create_data({{taco::Sparse, taco::Dense}, {1, 0}});
  O3->create_data({taco::Sparse, taco::Dense});

  X1->initialize_data();
  X2->initialize_data();
  W1->initialize_data();

  g.compile();
  g.compute();

  taco::Tensor<float> X1Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> X2Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> W1Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> O1Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> O2Taco({size, size}, {taco::Sparse, taco::Dense});
  taco::Tensor<float> O2_TTaco({size, size},
                               {{taco::Sparse, taco::Dense}, {1, 0}});
  taco::Tensor<float> O3Taco({size, size}, {taco::Sparse, taco::Dense});
  // fill_tensor(X1Taco, {X1Vec1, X1Vec2}, {size, size});
  // fill_tensor(X2Taco, {X2Vec1, X2Vec2}, {size, size});
  // fill_tensor(W1Taco, {denseVec, denseVec}, {size, size});
  X1Taco = *X1->data;
  X2Taco = *X2->data;
  W1Taco = *W1->data;
  taco::IndexVar i, j, k;
  O1Taco(i, j) = X1Taco(i, k) * X2Taco(k, j);
  O2Taco(i, j) = X2Taco(i, k) * W1Taco(k, j);
  O2_TTaco(i, j) = O2Taco(j, i);
  O3Taco(i, j) = O1Taco(i, k) * O2_TTaco(k, j);
  O3Taco.evaluate();

  assert(is_same(O3Taco, *O3->data, {size, size}) &&
         "Resuling tensors are different!");
  std::cout << "compare_taco_einsum() OK " << std::endl;
}

void test_get_sparsity_ratio() {
  // these tests may fail if the global size is less than 10
  float tolerance = 1e-5;
  auto rowSparsityVector = bitset("101");
  auto colSparsityVector = bitset("111");
  auto tensor = std::make_shared<Tensor>(
      std::vector<int>{3, 3},
      std::vector<bitset>{rowSparsityVector, colSparsityVector}, "X");
  assert(std::fabs(tensor->get_sparsity_ratio() - 0.333333f) < tolerance);

  rowSparsityVector = bitset("0010101011");
  colSparsityVector = bitset("1110100100");
  tensor = std::make_shared<Tensor>(
      std::vector<int>{10, 10},
      std::vector<bitset>{rowSparsityVector, colSparsityVector}, "X");
  assert(std::fabs(tensor->get_sparsity_ratio() - 0.75f) < tolerance);

  std::cout << "test_get_sparsity_ratio() OK " << std::endl;
}

void test_einsum_utils() {
  // our code doesn't support scalar outputs so this is a modified version of
  // https://optimized-einsum.readthedocs.io/en/stable/path_finding.html#format-of-the-path
  // example
  std::vector<std::string> contractionStrings{"ajac,acaj->a", "ikbd,bdik->bik",
                                              "bik,ikab->a", "a,a->a"};
  std::vector<std::pair<int, int>> contractionInds{
      {1, 3}, {0, 2}, {0, 2}, {0, 1}};
  std::vector<std::vector<int>> tensorSizes;
  tensorSizes.push_back({10, 17, 10, 9});
  tensorSizes.push_back({16, 13, 16, 15});
  tensorSizes.push_back({10, 9, 16, 10});
  tensorSizes.push_back({16, 15, 16, 13});
  tensorSizes.push_back({10, 9, 10, 17});

  auto graph = buildTree(tensorSizes, contractionStrings, contractionInds);
  assert(graph.inputs.size() == 9);
  assert(graph.nodes.size() == 4);
  std::cout << "test_einsum_utils() OK " << std::endl;
}

void test_init_data() {
  std::vector<std::vector<int>> sizes = {
      {13, 9},          {13, 5, 46},     {9, 27, 7},      {5, 17, 19},
      {27, 10, 68},     {17, 17, 3},     {10, 79, 3},     {46, 7, 15, 25},
      {15, 6, 26},      {25, 24, 9},     {19, 6, 68, 22}, {68, 5, 7},
      {22, 22, 11, 56}, {26, 24, 22, 7}, {7, 8, 7, 48},   {9, 68, 8, 6},
      {6, 4, 11},       {17, 5, 9},      {7, 11, 9},      {56, 7, 9},
      {48, 4, 9},       {11, 20, 9},     {20, 5},         {5, 6, 9},
      {6, 25},          {25, 79, 9}};

  for (auto s : sizes) {
    std::vector<bitset> bitVectors;
    int totalElements = 1;
    for (auto dim : s) {
      totalElements *= dim;
      bitVectors.push_back(generate_sparsity_vector(0.0, dim));
    }

    auto tmpVec = std::make_shared<Tensor>(s, bitVectors);
    tmpVec->create_data(generateModes(s.size()));
    tmpVec->initialize_data();
  }
  std::cout << "test_init_data() OK " << std::endl;
}

int main(int argc, char **argv) {
  test_propagation();
  test_addition();
  test_einsum();
  test_einsum_transpose();
  test_einsum_multiop_1();
  test_einsum_multiop_2();
  compare_taco_matmul();
  compare_taco_einsum();
  test_get_sparsity_ratio();
  test_init_data();
  test_einsum_utils();
}
