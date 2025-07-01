#include "graph.hpp"

void test_propagation() {
  int size = 2;

  auto X1 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("01"), bitset("11")}, "X1");
  auto W1 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("10")}, "W1");
  auto O1 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  auto matmul1 = std::make_shared<MatMul>(std::vector<TensorPtr>{X1, W1}, O1);

  auto X2 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto W2 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "W2");
  auto O2 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "O2");
  auto matmul2 = std::make_shared<MatMul>(std::vector<TensorPtr>{X2, W2}, O2);

  auto O3 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "O3");
  auto matmul3 = std::make_shared<MatMul>(std::vector<TensorPtr>{O1, O2}, O3);

  auto g = Graph::build_graph({X1, X2, W1, W2}, O3, {matmul1, matmul2, matmul3});

  g.run_propagation();

  assert(O1->sparsities[0][1] == 0 && "Forward propagation failed!");
  assert(O1->sparsities[1][0] == 0 && "Forward propagation failed!");
  assert(O3->sparsities[0][1] == 0 && "Forward propagation failed!");
  assert(O2->sparsities[0][0] == 0 && "Intra propagation failed!");
  assert(X2->sparsities[0][0] == 0 && "Backward propagation failed!");

  X1->create_data({ taco::Sparse, taco::Dense });
  X2->create_data({ taco::Sparse, taco::Dense });
  O1->create_data({ taco::Sparse, taco::Dense });
  O2->create_data({ taco::Sparse, taco::Dense });
  O3->create_data({ taco::Sparse, taco::Dense });
  W1->create_data({ taco::Sparse, taco::Dense });
  W2->create_data({ taco::Sparse, taco::Dense });

  X1->initialize_data();
  X2->initialize_data();
  W1->initialize_data();
  W2->initialize_data();

  g.compile();
  g.compute();

  assert(O3->data->at({1, 0}) == 0 && O3->data->at({1, 1}) == 0 && "Computation not sparse!"); 
}
//
// void test_compute() {
//   taco::Format format({taco::Dense, taco::Dense});
//   int size = 2;
//   auto X = std::make_shared<Tensor>(size, size, "X", format);
//   float one = 1.0;
//   float zero = 0.0;
//   X->data->insert({0, 0}, zero);
//   X->data->insert({0, 1}, zero);
//   X->data->insert({1, 0}, one);
//   X->data->insert({1, 1}, one);
//   auto W1 = std::make_shared<Tensor>(size, size, "W1", format);
//   W1->data->insert({0, 0}, one);
//   W1->data->insert({0, 1}, one);
//   W1->data->insert({1, 0}, one);
//   W1->data->insert({1, 1}, one);
//   auto W2 = std::make_shared<Tensor>(size, size, "W2", format);
//   W2->data->insert({0, 0}, one);
//   W2->data->insert({0, 1}, one);
//   W2->data->insert({1, 0}, one);
//   W2->data->insert({1, 1}, one);
//   auto O1 = std::make_shared<Tensor>(size, size, "O1", format);
//   auto O2 = std::make_shared<Tensor>(size, size, "O2", format);
//   auto O2_T = std::make_shared<Tensor>(size, size, "O2_T", format);
//
//   auto g = Graph::build_graph({X, W1, W2}, O2_T,
//                               {std::make_shared<MatMul>(X, W1, O1),
//                                std::make_shared<MatMul>(O1, W2, O2),
//                                std::make_shared<Transpose>(O2, O2_T)});
//
//   g.compile();
//   g.compute();
//   assert(0 == g.output->data->at({0, 0}));
//   assert(4 == g.output->data->at({0, 1}));
//   assert(0 == g.output->data->at({1, 0}));
//   assert(4 == g.output->data->at({1, 1}));
// }
//
void test_addition() {
  auto X1 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("01"), bitset("01")}, "X1");
  auto X2 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("01"), bitset("10")}, "X2");
  auto X3 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("01"), bitset("01")}, "X3");

  auto O1 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "O1");

  std::vector<TensorPtr> inputs{ X1, X2, X3 };

  auto add1 = std::make_shared<Add>(inputs, O1);

  auto g = Graph::build_graph({X1, X2, X3}, O1, {add1});
  g.run_propagation();

  assert(O1->sparsities[0][0] == 1 && "Add: Forward propagation failed!");
  assert(O1->sparsities[0][1] == 0 && "Add: Forward propagation failed!");
  assert(O1->sparsities[1][0] == 1 && "Add: Forward propagation failed!");
  assert(O1->sparsities[1][1] == 1 && "Add: Forward propagation failed!");
}

void test_einsum() {
  auto X1 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("01"), bitset("01")}, "X1");
  auto X2 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto O1 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  std::vector<TensorPtr> inputs1{ X1, X2 };
  auto einsum1 = std::make_shared<Einsum>(inputs1, O1, std::string{ "ik,kj->ij" });

  auto X3 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "X2");
  auto O2 = std::make_shared<Tensor>(std::vector<int>{size, size}, std::vector<bitset>{bitset("11"), bitset("11")}, "O1");
  std::vector<TensorPtr> inputs2{ O1, X3 };
  auto einsum2 = std::make_shared<Einsum>(inputs2, O2, std::string{ "ik,kj->ij" });

  assert(einsum1->reductionDims['i'].size() == 0 && "Only 'k' should be a reduction dim!");
  assert(einsum1->reductionDims['j'].size() == 0 && "Only 'k' should be a reduction dim!");
  assert(einsum1->reductionDims['k'].size() == 2 && "'k' should appear twice!");
  assert(einsum1->outputDims['i'].size() == 1 && "'i' should appear once!");
  assert(einsum1->outputDims['j'].size() == 1 && "'j' should appear once!");
  assert(einsum1->outputDims['k'].size() == 0 && "'k' shouldn't appear as an output dim!");

  assert(einsum1->reductionDims['k'][1].first == 1 && einsum1->reductionDims['k'][1].second == 0
          && "Should be the pair (1, 0)!");

  auto g = Graph::build_graph({X1, X2, X3}, O2, {einsum1, einsum2});
  g.run_propagation();

  X1->create_data({ taco::Sparse, taco::Dense });
  X2->create_data({ taco::Sparse, taco::Dense });
  X3->create_data({ taco::Sparse, taco::Dense });
  O1->create_data({ taco::Sparse, taco::Dense });
  O2->create_data({ taco::Sparse, taco::Dense });

  X1->initialize_data();
  X2->initialize_data();
  X3->initialize_data();

  assert(O2->sparsities[0][1] == 0 && "Forward propagation failed!");
  assert(O2->data->at({1, 0}) == 0 && O1->data->at({1, 1}) == 0 && "Computation not sparse!"); 

  g.compile();
  g.compute();
}

int main() {
  // test_compute();
  test_propagation();
  test_addition();
  test_einsum();
}
