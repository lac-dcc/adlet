#include "graph.hpp"

void test_pruning() {
  taco::Format format({taco::Dense, taco::Dense});
  taco::Format sparse({taco::Dense, taco::Sparse});
  int size = 2;
  auto X = std::make_shared<Tensor>(size, size, "X", format);
  float one = 1.0;
  float zero = 0.0;
  X->data->insert({0, 0}, one);
  X->data->insert({0, 1}, one);
  X->data->insert({1, 0}, one);
  X->data->insert({1, 1}, one);
  auto W1 = std::make_shared<Tensor>(size, size, "W1", format);
  W1->data->insert({0, 0}, one);
  W1->data->insert({0, 1}, one);
  W1->data->insert({1, 0}, one);
  W1->data->insert({1, 1}, one);
  auto W2 = std::make_shared<Tensor>(size, size, "W2", format);
  W2->data->insert({0, 0}, zero);
  W2->data->insert({0, 1}, zero);
  W2->data->insert({1, 0}, one);
  W2->data->insert({1, 1}, one);
  W2->rowSparsity = bitset("10");
  W2->colSparsity = bitset("11");

  auto O1 = std::make_shared<Tensor>(size, size, "O1", format);
  auto O2 = std::make_shared<Tensor>(size, size, "O2", format);
  auto O2_T = std::make_shared<Tensor>(size, size, "O2_T", format);
  auto matmul1 = std::make_shared<MatMul>(X, W1, O1);
  auto matmul2 = std::make_shared<MatMul>(O1, W2, O2);
  auto g = Graph::build_graph(X, O2, {matmul1, matmul2});

  assert(format == O1->data->getFormat());
  assert(format == W1->data->getFormat());
  assert(format == W2->data->getFormat());

  assert(1.0f == matmul1->inputs[1]->data->at({0, 0}));
  assert(1.0f == matmul1->inputs[1]->data->at({1, 0}));

  assert(0.0f == O1->data->at({0, 0}));
  assert(0.0f == O1->data->at({1, 0}));

  assert(1.0f == W1->data->at({0, 0}));
  assert(1.0f == W1->data->at({1, 0}));

  g.run_analysis();
  g.run_propagation();

  assert(0.0f == O1->data->at({0, 0}));
  assert(0.0f == O1->data->at({1, 0}));

  assert(0.0f == W1->data->at({0, 0}));
  assert(0.0f == W1->data->at({1, 0}));

  assert(sparse == O1->data->getFormat());
  assert(sparse == W1->data->getFormat());
  assert(format == W2->data->getFormat());
}

void test_propagation() {
  taco::Format format({taco::Dense, taco::Dense});
  int size = 2;
  auto X = std::make_shared<Tensor>(size, size, "X", format);
  float one = 1.0;
  float zero = 0.0;
  X->data->insert({0, 0}, one);
  X->data->insert({0, 1}, one);
  X->data->insert({1, 0}, one);
  X->data->insert({1, 1}, one);
  auto W1 = std::make_shared<Tensor>(size, size, "W1", format);
  W1->data->insert({0, 0}, one);
  W1->data->insert({0, 1}, one);
  W1->data->insert({1, 0}, one);
  W1->data->insert({1, 1}, one);
  auto W2 = std::make_shared<Tensor>(size, size, "W2", format);
  W2->data->insert({0, 0}, zero);
  W2->data->insert({0, 1}, zero);
  W2->data->insert({1, 0}, one);
  W2->data->insert({1, 1}, one);
  W2->rowSparsity = bitset("10");
  W2->colSparsity = bitset("11");
  auto O1 = std::make_shared<Tensor>(size, size, "O1", format);
  auto O2 = std::make_shared<Tensor>(size, size, "O2", format);
  auto O2_T = std::make_shared<Tensor>(size, size, "O2_T", format);
  auto g = Graph::build_graph(X, O2_T,
                              {std::make_shared<MatMul>(X, W1, O1),
                               std::make_shared<MatMul>(O1, W2, O2),
                               std::make_shared<Transpose>(O2, O2_T)});

  assert(size == count_bits(X->rowSparsity, size));
  assert(size == count_bits(X->colSparsity, size));
  assert(size == count_bits(O1->rowSparsity, size));
  assert(size == count_bits(O1->colSparsity, size));
  assert(size == count_bits(W1->rowSparsity, size));
  assert(size == count_bits(W1->colSparsity, size));
  assert(size - 1 == count_bits(W2->rowSparsity, size));
  assert(size == count_bits(W2->colSparsity, size));
  g.run_analysis();
  assert(size == count_bits(X->rowSparsity, size));
  assert(size == count_bits(X->colSparsity, size));
  assert(size == count_bits(O1->rowSparsity, size));
  assert(size - 1 == count_bits(O1->colSparsity, size));
  assert(size == count_bits(W1->rowSparsity, size));
  assert(size - 1 == count_bits(W1->colSparsity, size));
  assert(size - 1 == count_bits(W2->rowSparsity, size));
  assert(size == count_bits(W2->colSparsity, size));
}

void test_compute() {
  taco::Format format({taco::Dense, taco::Dense});
  int size = 2;
  auto X = std::make_shared<Tensor>(size, size, "X", format);
  float one = 1.0;
  float zero = 0.0;
  X->data->insert({0, 0}, zero);
  X->data->insert({0, 1}, zero);
  X->data->insert({1, 0}, one);
  X->data->insert({1, 1}, one);
  auto W1 = std::make_shared<Tensor>(size, size, "W1", format);
  W1->data->insert({0, 0}, one);
  W1->data->insert({0, 1}, one);
  W1->data->insert({1, 0}, one);
  W1->data->insert({1, 1}, one);
  auto W2 = std::make_shared<Tensor>(size, size, "W2", format);
  W2->data->insert({0, 0}, one);
  W2->data->insert({0, 1}, one);
  W2->data->insert({1, 0}, one);
  W2->data->insert({1, 1}, one);
  auto O1 = std::make_shared<Tensor>(size, size, "O1", format);
  auto O2 = std::make_shared<Tensor>(size, size, "O2", format);
  auto O2_T = std::make_shared<Tensor>(size, size, "O2_T", format);

  auto g = Graph::build_graph(X, O2_T,
                              {std::make_shared<MatMul>(X, W1, O1),
                               std::make_shared<MatMul>(O1, W2, O2),
                               std::make_shared<Transpose>(O2, O2_T)});

  g.compile();
  g.compute();
  assert(0 == g.output->data->at({0, 0}));
  assert(4 == g.output->data->at({0, 1}));
  assert(0 == g.output->data->at({1, 0}));
  assert(4 == g.output->data->at({1, 1}));
}

void run_all() {
  test_compute();
  test_propagation();
  test_pruning();
}
