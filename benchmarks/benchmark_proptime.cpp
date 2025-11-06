#include "../include/graph.hpp"
#include "../include/node.hpp"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include <cassert>

void benchmark_proptime() {
  int size = MAX_SIZE;
  double sparsity{ 0.5 }; // arbitrary: static analysis runtime doesn't change

  auto A = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<SparsityVector>{
        generate_sparsity_vector(0.5, size),
        generate_sparsity_vector(0.5, size)},
      "A");
  auto B = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<SparsityVector>{
        generate_sparsity_vector(0.5, size),
        generate_sparsity_vector(0.5, size)},
      "B");
  auto C = std::make_shared<Tensor>(
      std::vector<int>{size, size},
      std::vector<SparsityVector>{
        generate_sparsity_vector(0.5, size),
        generate_sparsity_vector(0.5, size)},
      "C");

  auto matmul =
      std::make_shared<Einsum>(std::vector<TensorPtr>{A, B}, C, "ik,kj->ij");

  A->create_data();
  B->create_data();
  C->create_data();

  auto g = Graph::build_graph({A, B}, C, {matmul});
  auto startLoad = begin();
  g.run_propagation();
  std::cout << MAX_SIZE << std::endl;
  end(startLoad, "proptime = ");
}
