#include "../include/utils.hpp"
#include "taco.h"
#include "taco/format.h"
#include <vector>

int main(int argc, char **argv) {
  std::vector<int> sizes{1024, 2048, 4096};
  std::vector<double> sparsities{0.1, 0.3, 0.5, 0.7, 0.9};
  taco::Format dense({taco::Dense, taco::Dense});
  taco::Format sparse({taco::Dense, taco::Sparse});
  for (auto size : sizes) {
    for (auto sparsity : sparsities) {
      std::cout << size << "," << sparsity << std::endl;
      taco::Tensor<float> dTensor({size, size}, dense);
      taco::Tensor<float> sTensor({size, size}, sparse);
      fill_tensor(dTensor, sparsity, sparsity, size, size);
      fill_tensor(sTensor, sparsity, sparsity, size, size);
      print_tensor_memory_usage(dTensor, "dense = ");
      print_tensor_memory_usage(sTensor, "sparse = ");
      print_memory_usage();
      std::cout << dTensor.getAllocSize() << std::endl;
    }
  }
  return 0;
}
