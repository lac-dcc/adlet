#pragma once
#include "taco.h"
#include <bitset>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

constexpr int size = 2048;

// should be used for creating non-adlet tensors for comparison
void fill_tensor(taco::Tensor<float> &tensor, double rowSparsityRatio,
                 double colSparsityRatio, int rows, int cols) {
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

      float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      tensor.insert({i, j}, val);
    }
  }

  tensor.pack();
}

taco::Format getFormat(const std::string format) {

  taco::Format outFormat;
  if (format == "CSR")
    outFormat = taco::Format({taco::Dense, taco::Sparse});
  else if (format == "CSC")
    outFormat = taco::Format({taco::Dense, taco::Sparse}, {1, 0});
  else if (format == "DD")
    outFormat = taco::Format({taco::Dense, taco::Dense});
  else if (format == "DCSR") {
    outFormat = taco::Format({taco::Sparse, taco::Sparse}, {0, 1});
  } else if (format == "DCSC") {
    outFormat = taco::Format({taco::Sparse, taco::Sparse}, {1, 0});
  } else if (format == "SparseDense") {
    outFormat = taco::Format({taco::Sparse, taco::Dense});
  } else if (format == "SparseDense10") {
    outFormat = taco::Format({taco::Sparse, taco::Dense}, {1, 0});
  }
  return outFormat;
}

int count_bits(std::bitset<size> A, int pos) {
  if (pos < 0)
    return 0;
  if (pos == size)
    return A.count();

  std::bitset<size> mask((1ULL << pos) - 1);
  return (A & mask).count();
}

std::vector<int> get_indices(std::vector<int> &dimSizes, int numElement) {
  int numDims = dimSizes.size();
  std::vector<int> indices(numDims);
  std::vector<int> cumulativeSize(numDims);
  cumulativeSize[0] = 1;

  for (int i = 1; i < numDims; ++i)
    cumulativeSize[i] = cumulativeSize[i - 1] * dimSizes[i - 1];

  for (int i = 0; i < numDims; ++i) {
    if (numElement < cumulativeSize[numDims - 1 - i])
      continue;
    indices[i] = numElement / cumulativeSize[numDims - 1 - i];
    numElement %= cumulativeSize[numDims - 1 - i];
  }

  return indices;
}

std::bitset<size> generate_sparsity_vector(double sparsity, int length) {
  std::bitset<size> sparsityVector;
  sparsityVector.set();

  int numZeros = static_cast<int>(length * sparsity);

  std::vector<int> indices(length);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(),
               std::mt19937{std::random_device{}()});
  for (int i = 0; i < numZeros; ++i)
    sparsityVector.set(indices[i], 0);

  return sparsityVector;
}

void print_tensor_memory_usage(const taco::Tensor<float> &tensor, const std::string &name){
  taco::TensorStorage s = tensor.getStorage();
  std::cout << name << " memory used = " 
    << std::fixed << s.getSizeInBytes() / (1024.0 * 1024.0) << "MB" << std::endl;
}

void print_memory_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  std::cout << "memory used = " << usage.ru_maxrss / (1024.0 * 1024.0)
            << std::endl;
#else
  std::cout << "memory used = " << usage.ru_maxrss / 1024.0 << std::endl;
#endif
}

void write_kernel(const std::string& filename, const taco::Tensor<float> &compiledOut) {
    std::ofstream file;
    file.open(filename);
    file << compiledOut.getSource();
    file.close();
}
