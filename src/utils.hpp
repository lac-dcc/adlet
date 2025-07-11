#pragma once
#include "taco.h"

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
