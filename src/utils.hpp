#include "taco.h"
#include "taco/format.h"
#include <stdexcept>
#include <string>
#include <vector>

constexpr int size = 2048;

std::vector<int> get_permutation_from_einsum(const std::string &einsum) {
  auto arrow = einsum.find("->");
  if (arrow == std::string::npos) {
    throw std::invalid_argument("Expected '->' in einsum string.");
  }

  std::string source = einsum.substr(0, arrow);
  std::string target = einsum.substr(arrow + 2);

  if (source.size() != target.size()) {
    throw std::invalid_argument("Source and target must be the same length.");
  }

  std::vector<int> permutation;
  for (char c : source) {
    auto pos = target.find(c);
    if (pos == std::string::npos) {
      throw std::invalid_argument("Character from source not found in target.");
    }
    permutation.push_back(static_cast<int>(pos));
  }

  return permutation;
}

std::vector<taco::ModeFormat>
permute_format(const std::vector<int> permutation,
               const std::vector<taco::ModeFormat> &formats) {
  if (permutation.size() != formats.size()) {
    throw std::invalid_argument(
        "Permutation and format vector must have the same size.");
  }

  std::vector<taco::ModeFormat> reordered(formats.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    reordered[permutation[i]] = formats[i];
  }

  return reordered;
}

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
