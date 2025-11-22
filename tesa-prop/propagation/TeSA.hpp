#pragma once
#include "utils.hpp"
#include <bitset>
#include <cassert>
#include <vector>

class TeSA {
  std::vector<SparsityVector> sparsityMatrix{};
  int rows{};
  int cols{};

public:
  TeSA(int rows, int cols) : rows{rows}, cols{cols} {
    for (unsigned int i = 0; i < rows; ++i) {
      sparsityMatrix.push_back(SparsityVector{});
      for (unsigned int j = 0; j < cols; ++j) {
        sparsityMatrix[i][j] = 1;
      }
    }
  }
  TeSA(int rows, int cols, std::vector<SparsityVector> &sparsityMatrix)
      : rows{rows}, cols{cols}, sparsityMatrix{sparsityMatrix} {}

  SparsityVector getRow(int index) const {
    assert(index < rows && "Index out of bounds.");
    return sparsityMatrix[index];
  };
  SparsityVector getCol(int index) const {
    assert(index < cols && "Index out of bounds.");
    SparsityVector res{};
    for (int i = 0; i < rows; ++i)
      res[i] = sparsityMatrix[i][index];

    return res;
  };
  int getRows() const { return rows; }
  int getCols() const { return cols; }

  TeSA transpose();

  friend TeSA operator*(const TeSA &a, const TeSA &b);
  friend TeSA operator+(const TeSA &a, const TeSA &b);
  friend TeSA operator&(const TeSA &a, const TeSA &b);
  friend TeSA operator|(const TeSA &a, const TeSA &b);
};
