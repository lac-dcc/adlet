#include "TeSA.hpp"
#include "utils.hpp"
#include <cassert>

TeSA TeSA::transpose() {
  std::vector<SparsityVector> newSparsityMatrix{};
  for (int i = 0; i < cols; ++i) {
    newSparsityMatrix.push_back(getCol(i));
  }
  return TeSA(cols, rows, newSparsityMatrix);
}

TeSA operator*(const TeSA &a, const TeSA &b) {
  assert(a.getCols() == b.getRows() &&
         "Columns of first TeSA must match rows of second TeSA.");
  std::vector<SparsityVector> sparsityMatrix{};

  for (int m = 0; m < a.getRows(); ++m) {
    sparsityMatrix.push_back(SparsityVector{});
    auto &curr = sparsityMatrix.back();
    for (int n = 0; n < b.getCols(); ++n) {
      curr[n] = curr.any();
    }
  }

  return TeSA(a.getRows(), b.getCols(), sparsityMatrix);
}

TeSA operator+(const TeSA &a, const TeSA &b) { return a | b; }

TeSA operator&(const TeSA &a, const TeSA &b) {
  assert(a.getRows() == b.getRows() && a.getCols() == b.getCols() &&
         "TeSA shapes must match.");
  std::vector<SparsityVector> sparsityMatrix{};
  for (int i = 0; i < a.getRows(); ++i)
    sparsityMatrix.push_back(a.getRow(i) & b.getRow(i));

  return TeSA(a.getRows(), b.getRows(), sparsityMatrix);
}

TeSA operator|(const TeSA &a, const TeSA &b) {
  assert(a.getRows() == b.getRows() && a.getCols() == b.getCols() &&
         "TeSA shapes must match.");
  std::vector<SparsityVector> sparsityMatrix{};
  for (int i = 0; i < a.getRows(); ++i)
    sparsityMatrix.push_back(a.getRow(i) | b.getRow(i));

  return TeSA(a.getRows(), b.getRows(), sparsityMatrix);
}
