#include "propagation/TeSA.hpp"
#include "propagation/propagate.hpp"
#include <chrono>
#include <iostream>

int main() {
  int rows = SIZE;
  int cols = SIZE;
  TeSA A(rows, cols);
  TeSA B(rows, cols);
  TeSA C(rows, cols);

  auto start = std::chrono::steady_clock::now();
  propagateMatmul(A, B, C);
  double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                       std::chrono::steady_clock::now() - start)
                       .count();

  std::cout << "proptime = " << elapsed << std::endl;

  return 0;
}
