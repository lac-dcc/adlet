#include "benchmark_format.hpp"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"

double compute(taco::Tensor<float> &A, const taco::Tensor<float> &B,
               const taco::Tensor<float> &C) {

  taco::IndexVar i("i"), j("j"), k("k");
  A(i, j) = taco::sum(k, B(i, k) * C(k, j));
  A.compile();
  A.assemble();
  const auto startRun{std::chrono::steady_clock::now()};
  A.compute();
  const auto finishRun{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> runtimeSecs{finishRun - startRun};
  const auto time = runtimeSecs.count();
  return time;
}

taco::Tensor<float> assembleTensor(const int rows, const int cols,
                                   double rowSparsity, double columnSparsity,
                                   taco::Format format) {

  taco::Tensor<float> A({rows, cols}, format);
  fill_matrix(A, rowSparsity, columnSparsity, rows, cols);
  return A;
}

void run(const int rows, const int cols, const std::string out_format,
         const std::string left_format, const std::string right_format,
         double left_row_sparsity, double left_col_sparsity,
         double right_row_sparsity, double right_col_sparsity) {
  taco::Format taco_left_format = get_format(left_format);
  taco::Format taco_right_format = get_format(right_format);
  taco::Format taco_out_format = get_format(out_format);
  auto A = taco::Tensor<float>({rows, cols}, taco_out_format);
  auto B = assembleTensor(rows, cols, left_row_sparsity, left_col_sparsity,
                          taco_left_format);
  auto C = assembleTensor(rows, cols, right_row_sparsity, right_col_sparsity,
                          taco_right_format);

  auto time = compute(A, B, C);
  std::cout << "rows, cols, out_format, left_format, right_format,"
               "left_row_sparsity, left_col_sparsity, right_row_sparsity, "
               "right_col_sparsity, exec_time"
            << std::endl;
  std::cout << rows << "," << cols << "," << out_format << "," << left_format
            << "," << right_format << "," << left_row_sparsity << ","
            << left_col_sparsity << "," << right_row_sparsity << ","
            << right_col_sparsity << "," << time << std::endl;
}

void show_sizes(const std::string format, int rank, std::vector<int> sizes,
                std::vector<double> sparsities) {

  assert(sizes.size() == sparsities.size());

  std::vector<bitset> sparsity_vector;
  for (int i = 0; i < rank; i++) {
    sparsity_vector.push_back(
        generate_sparsity_vector(sparsities[i], sizes[i]));
  }
  auto tensor =
      std::make_shared<Tensor>(sizes, sparsity_vector, "tensor", false);
  tensor->create_data(get_format(format));
  tensor->fill_tensor();
  tensor->sparsities = sparsity_vector;

  std::cout << format << "," << rank << ",";
  for (int i = 0; i < rank; i++) {
    std::cout << sizes[i] << ",";
  }
  for (int i = 0; i < rank; i++) {
    std::cout << sparsities[i] << ",";
  }
  std::cout << get_tensor_memory_usage(*(tensor->data)) << ","
            << get_memory_usage_mb() << std::endl;
}

/**
 * fused          = S * S * D
 * SpMM           = S * (S * D)
 * mode ordering  = abc,dbe,adce->ade
 */

double fused(taco::Tensor<float> A, taco::Tensor<float> B,
             taco::Tensor<float> C) {

  taco::Tensor<float> result({A.getDimension(0), C.getDimension(1)},
                             taco::Format({taco::Dense, taco::Dense}));

  taco::IndexVar i("i"), j("j"), k("k"), l("l");
  result(i, l) = A(i, j) * B(j, k) * C(k, l);
  result.compile();
  auto start = begin();
  result.evaluate();
  auto time = end(start);
  return time;
}

double gspmm(taco::Tensor<float> A, taco::Tensor<float> B,
             taco::Tensor<float> C) {

  taco::Tensor<float> result({B.getDimension(0), C.getDimension(1)},
                             taco::Format({taco::Dense, taco::Dense}));
  taco::Tensor<float> t1({A.getDimension(0), B.getDimension(1)},
                         taco::Format({taco::Dense, taco::Dense}));
  taco::IndexVar i("i"), j("j"), k("k"), l("l");
  t1(j, l) = B(j, k) * C(k, l);
  result(i, l) = A(i, j) * t1(j, l);
  t1.compile();
  result.compile();
  auto start = begin();
  t1.evaluate();
  result.evaluate();
  auto time = end(start);
  return time;
}

int poc_matrix(int argc, char *argv[]) {
  int param = 1;
  int N = std::stoi(argv[++param]);
  double sparsity = std::stod(argv[++param]);
  int opt = std::stoi(argv[++param]);
  const int M = N;
  const int K = N;
  taco::Tensor<float> A({M, N}, {taco::Dense, taco::Sparse});
  fill_matrix(A, sparsity, M, N);
  taco::Tensor<float> B({N, K}, {taco::Dense, taco::Sparse});
  fill_matrix(B, sparsity, N, K);

  taco::Tensor<float> C({M, K}, {taco::Dense, taco::Sparse});
  fill_matrix(B, 0.0, N, K);

  double r_time;
  if (opt == 0) {
    std::cout << N << "," << sparsity << "," << fused(A, B, C);
  } else {
    std::cout << N << "," << sparsity << "," << gspmm(A, B, C);
  }

  return 0;
}

int parseArguments(int argc, char *argv[]) {
  // return parseArguments(argc, argv);
  /*return benchmarkKernels(argc, argv);*/
  return poc_matrix(argc, argv);
}
