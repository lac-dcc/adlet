#include "benchmark_format.hpp"
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
  fill_tensor(A, rowSparsity, columnSparsity, rows, cols);
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

int parseArguments(int argc, char *argv[]) {
  if (argc != 11) {
    std::cerr
        << "Usage: " << argv[0]
        << " format <rows> <cols> <out_format> <left_format> <right_format> "
           "<left_row_sparsity> <left_col_sparsity> <right_row_sparsity> "
           "<right_col_sparsity> \n";
    return 1;
  }
  int param = 1;
  int rows = std::stoi(argv[++param]);
  int cols = std::stoi(argv[++param]);
  std::string out_format = argv[++param];
  std::string left_format = argv[++param];
  std::string right_format = argv[++param];
  double left_row_sparsity = std::stod(argv[++param]);
  double left_col_sparsity = std::stod(argv[++param]);
  double right_row_sparsity = std::stod(argv[++param]);
  double right_col_sparsity = std::stod(argv[++param]);
  run(rows, cols, out_format, left_format, right_format, left_row_sparsity,
      left_col_sparsity, right_row_sparsity, right_col_sparsity);
  return 0;
}

int benchmarkFormats(int argc, char *argv[]) {
  return parseArguments(argc, argv);
}
