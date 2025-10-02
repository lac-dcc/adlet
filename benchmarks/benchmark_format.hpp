#pragma once

#include "taco.h"
#include "taco/format.h"
#include <string>

double compute(taco::Tensor<float> &A, const taco::Tensor<float> &B,
               const taco::Tensor<float> &C);

taco::Tensor<float> assembleTensor(const int rows, const int cols,
                                   double rowSparsity, double columnSparsity,
                                   taco::Format format);

void run(const int rows, const int cols, const std::string out_format,
         const std::string left_format, const std::string right_format,
         double left_row_sparsity, double left_col_sparsity,
         double right_row_sparsity, double right_col_sparsity);

int parseArguments(int argc, char *argv[]);
int benchmarkFormats(int argc, char *argv[]);
double fused(taco::Tensor<float> A, taco::Tensor<float> B,
             taco::Tensor<float> C);
double gspmm(taco::Tensor<float> A, taco::Tensor<float> B,
             taco::Tensor<float> C);
int poc_matrix(int argc, char *argv[]);
