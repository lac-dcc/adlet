#pragma once
#include "taco/format.h"

void deepFM(taco::Format format, bool propagate, float row_sparsity,
            float col_sparsity);

void bert(taco::Format format, bool propagate, float row_sparsity,
          float col_sparsity);

void run(taco::Format format, bool propagate, float row_sparsity,
         float col_sparsity);

void memtest(taco::Format format, bool propagate, float row_sparsity,
             float col_sparsity);

int benchmark_graph(int argc, char *argv[]);
