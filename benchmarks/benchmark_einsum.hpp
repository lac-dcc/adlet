#pragma once
#include <string>

void run(const std::string &file_path, const bool propagate,
         const double sparsity, const bool sparse);

void run_prop(const std::string &file_path, const double sparsity, bool run_fw,
              bool run_lat, bool run_bw);

int benchmark_einsum(int argc, char *argv[]);
