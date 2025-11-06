#pragma once
#include <string>

void run(const std::string &file_path, const double sparsity, bool run_fw,
              bool run_lat, bool run_bw);

int benchmark_einsum_prop(int argc, char *argv[]);
