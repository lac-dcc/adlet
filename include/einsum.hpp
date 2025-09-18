#pragma once

#include "../include/graph.hpp"
#include "../include/utils.hpp"
#include "taco/format.h"

struct EinsumBenchmark {
  std::vector<std::pair<int, int>> path;
  std::vector<std::string> strings;
  std::vector<std::vector<int>> sizes;
};

std::vector<std::pair<int, int>> get_contraction_path(const std::string &line);
std::vector<std::string> get_contraction_strings(const std::string &line);
std::vector<std::vector<int>> get_tensor_sizes(const std::string &line);
std::string extract_outputs(std::string const &einsumString);
std::vector<std::string> extract_inputs(std::string const &einsumString);
std::unordered_map<char, int>
construct_size_map(std::vector<std::string> const &inputs,
                   std::vector<std::vector<int>> const &tensorSizes);
std::vector<int> deduceOutputDims(std::string const &einsumString,
                                  std::vector<int> const &sizes1,
                                  std::vector<int> const &sizes2);
std::vector<taco::ModeFormatPack> generate_modes(int order,
                                                 bool sparse = false);
EinsumBenchmark read_einsum_benchmark(const std::string &filename);
Graph build_tree(const std::vector<std::vector<int>> &tensorSizes,
                 const std::vector<std::string> &contractionStrings,
                 const std::vector<std::pair<int, int>> &contractionInds,
                 const double sparsity = 0.5);
