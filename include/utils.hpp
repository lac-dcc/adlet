#pragma once
#include "taco.h"
#include <bitset>
#include <cstddef>

#ifndef SIZE_MACRO
  #define SIZE_MACRO 2048
#endif
constexpr int MAX_SIZE = SIZE_MACRO;

using SparsityVector = std::bitset<MAX_SIZE>;

enum Direction { FORWARD, INTRA, BACKWARD };

size_t count_bits(SparsityVector A, int pos);

extern unsigned int SEED;

// should be used for creating non-adlet tensors for comparison
void fill_tensor(taco::Tensor<float> &tensor, double rowSparsityRatio,
                 double colSparsityRatio, int rows, int cols);

taco::Format get_format(const std::string format);

std::vector<int> get_indices(std::vector<int> dimSizes, int numElement);

SparsityVector generate_sparsity_vector(double sparsity, int length);

void print_tensor_memory_usage(const taco::Tensor<float> &tensor,
                               const std::string &name);
void print_memory_usage();

void write_kernel(const std::string &filename,
                  const taco::Tensor<float> &compiledOut);

std::chrono::time_point<std::chrono::high_resolution_clock> begin();

void end(
    const std::chrono::time_point<std::chrono::high_resolution_clock> &start,
    const std::string &message);

bool randomBool(double probability = 0.5);
std::vector<taco::ModeFormatPack> generate_modes(int order, bool sparse);
