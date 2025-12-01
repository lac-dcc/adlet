/**
 * @file utils.hpp
 * @brief Utility definitions and helper functions for the Sparsity Propagation
 * Analysis (SPA) framework.
 *
 * This file defines the core data structures for the abstract domain,
 * utility functions for bit manipulation, tensor initialization, and
 * performance monitoring.
 */

#pragma once
#include "taco.h"
#include <bitset>
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

// Maximum size for any tensor dimension, used to define the bitset size.
// Should be a large enough constant to cover all dimensions in the target
// application.
#ifndef SIZE_MACRO
#define SIZE_MACRO 4096
#endif
/// @brief The maximum size (extent) for any dimension, setting the length of a
/// SparsityVector.
constexpr int MAX_SIZE = SIZE_MACRO;

/**
 * @brief The core abstract domain element: a fixed-size bitset representing the
 * sparsity of a tensor slice.
 *
 * Each bit corresponds to a coordinate along one dimension. A set bit ('1')
 * means the corresponding slice *may* contain non-zero elements. A clear bit
 * ('0') means the corresponding slice is *structurally zero*.
 */
using SparsityVector = std::bitset<MAX_SIZE>;

/// @brief Defines the direction of sparsity propagation through the
/// computational graph.
enum Direction {
  /// @brief Propagation from inputs to outputs.
  FORWARD,
  /// @brief Propagation within a single operation node (e.g., intermediate
  /// results).
  INTRA,
  /// @brief Propagation from output to inputs (e.g., for pruning or identifying
  /// required input slices).
  BACKWARD
};

/**
 * @brief Counts the number of set bits (non-zero slices) up to a specific
 * position.
 * @param A The SparsityVector (bitset) to check.
 * @param pos The number of elements (up to MAX_SIZE) to check.
 * @return The count of set bits (non-zero slices).
 */
size_t count_bits(SparsityVector A, int pos);

/// @brief Global random seed used for all randomization/shuffling operations
/// (e.g., for data initialization).
extern unsigned int SEED;

// --- Tensor Initialization and Filling Functions ---

/**
 * @brief Fills a concrete TACO tensor with random non-zero values based on row
 * and column sparsity ratios.
 *
 * @note This is typically used for creating ground-truth tensors for
 * comparison, not for SPA-managed Tensors.
 * @param tensor The TACO tensor to fill.
 * @param rowSparsityRatio The ratio of rows to be zeroed (0.0=dense, 1.0=fully
 * sparse).
 * @param colSparsityRatio The ratio of columns to be zeroed.
 * @param rows The row dimension size.
 * @param cols The column dimension size.
 */
void fill_tensor(taco::Tensor<float> &tensor, double rowSparsityRatio,
                 double colSparsityRatio, int rows, int cols);

/**
 * @brief Fills a concrete TACO tensor with random non-zero values based on a
 * single overall sparsity ratio.
 *
 * @param tensor The TACO tensor to fill.
 * @param sparsityRatio The overall element-wise sparsity ratio.
 * @param rows The row dimension size.
 * @param cols The column dimension size.
 */
void fill_tensor(taco::Tensor<float> &tensor, double sparsityRatio, int rows,
                 int cols);

/**
 * @brief Converts a string representation of a format (e.g., "SS" for
 * Sparse-Sparse) into a TACO Format object.
 * @param format A string where 'S' denotes Sparse and any other character
 * denotes Dense.
 * @return The corresponding TACO Format.
 */
taco::Format get_format(const std::string format);

/**
 * @brief Converts a 1D linear index (numElement) into a multidimensional
 * coordinate vector based on dimension sizes.
 * @param dimSizes The size of each dimension.
 * @param numElement The linear index (0 to product(dimSizes) - 1).
 * @return A vector of integer indices representing the coordinate.
 */
std::vector<int> get_indices(std::vector<int> dimSizes, int numElement);

/**
 * @brief Generates a SparsityVector where a given percentage of bits are
 * randomly set to '0' (structurally zero).
 * @param sparsity The ratio of zeroed slices (0.0 to 1.0).
 * @param length The number of elements in the vector (must be <= MAX_SIZE).
 * @return The newly created SparsityVector.
 */
SparsityVector generate_sparsity_vector(double sparsity, int length);

// --- Memory and Timing Functions ---

/**
 * @brief Prints the memory usage of a specific TACO tensor in megabytes.
 * @param tensor The TACO tensor to inspect.
 * @param name The name of the tensor to print in the output.
 */
void print_tensor_memory_usage(const taco::Tensor<float> &tensor,
                               const std::string &name);

/**
 * @brief Calculates the memory usage of a specific TACO tensor in megabytes
 * (MB).
 * @param tensor The TACO tensor to inspect.
 * @return The memory usage in MB.
 */
double get_tensor_memory_usage(const taco::Tensor<float> &tensor);

/**
 * @brief Retrieves the current maximum resident set size (RSS) memory usage of
 * the process in megabytes.
 * @return The memory usage in MB.
 */
double get_memory_usage_mb();

/// @brief Prints the current memory usage of the process.
void print_memory_usage();

/**
 * @brief Writes the generated kernel source code from a TACO tensor compilation
 * to a file.
 * @param filename The path to the output file.
 * @param compiledOut The TACO tensor containing the compiled kernel source.
 */
void write_kernel(const std::string &filename,
                  const taco::Tensor<float> &compiledOut);

/**
 * @brief Records the current high-resolution time point for timing purposes.
 * @return The starting time point.
 */
std::chrono::time_point<std::chrono::high_resolution_clock> begin();

/**
 * @brief Stops timing, prints the elapsed time with a message, and returns
 * void.
 * @param start The starting time point.
 * @param message The message to precede the duration output.
 */
void end(
    const std::chrono::time_point<std::chrono::high_resolution_clock> &start,
    const std::string &message);

/**
 * @brief Stops timing and returns the elapsed time in seconds.
 * @param start The starting time point.
 * @return The elapsed time in seconds (double).
 */
double
end(const std::chrono::time_point<std::chrono::high_resolution_clock> &start);

/**
 * @brief Generates a random boolean value based on a given probability.
 * @param probability The probability of returning `true` (default is 0.5).
 * @return `true` or `false`.
 */
bool randomBool(double probability = 0.5);

/**
 * @brief Generates a vector of TACO ModeFormatPack, all set to the same
 * sparsity.
 * @param order The number of dimensions (tensor rank).
 * @param sparse If true, returns all `taco::Sparse`; otherwise, all
 * `taco::Dense`.
 * @return A vector of ModeFormatPack for initializing a TACO Format.
 */
std::vector<taco::ModeFormatPack> generate_modes(int order, bool sparse);
