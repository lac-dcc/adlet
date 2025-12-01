#pragma once

#include "../include/graph.hpp"
#include "../include/utils.hpp"
#include "taco/format.h"

/**
 * @brief Represents the data parsed from a benchmark file describing a sequence
 * of Einsum operations.
 *
 * This structure holds all the necessary information to construct the
 * computational graph for Sparsity Propagation Analysis (SPA), including the
 * contraction order, the Einsum strings, and the dimensions of the initial
 * tensors.
 */
struct EinsumBenchmark {
  /// The contraction path (order of tensor multiplications), e.g., [(0, 1), (0,
  /// 2)].
  std::vector<std::pair<int, int>> path;
  /// The Einsum strings for each contraction step, e.g., {"ij,jk->ik",
  /// "ik,kl->il"}.
  std::vector<std::string> strings;
  /// The dimensions/sizes for the initial set of input tensors.
  std::vector<std::vector<int>> sizes;
};

/**
 * @brief Parses a contraction path string to extract the sequence of tensor
 * indices to be contracted.
 *
 * The contraction path defines the order in which the initial tensors are
 * multiplied in the reduction process, which is critical for optimization.
 *
 * @param line The string containing the contraction path, e.g., "[(0, 1), (2,
 * 3)]".
 * @return A vector of pairs, where each pair (i, j) indicates that the i-th and
 * j-th tensors currently in the stack should be contracted.
 */
std::vector<std::pair<int, int>> get_contraction_path(const std::string &line);

/**
 * @brief Parses a line to extract the Einsum notation strings for each
 * contraction.
 *
 * These strings (e.g., "ij,jk->ik") define the input dimensions, output
 * dimensions, and reduction dimensions for a binary tensor operation.
 *
 * @param line The string containing the list of Einsum strings, typically
 * single-quoted.
 * @return A vector of strings, where each string is an Einsum expression.
 */
std::vector<std::string> get_contraction_strings(const std::string &line);

/**
 * @brief Parses a line to extract the sizes (dimensions) of the initial input
 * tensors.
 *
 * @param line The string containing the list of tensor size tuples, e.g., "[(3,
 * 4), (4, 5)]".
 * @return A vector of vectors, where each inner vector represents the
 * dimensions of an input tensor.
 */
std::vector<std::vector<int>> get_tensor_sizes(const std::string &line);

/**
 * @brief Extracts the output index variables from a given Einsum string.
 *
 * @param einsumString The Einsum string (e.g., "i,j->ij").
 * @return A string containing only the output index variables (e.g., "ij").
 */
std::string extract_outputs(std::string const &einsumString);

/**
 * @brief Extracts the input index variable strings from a given Einsum string.
 *
 * @param einsumString The Einsum string (e.g., "ij,jk->ik").
 * @return A vector of strings, where each string represents the indices of an
 * input tensor (e.g., {"ij", "jk"}).
 */
std::vector<std::string> extract_inputs(std::string const &einsumString);

/**
 * @brief Constructs a map from a dimension index variable (char) to its size
 * (int).
 *
 * This map is essential for deducing the size of the output tensor in an Einsum
 * operation, as it ensures that shared indices have a consistent size across
 * inputs.
 *
 * @param inputs A vector of strings representing the indices of the input
 * tensors (e.g., {"ij", "jk"}).
 * @param tensorSizes A vector of vectors containing the actual sizes of the
 * input tensors.
 * @return An unordered map where keys are index variables (e.g., 'i', 'j', 'k')
 * and values are their corresponding dimension sizes.
 */
std::unordered_map<char, int>
construct_size_map(std::vector<std::string> const &inputs,
                   std::vector<std::vector<int>> const &tensorSizes);

/**
 * @brief Deduces the concrete dimension sizes of the resulting output tensor
 * from a binary Einsum operation.
 *
 * Uses the index-to-size map (constructed from inputs) to look up the size for
 * each index variable in the output string.
 *
 * @param einsumString The Einsum string for the operation (e.g., "ij,jk->ik").
 * @param sizes1 The dimension sizes of the first input tensor.
 * @param sizes2 The dimension sizes of the second input tensor.
 * @return A vector of integers representing the dimension sizes of the output
 * tensor.
 */
std::vector<int> deduceOutputDims(std::string const &einsumString,
                                  std::vector<int> const &sizes1,
                                  std::vector<int> const &sizes2);

/**
 * @brief Generates TACO ModeFormatPack structures for a tensor's dimensions,
 * potentially using SPA results.
 *
 * This function determines the storage format (e.g., Dense or Sparse) for each
 * dimension (mode). When \p sparse is true, the actual implementation uses the
 * **Sparsity Propagation Analysis (SPA)** results (**Sparsity Vectors**) to
 * determine if a dimension is structurally sparse (TACO::Sparse) or fully dense
 * (TACO::Dense).
 *
 * @param order The rank (number of dimensions) of the tensor.
 * @param sparse If true, use sparsity analysis results to determine mode
 * formats.
 * @return A vector of taco::ModeFormatPack indicating the storage format for
 * each mode.
 */
std::vector<taco::ModeFormatPack> generate_modes(int order,
                                                 bool sparse = false);

/**
 * @brief Reads a benchmark file containing Einsum specifications and parses its
 * components.
 *
 * The file is expected to contain three lines: contraction path, Einsum
 * strings, and tensor sizes.
 *
 * @param filename The path to the benchmark file.
 * @return An EinsumBenchmark structure populated with the parsed data.
 */
EinsumBenchmark read_einsum_benchmark(const std::string &filename);

/**
 * @brief Constructs the computational graph (expression tree) for a sequence of
 * Einsum operations.
 *
 * This function takes the parsed benchmark data and creates the Graph structure
 * used by SPA. It initializes the input tensors with concrete **sparsity
 * vectors** (bitmaps) based on the initial \p sparsity density and then builds
 * the sequence of binary Einsum operations (nodes) following the \p
 * contractionInds path.
 *
 * @param tensorSizes The dimensions of the initial input tensors.
 * @param contractionStrings The Einsum strings for each binary contraction.
 * @param contractionInds The path specifying the order of contractions.
 * @param sparsity The initial sparsity (density) value (e.g., 0.5) used to
 * initialize the input SparsityVectors.
 * @return A Graph object representing the computational graph for the Einsum
 * sequence.
 */
Graph build_tree(const std::vector<std::vector<int>> &tensorSizes,
                 const std::vector<std::string> &contractionStrings,
                 const std::vector<std::pair<int, int>> &contractionInds,
                 const double sparsity = 0.5);
