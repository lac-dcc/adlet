#pragma once

#include "../include/utils.hpp"
#include "taco.h"
#include <memory>
#include <string>
#include <vector>

// Forward declaration of OpNode and its pointer alias
class OpNode;
using OpNodePtr = std::shared_ptr<OpNode>;

// Assume SparsityVector is defined in utils.hpp, likely as a std::bitset or
// similar structure used for the abstract domain (sparsity bitmaps).

/**
 * @brief Represents a tensor in the computational graph, encapsulating both its
 * concrete data and the abstract state used by Sparsity Propagation Analysis
 * (SPA).
 *
 * A Tensor object holds the actual TACO tensor data and the analysis result:
 * a set of **Sparsity Vectors** (bitmaps) for each dimension.
 */
class Tensor {
public:
  /// @brief The concrete tensor data, managed by TACO (Tensor Algebra
  /// Compiler).
  std::shared_ptr<taco::Tensor<float>> data;
  /// @brief The rank (number of dimensions) of the tensor.
  int numDims{};
  /**
   * @brief The core of the abstract state: A vector of **Sparsity Vectors**
   * (bitmaps), one for each dimension. SparsityVector[i][j] = 0 means the j-th
   * slice along dimension i is structurally zero (or potentially zero).
   */
  std::vector<SparsityVector> sparsities;
  /// @brief The unique name of the tensor (e.g., "T1", "O1").
  const std::string name;
  /// @brief The size of each dimension.
  std::vector<int> sizes;
  /// @brief The number of operations where this tensor is used as an input
  /// operand.
  int numOps{0};
  /// @brief Flag indicating if this tensor is an intermediate or final output
  /// tensor in the graph.
  bool outputTensor = false;

  /// @brief A list of operations (OpNodes) for which this tensor is an input.
  /// Used for Backward and Intra-Op propagation.
  std::vector<OpNodePtr> inputOps;
  /// @brief The operation (OpNode) that produces this tensor as its output.
  /// Used for propagation.
  OpNodePtr outputOp;

  /**
   * @brief Constructor for abstract tensors, primarily used for intermediate or
   * final outputs.
   *
   * @param sizes The dimensions of the tensor.
   * @param sparsities The initial abstract state (Sparsity Vectors).
   * @param n The name of the tensor.
   * @param outputTensor True if this tensor is the result of an operation.
   */
  Tensor(std::vector<int> sizes, std::vector<SparsityVector> sparsities,
         const std::string &n = "", const bool outputTensor = false);

  /**
   * @brief Constructor for an empty output tensor, initialized with all-set
   * Sparsity Vectors (fully dense abstract state).
   *
   * @param sizes The dimensions of the tensor.
   * @param n The name of the tensor.
   */
  Tensor(std::vector<int> sizes, const std::string &n = "");

  /**
   * @brief Constructor for a tensor with a predefined TACO format.
   *
   * @param sizes The dimensions of the tensor.
   * @param n The name of the tensor.
   * @param format The TACO ModeFormatPack describing the storage layout.
   */
  Tensor(std::vector<int> sizes, const std::string &n, taco::Format format);

  /**
   * @brief Constructor for an input tensor, initializing Sparsity Vectors based
   * on density ratios.
   *
   * @param sizes The dimensions of the tensor.
   * @param sparsityRatios A vector of floating-point density ratios (0.0
   * to 1.0), one for each dimension.
   * @param n The name of the tensor.
   * @param format The TACO format for the concrete data.
   */
  Tensor(std::vector<int> sizes, std::vector<float> sparsityRatios,
         const std::string &n = "",
         taco::Format format = {taco::Dense, taco::Dense});

  /**
   * @brief Creates the concrete TACO tensor data structure based on the current
   * Sparsity Vectors.
   *
   * This function uses the results of the SPA (the Sparsity Vectors) to decide
   * the
   * **mode format** (TACO::Sparse or TACO::Dense) for each dimension, applying
   * a density \p threshold.
   *
   * @param threshold The density threshold (0.0 to 1.0) to decide between Dense
   * and Sparse mode format.
   */
  void create_data(const double threshold = 0.5);

  /**
   * @brief Creates the concrete TACO tensor data structure with a specific
   * format.
   *
   * @param format The TACO ModeFormatPack to use for the storage layout.
   */
  void create_data(taco::Format format);

  /**
   * @brief Initializes the concrete tensor data by inserting non-zero values
   * only for entries where all corresponding dimension slices are marked as
   * non-zero in the `sparsities` vectors (i.e., where all bits are '1').
   */
  void initialize_data();

  /**
   * @brief Recursively generates coordinates (indices) for non-zero elements
   * based on the current state of the Sparsity Vectors.
   */
  void gen_coord(size_t d, std::vector<std::vector<int>> &indices,
                 std::vector<int> &positions);

  /// @brief Fills the concrete tensor with random values at the non-zero
  /// coordinates.
  void fill_tensor();

  /// @brief Prints the tensor's contents as a matrix (only for 2D tensors).
  void print_matrix();

  /**
   * @brief Prints the full abstract state (Sparsity Vectors) for all
   * dimensions, showing '0' for guaranteed zero slices and '1' for potentially
   * non-zero slices.
   */
  void print_full_sparsity();

  /**
   * @brief Calculates the estimated sparsity ratio (density of zero elements)
   * based on the product of non-zero slice counts in the Sparsity Vectors.
   *
   * @return The estimated sparsity ratio (0.0 to 1.0).
   */
  float get_sparsity_ratio();

  /**
   * @brief Calculates the maximum possible number of non-zero (NNZ) elements
   * based on the intersection of the Sparsity Vectors.
   *
   * @return The estimated NNZ count.
   */
  size_t get_nnz();

  /**
   * @brief Computes the estimated memory size in bytes based on the dimensions,
   * chosen mode formats, and the SPA-derived non-zero counts.
   *
   * @return The estimated memory size in bytes.
   */
  size_t compute_size_in_bytes();

  /// @brief Prints the dimension sizes of the tensor.
  void print_shape();
};

/// @brief Type alias for a shared pointer to a Tensor.
using TensorPtr = std::shared_ptr<Tensor>;
