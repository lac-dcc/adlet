#pragma once

#include "../include/node.hpp"

/**
 * @brief Represents the computational graph for tensor operations, serving as
 * the central structure for Sparsity Propagation Analysis (SPA).
 *
 * The Graph encapsulates the input tensors, the sequence of operations
 * (OpNodes), and the final output tensor. It manages the entire lifecycle of
 * the SPA, including the iterative propagation of sparsity information in three
 * directions (Forward, Backward, and Intra-Op/Lateral).
 */
class Graph {

public:
  /// @brief A list of all operation nodes (e.g., Einsum) in the graph.
  std::vector<OpNodePtr> nodes;
  /// @brief The initial input tensors to the computational graph.
  std::vector<TensorPtr> inputs;
  /// @brief The final output tensor of the entire computation.
  TensorPtr output;

  /**
   * @brief Factory method to construct and initialize the computational graph.
   *
   * This method establishes the connections (edges) between Tensors and
   * OpNodes, setting up the `inputOps` and `outputOp` pointers for traversal
   * during SPA.
   *
   * @param inputs The set of initial tensors.
   * @param out The final result tensor.
   * @param ops The ordered sequence of operations (e.g., Einsum nodes) in the
   * graph.
   * @return A fully constructed Graph object.
   */
  static Graph build_graph(std::vector<TensorPtr> inputs, TensorPtr out,
                           const std::vector<OpNodePtr> &ops);

  /// @brief Default destructor.
  ~Graph() = default;

  /**
   * @brief Executes the complete Sparsity Propagation Analysis (SPA) by running
   * propagation in all three directions: Forward, Intra-Op/Lateral, and
   * Backward.
   *
   * The analysis is performed iteratively until a fixed point is reached for
   * the sparsity vectors across all tensors.
   */
  void run_propagation();

  /**
   * @brief Executes sparsity propagation in a specific direction.
   *
   * This method implements the transfer functions (propagation logic) defined
   * in the `OpNode`s for the specified direction, which are the core of SPA.
   *
   * @param dir The direction of propagation: Direction::FORWARD,
   * Direction::INTRA (Lateral), or Direction::BACKWARD.
   */
  void run_propagation(Direction dir);

  /**
   * @brief Assembles the TACO expressions for all operations in the graph.
   *
   * This step is necessary to prepare the expressions for compilation and
   * computation after SPA has determined the optimal mode formats.
   */
  void assemble_expressions();

  /**
   * @brief Compiles the assembled TACO expressions for efficient execution.
   *
   * The sparsity information (i.e., the mode formats) determined by SPA is now
   * locked in and used by the TACO compiler.
   */
  void compile();

  /**
   * @brief Computes the result of the entire tensor expression defined by the
   * graph.
   *
   * @return A pointer to the resulting output tensor.
   */
  TensorPtr compute();

  /**
   * @brief Prints a textual representation of the computational graph, showing
   * the inputs, operations, and the output.
   */
  void print();

  /**
   * @brief Prints the abstract sparsity state (Sparsity Vectors/bitmaps) for
   * all tensors in the graph, as determined by the SPA.
   */
  void print_sparsity();

  /**
   * @brief Calculates the average sparsity ratio (density) across all
   * intermediate and input tensors in the graph.
   *
   * @return The overall sparsity ratio as a float.
   */
  float get_sparsity_ratio();

  /**
   * @brief Computes and prints the total memory size of all tensors in the
   * graph in megabytes.
   */
  void get_tensor_sizes();
};
