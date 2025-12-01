#pragma once

#include "../include/tensor.hpp"
#include <typeinfo> // Used in the implementation for type checking
#include <vector>

/**
 * @brief Represents an abstract base class for a node (operator) in the
 * computational graph.
 *
 * An OpNode defines a tensor operation (e.g., Addition, Einsum) that takes
 * one or more input tensors and produces a single output tensor. It is the core
 * component responsible for propagating the **abstract state** (Sparsity
 * Vectors) during the Sparsity Propagation Analysis (SPA) phase.
 */
class OpNode {
public:
  /// @brief The input tensors to this operation.
  std::vector<TensorPtr> inputs;
  /// @brief The output tensor produced by this operation.
  TensorPtr output;

  /**
   * @brief Abstract method to set up the concrete TACO tensor expression.
   *
   * This method defines the algebraic relationship between inputs and output
   * (e.g., A(i) + B(i) = C(i)).
   */
  virtual void set_expression() = 0;

  /**
   * @brief Abstract method for propagating sparsity information.
   *
   * This implements the **Transfer Function** for the specific operation and
   * direction (Forward, Backward, or Intra-Op/Lateral), which is the core of
   * the SPA fixed-point iteration.
   *
   * @param dir The direction of propagation (Direction::FORWARD,
   * Direction::INTRA, or Direction::BACKWARD).
   */
  virtual void propagate(Direction dir) = 0;

  /// @brief Abstract method to print the operation in the graph.
  virtual void print() = 0;

  /// @brief Abstract method to print the abstract sparsity state (Sparsity
  /// Vectors) for the operation.
  virtual void print_sparsity() = 0;

  /// @brief Abstract method to return the type of the operation (e.g., "Add",
  /// "Einsum").
  virtual std::string op_type() const = 0;

  /// @brief Abstract method to perform the actual tensor computation.
  virtual void compute() = 0;

  /// @brief Default destructor.
  virtual ~OpNode() = default;
};

/// @brief Type alias for a shared pointer to an OpNode.
using OpNodePtr = std::shared_ptr<OpNode>;

/**
 * @brief Represents a tensor Addition operation in the computational graph.
 *
 * In SPA, addition performs a **logical OR** ($\bigvee$) operation on the
 * Sparsity Vectors of its inputs to conservatively infer the output sparsity
 * (Forward Propagation).
 */
class Add : public OpNode {
public:
  /**
   * @brief Constructs an Add node.
   * @param inputs The input tensors.
   * @param Out The output tensor.
   */
  Add(std::vector<TensorPtr> inputs, TensorPtr &Out);

  /**
   * @brief Helper to retrieve the Sparsity Vectors for a specific dimension of
   * all inputs.
   *
   * @param inputDim The dimension index.
   * @return A vector of shared pointers to the Sparsity Vectors.
   */
  std::vector<std::shared_ptr<SparsityVector>>
  get_input_sparsity_vectors(int inputDim);

  void set_expression() override;
  void propagate(Direction dir) override;
  void print() override;
  void print_sparsity() override;
  std::string op_type() const override;
  void compute() override;

  ~Add() = default;
};

/**
 * @brief Represents an Einsum (Einstein Summation) operation.
 *
 * This is the most complex operator for SPA, implementing transfer functions
 * for Forward, Backward, and Intra-Op/Lateral propagation based on output
 * dimensions ($Od$) and reduction dimensions ($Rd$).
 */
class Einsum : public OpNode {
public:
  /// @brief The string representation of the Einsum expression (e.g.,
  /// "ij,jk->ik").
  std::string expression;
  /// @brief The output index variables (e.g., "ik").
  std::string outputInds;
  /// @brief The index variables for each input tensor (e.g., {"ij", "jk"}).
  std::vector<std::string> tensorIndicesVector;
  /**
   * @brief Map of output index char to (input index, input dimension) pairs.
   * Used for Forward and Backward propagation across output dimensions ($Od$).
   */
  std::unordered_map<char, std::vector<std::pair<int, int>>> outputDims;
  /**
   * @brief Map of reduction index char to (input index, input dimension) pairs.
   * Used for Intra-Op/Lateral propagation across reduction dimensions ($Rd$).
   */
  std::unordered_map<char, std::vector<std::pair<int, int>>> reductionDims;

  /**
   * @brief Constructs an Einsum node, parsing the index variables and setting
   * up the internal index maps (`outputDims` and `reductionDims`).
   * @param inputs The input tensors.
   * @param Out The output tensor.
   * @param expression The Einsum string.
   */
  Einsum(std::vector<TensorPtr> inputs, TensorPtr Out, std::string expression);

  /// @brief Default destructor.
  ~Einsum() = default;

  /**
   * @brief Retrieves the Sparsity Vectors for all dimensions involved in a
   * specific reduction index variable.
   * @param indexVar The reduction index character (e.g., 'j' in 'ij,jk->ik').
   * @return A vector of shared pointers to the relevant Sparsity Vectors.
   */
  std::vector<std::shared_ptr<SparsityVector>>
  get_reduction_sparsity_vectors(char indexVar);

  /**
   * @brief Retrieves the Sparsity Vectors for all dimensions corresponding to a
   * specific output index variable.
   * @param indexVar The output index character (e.g., 'i' or 'k' in
   * 'ij,jk->ik').
   * @return A vector of shared pointers to the relevant Sparsity Vectors.
   */
  std::vector<std::shared_ptr<SparsityVector>>
  get_output_sparsity_vectors(char indexVar);

  /**
   * @brief Gets the index variable character corresponding to a tensor's
   * dimension.
   * @param tensor The tensor (must be one of the inputs).
   * @param indDimension The dimension index (0-based) in the tensor.
   * @return The index character (e.g., 'j').
   */
  char get_tensor_ind_var(TensorPtr tensor, int indDimension);

  /**
   * @brief Gets the output dimension index (0-based) corresponding to an index
   * variable.
   * @param tensor The output tensor.
   * @param indexVar The index character (e.g., 'k').
   * @return The output dimension index, or -1 if the char is not an output
   * index.
   */
  int get_tensor_char_ind(TensorPtr tensor, char indexVar);

  /**
   * @brief Implements the logical OR ($\bigvee$) for Add operations (part of
   * the multi-op SPA).
   *
   * Used when a tensor is an input to an Add operation whose results are being
   * propagated to this Einsum.
   *
   * @param op Pointer to the Add operation.
   * @param inputInd The index of the tensor being propagated in the current
   * Einsum's input list.
   * @param inputDim The dimension index of the tensor being propagated.
   * @return The resulting Sparsity Vector after the OR operation.
   */
  SparsityVector or_all_operands_add(Add *op, int inputInd, int inputDim);

  /**
   * @brief Implements the logical AND ($\bigwedge$) for Einsum operations (part
   * of the multi-op SPA).
   *
   * This is part of the **Intra-Op/Lateral Propagation** rule: $S_i = S_{out}
   * \land \bigwedge_{j \ne i} S_j$. It calculates the $\bigwedge_{j \ne i} S_j$
   * component.
   *
   * @param einsumOp Pointer to the Einsum operation where propagation occurs.
   * @param inputInd Index of the tensor being propagated in this Einsum's
   * inputs.
   * @param inputDim Dimension index of the tensor being propagated.
   * @return The resulting Sparsity Vector after the AND operation.
   */
  SparsityVector and_all_operands_einsum(Einsum *einsumOp, int inputInd,
                                         int inputDim);

  /**
   * @brief Retrieves the output sparsity component ($S_{out}$) for an Einsum
   * transfer function.
   *
   * This is part of the **Intra-Op/Lateral Propagation** rule.
   *
   * @param einsumOp Pointer to the Einsum operation.
   * @param inputInd Index of the tensor being propagated.
   * @param inputDim Dimension index of the tensor being propagated.
   * @return The output Sparsity Vector $S_{out}$ if the dimension is an output
   * dimension ($Od$), otherwise an empty set.
   */
  SparsityVector op_output_sparsity_einsum(Einsum *einsumOp, int inputInd,
                                           int inputDim);

  /**
   * @brief General method to propagate sparsity for a dimension during intra-op
   * analysis, considering a tensor that might be the output of an Add or Einsum
   * op.
   *
   * @param op Pointer to the operation (Add or Einsum) that produced the
   * tensor.
   * @param inputInd Index of the tensor being propagated in this Einsum's
   * inputs.
   * @param inputDim Dimension index of the tensor being propagated.
   * @return The propagated Sparsity Vector.
   */
  SparsityVector propagate_intra_multiop(OpNodePtr op, int inputInd,
                                         int inputDim);

  /**
   * @brief Calculates the sparsity contribution from a single dimension during
   * propagation.
   *
   * This involves looking at all operations that consume the tensor and
   * combining their propagation results.
   *
   * @param inputInd Index of the input tensor in the current Einsum.
   * @param inputDim Dimension index of the input tensor.
   * @param indexChar The index variable character (e.g., 'i').
   * @return The resulting Sparsity Vector for the dimension.
   */
  SparsityVector propagate_intra_dimension(int inputInd, int inputDim,
                                           char indexChar);

  /**
   * @brief Computes the sparsity vector for an input dimension when propagating
   * from an *output* of an Einsum operation.
   */
  SparsityVector compute_multiop_einsum_sparsity(Einsum *opPtr, int inputInd,
                                                 int inputDim);

  /**
   * @brief Computes the sparsity vector for an input dimension when propagating
   * from an *output* of an Add operation.
   */
  SparsityVector compute_multiop_add_sparsity(Add *opPtr, int inputInd,
                                              int inputDim);

  /**
   * @brief Dispatches to the correct multi-op sparsity computation function
   * based on operator type.
   */
  SparsityVector compute_multiop_sparsity(OpNode *opPtr, int inputInd,
                                          int inputDim);

  void propagate(Direction dir) override;
  /**
   * @brief Implements the **Forward Propagation** transfer function for Einsum.
   *
   * Uses logical AND ($\bigwedge$) across all input dimensions mapped to the
   * same output dimension index.
   */
  void propagate_forward();
  /**
   * @brief Implements the **Intra-Op/Lateral Propagation** transfer function
   * for Einsum.
   *
   * Propagates sparsity across reduction dimensions ($Rd$) between inputs.
   */
  void propagate_intra();
  /**
   * @brief Implements the **Backward Propagation** transfer function for
   * Einsum.
   *
   * Propagates sparsity from the output back to the input dimensions that
   * appear in the output ($Od$).
   */
  void propagate_backward();

  void set_expression() override;
  void print() override;
  void print_sparsity() override;
  std::string op_type() const override;
  void compute() override;
};
