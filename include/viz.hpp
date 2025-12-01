/**
 * @file viz.hpp
 * @brief Utilities for visualizing the computational graph using the DOT file
 * format (Graphviz).
 *
 * This file provides functions to output the dependency graph (Graph)
 * and the sparsity state of Tensors into a human-readable format,
 * which is particularly useful for debugging and analyzing the
 * Sparsity Propagation Analysis (SPA) results.
 */

#pragma once

#include "graph.hpp"
#include <memory>
#include <string>

/**
 * @brief Determines the color of a tensor node based on its sparsity ratio.
 *
 * This function implements a specific color-coding scheme to visually
 * represent the sparsity level (ratio of zero elements) of a Tensor in the
 * generated DOT graph.
 *
 * - **Crimson**: Highly sparse (ratio >= 0.7)
 * - **Blue**: Moderately sparse (ratio >= 0.5)
 * - **Dark Green**: Slightly sparse (ratio >= 0.3)
 * - **Cyan**: Low sparsity (ratio >= 0.1)
 * - **Black**: Otherwise (e.g., low sparsity or disabled coloring)
 *
 * @param tensor The Tensor object whose sparsity ratio will be checked.
 * @param shouldColor If false, always returns "black", effectively disabling
 * color-coding.
 * @return A string representing a standard DOT graph color name.
 */
std::string get_color(std::shared_ptr<Tensor> tensor, bool shouldColor);

/**
 * @brief Generates and writes the computational graph to a DOT file.
 *
 * This function traverses the Graph structure and outputs the OpNodes (boxes)
 * and Tensors (ellipses) as nodes and edges in the Graphviz DOT format.
 *
 * @param graph The Graph object representing the computational graph.
 * @param file_name The name of the output DOT file (default is "graph.dot").
 * @param colors If true, tensor nodes are color-coded based on their sparsity
 * ratio (as determined by get_color()).
 */
void write_dot(const Graph &graph, std::string file_name = "graph.dot",
               bool colors = true);
