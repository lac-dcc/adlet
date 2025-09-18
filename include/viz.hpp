#pragma once

#include "graph.hpp"

std::string get_color(std::shared_ptr<Tensor> tensor, bool shouldColor);
void write_dot(const Graph &graph, std::string file_name = "graph.dot",
               bool colors = true);
