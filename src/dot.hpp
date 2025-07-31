#include "graph.hpp"
#include <fstream>
#include <memory>
#include <string>

std::string getColor(std::shared_ptr<Tensor> tensor, bool shouldColor) {
  std::string color = std::string("black");

  if (!shouldColor)
    return color;

  float ratio = tensor->get_sparsity_ratio();
  if (ratio >= 0.7) {
    color = std::string("crimson");
  } else if (ratio >= 0.5) {
    color = std::string("blue");
  } else if (ratio >= 0.3) {
    color = std::string("darkgreen");
  } else if (ratio >= 0.1) {
    color = std::string("cyan");
  }
  return color;
}

void print_dot(const Graph &graph, std::string file_name = "graph.dot",
               bool colors = true) {
  std::ofstream file(file_name);
  file << "digraph G {\n";
  file << "  rankdir=LR;\n";

  std::unordered_map<const void *, std::string> nodeIDs;
  int id = 0;

  auto getID = [&](const void *ptr) -> std::string {
    if (nodeIDs.count(ptr))
      return nodeIDs[ptr];
    return nodeIDs[ptr] = "n" + std::to_string(id++);
  };

  for (const auto &op : graph.nodes) {
    std::string opID = getID(op.get());
    file << "  " << opID << " [label=\"" << op->op_type()
         << "\", shape=box, penwidth=2];\n";

    for (const auto &input : op->inputs) {
      file << "  " << getID(input.get()) << " [label=\""
           << input->data->getName()
           << "\", shape=ellipse, penwidth=2, color=" << getColor(input, colors)
           << "];\n";
      file << "  " << getID(input.get()) << " -> " << opID << ";\n";
    }

    file << "  " << getID(op->output.get()) << " [label=\""
         << op->output->data->getName() << "\", shape=ellipse, penwidth=2];\n";
    file << "  " << opID << " -> " << getID(op->output.get()) << ";\n";
  }
  file << "}\n";
}
