#include "graph.hpp"
#include <fstream>

void print_dot(const Graph &graph, std::string file_name = "graph.dot") {
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
         << "\", shape=box];\n";

    for (const auto &input : op->inputs) {
      file << "  " << getID(input.get()) << " [label=\""
           << input->data->getName() << "\", shape=ellipse];\n";
      file << "  " << getID(input.get()) << " -> " << opID << ";\n";
    }

    file << "  " << getID(op->output.get()) << " [label=\""
         << op->output->data->getName() << "\", shape=ellipse];\n";
    file << "  " << opID << " -> " << getID(op->output.get()) << ";\n";
  }
  file << "}\n";
}
