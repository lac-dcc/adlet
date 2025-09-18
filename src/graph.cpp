#include "../include/graph.hpp"
Graph Graph::build_graph(std::vector<TensorPtr> inputs, TensorPtr out,
                         const std::vector<OpNodePtr> &ops) {
  Graph g;
  g.inputs = inputs;
  g.output = out;
  g.nodes = ops;
  for (auto op : ops) {
    for (auto input : op->inputs) {
      input->inputOps.push_back(op);
    }
    op->output->outputOp = op;
  }
  return g;
}

void Graph::run_propagation() {
  run_propagation(Direction::FORWARD);
  run_propagation(Direction::INTRA);
  run_propagation(Direction::BACKWARD);
}

void Graph::run_propagation(Direction dir) {
  if (dir == INTRA || dir == BACKWARD) {
    std::vector<OpNodePtr> intraStack{output->outputOp};
    std::unordered_map<TensorPtr, bool> doneProp;
    while (intraStack.size() > 0) {
      auto op = intraStack.back();
      intraStack.pop_back();
      op->propagate(dir);
      for (auto input : op->inputs) {
        doneProp[input] = true;

        if (!input->outputOp)
          continue;
        bool allDone{false};
        for (auto otherInput : input->outputOp->inputs) {
          for (auto otherOp : otherInput->inputOps)
            allDone |= doneProp[otherOp->output];
        }
        if (allDone)
          intraStack.push_back(input->outputOp);
      }
    }
  } else {
    for (auto &op : nodes)
      op->propagate(dir);
  }
}

void Graph::assemble_expressions() {
  for (auto &op : nodes)
    op->set_expression();
}

void Graph::compile() {
  assemble_expressions();
  this->output->data->compile();
}

TensorPtr Graph::compute() {
  for (auto &op : nodes)
    op->compute();
  this->output->data->assemble();
  this->output->data->compute();
  return this->output;
}

void Graph::print() {
  for (auto &input : this->inputs) {
    std::cout << input->name << ",";
  }
  for (auto &op : this->nodes) {
    op->print();
  }
  std::cout << "->" << this->output->name << std::endl;
}

void Graph::print_sparsity() {
  for (auto &op : this->nodes) {
    op->print_sparsity();
  }
}

float Graph::get_sparsity_ratio() {
  size_t count = 0;
  float total_ratio = 0;
  for (auto &ops : this->nodes) {
    for (auto &input : ops->inputs) {
      count++;
      total_ratio += input->get_sparsity_ratio();
    }
  }
  total_ratio += this->output->get_sparsity_ratio();
  count++;
  return total_ratio / count;
}

void Graph::get_tensor_sizes() {
  size_t total_size = 0;
  for (auto t : this->inputs)
    total_size += t->data->getStorage().getSizeInBytes();
  for (auto &op : nodes)
    total_size += op->output->data->getStorage().getSizeInBytes();
  total_size += this->output->data->getStorage().getSizeInBytes();
  std::cout << "tensors size = " << total_size / (1024.0 * 1024.0) << std::endl;
}
