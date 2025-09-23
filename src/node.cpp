#include "../include/node.hpp"
#include "taco/format.h"
#include "taco/parser/einsum_parser.h"

Add::Add(std::vector<TensorPtr> inputs, TensorPtr &Out) {
  this->inputs = inputs;
  this->output = Out;
  this->output->outputTensor = true;
  for (auto &input : inputs)
    input->numOps++;
}

void Add::set_expression() {
  std::vector<taco::IndexVar> inds(output->numDims);
  for (auto &input : inputs)
    (*output->data)(inds) += (*input->data)(inds);
  this->output->data->compile();
}

void Add::propagate(Direction dir) {
  if (dir == FORWARD) {
    for (int dim = 0; dim < output->numDims; ++dim) {
      SparsityVector inputSparsity;
      for (auto input : inputs)
        inputSparsity |= input->sparsities[dim];

      output->sparsities[dim] &= inputSparsity;
    }
  }
}

void Add::print() {
  std::cout << "->Add(";
  for (int i = 0; i < inputs.size(); ++i) {
    std::cout << inputs[i]->name;
    if (i != inputs.size() - 1)
      std::cout << ", ";
  }
  std::cout << ", out=" << output->name << ")";
}

std::vector<std::shared_ptr<SparsityVector>>
Add::get_input_sparsity_vectors(int inputDim) {
  std::vector<std::shared_ptr<SparsityVector>> ret;
  for (auto input : inputs)
    ret.push_back(
        std::make_shared<SparsityVector>(input->sparsities[inputDim]));
  return ret;
}

void Add::print_sparsity() {
  for (int i = 0; i < inputs.size(); ++i) {
    inputs[i]->print_full_sparsity();
    if (i != inputs.size() - 1)
      std::cout << "+";
  }
  std::cout << " = " << std::endl;
  output->print_full_sparsity();
  std::cout << std::endl;
}
std::string Add::op_type() const { return "Add"; }

void Add::compute() {
  this->output->data->assemble();
  this->output->data->compute();
}

Einsum::Einsum(std::vector<TensorPtr> inputs, TensorPtr Out,
               std::string expression) {
  this->inputs = inputs;
  for (auto &input : inputs)
    input->numOps++;
  this->expression = expression;
  this->output = Out;
  this->output->outputTensor = true;

  int arrowPos = expression.find("->");
  std::string lhs = expression.substr(0, arrowPos);
  outputInds = expression.substr(arrowPos + 2);

  std::stringstream ss(lhs);
  std::string token;

  while (std::getline(ss, token, ','))
    tensorIndicesVector.push_back(token);

  for (char c : outputInds) {
    for (int i = 0; i < tensorIndicesVector.size(); ++i) {
      int pos = tensorIndicesVector[i].find(c);
      if (pos != std::string::npos)
        outputDims[c].push_back(std::make_pair(i, pos));
    }
  }

  for (int i = 0; i < tensorIndicesVector.size(); ++i) {
    for (int j = 0; j < tensorIndicesVector[i].size(); ++j) {
      char c = tensorIndicesVector[i][j];

      if (outputDims.find(c) != outputDims.end())
        continue;
      if (reductionDims.find(c) != reductionDims.end())
        continue;

      reductionDims[c].push_back(std::make_pair(i, j));

      for (int k = i + 1; k < tensorIndicesVector.size(); ++k) {
        int pos = tensorIndicesVector[k].find(c);
        if (pos != std::string::npos) {
          reductionDims[c].push_back(std::make_pair(k, pos));
        }
      }
    }
  }
}

std::vector<std::shared_ptr<SparsityVector>>
Einsum::get_reduction_sparsity_vectors(char indexVar) {
  std::vector<std::shared_ptr<SparsityVector>> ret;
  for (auto tensorLoc : reductionDims[indexVar])
    ret.push_back(std::make_shared<SparsityVector>(
        inputs[tensorLoc.first]->sparsities[tensorLoc.second]));

  return ret;
}

std::vector<std::shared_ptr<SparsityVector>>
Einsum::get_output_sparsity_vectors(char indexVar) {
  std::vector<std::shared_ptr<SparsityVector>> ret;
  for (auto tensorLoc : outputDims[indexVar])
    ret.push_back(std::make_shared<SparsityVector>(
        inputs[tensorLoc.first]->sparsities[tensorLoc.second]));

  return ret;
}

char Einsum::get_tensor_ind_var(TensorPtr tensor, int indDimension) {
  char ret{'?'};
  for (int i = 0; i < inputs.size(); ++i) {
    if (tensor.get() == inputs[i].get()) {
      ret = tensorIndicesVector[i][indDimension];
      break;
    }
  }
  assert(ret != '?' && "Tensor has to be an input to use this function!");
  return ret;
}

int Einsum::get_tensor_char_ind(TensorPtr tensor, char indexVar) {
  int ind = -1;
  for (int i = 0; i < outputInds.length(); ++i) {
    if (outputInds[i] == indexVar) {
      ind = i;
      break;
    }
  }
  return ind;
}

void Einsum::set_expression() {
  std::vector<taco::TensorBase> tensors;
  for (auto input : inputs)
    tensors.push_back(*input->data);
  taco::Format format{output->data->getStorage().getFormat()};
  taco::parser::EinsumParser parser(expression, tensors, format,
                                    taco::Datatype::Float32);
  parser.parse();
  std::string name = output->data->getName();
  output->data =
      std::make_shared<taco::Tensor<float>>(parser.getResultTensor());
  output->data->setName(name);
  this->output->data->compile();
}

void Einsum::propagate_forward() {
  if (output->numDims == 0)
    return;
  for (int i = 0; i < outputInds.length(); ++i) {
    SparsityVector inputSparsityVector;
    inputSparsityVector.set();

    char c = outputInds[i];
    for (auto p : outputDims[c]) {
      int inputInd = p.first;  // which of the inputs
      int inputDim = p.second; // which dimension
      inputSparsityVector &= inputs[inputInd]->sparsities[inputDim];
    }
    output->sparsities[i] &= inputSparsityVector;
  }
}

void Einsum::propagate_intra() {
  for (auto kv : reductionDims) { // iterate over character: pair(inputInd,
                                  // inputDim) map.
    for (auto p : kv.second) {
      int inputInd = p.first;  // which of the inputs
      int inputDim = p.second; // which dimension
      inputs[inputInd]->sparsities[inputDim] &=
          propagate_intra_dimension(inputInd, inputDim, kv.first);
    }
  }
}

void Einsum::propagate_backward() {
  for (auto kv : outputDims) { // iterate over character: pair(inputInd,
                               // inputDim) map.
    for (auto p : kv.second) {
      int inputInd = p.first;  // which of the inputs
      int inputDim = p.second; // which dimension
      inputs[inputInd]->sparsities[inputDim] &=
          propagate_intra_dimension(inputInd, inputDim, kv.first);
    }
  }
}

void Einsum::propagate(Direction dir) {
  switch (dir) {
  case FORWARD:
    propagate_forward();
    break;
  case INTRA:
    if (inputs.size() >= 2)
      propagate_intra();
    break;
  case BACKWARD:
    propagate_backward();
    break;
  }
}
// op: pointer to Add
// inputInd: the location of the input propagating to in THIS Einsum
// inputDim: the dim of the input propagating to in THIS Einsum
SparsityVector Einsum::or_all_operands_add(Add *op, int inputInd,
                                           int inputDim) {
  SparsityVector inputSparsityVector;
  for (auto input : op->inputs) { // go through ops in the addition and skip
                                  // the current one
    if (input.get() == inputs[inputInd].get())
      continue;
    inputSparsityVector |= input->sparsities[inputDim]; // all operands have the
                                                        // same dimensionality
  }
  return inputSparsityVector;
}

// inputInd: the location of the input propagating to in THIS Einsum
// inputDim: the dim of the input propagating to in THIS Einsum
SparsityVector Einsum::and_all_operands_einsum(Einsum *einsumOp, int inputInd,
                                               int inputDim) {
  SparsityVector inputSparsityVector;
  inputSparsityVector.set();
  int currInd{};
  char currChar{};
  // find the index of this operand in einsumOp, save into currInd
  // use currInd to get the char IndexVar of this tensor, save to currChar
  for (int i = 0; i < einsumOp->inputs.size(); ++i) {
    if (einsumOp->inputs[i].get() != inputs[inputInd].get())
      continue;
    currInd = i;
    currChar = einsumOp->tensorIndicesVector[currInd][inputDim];
    break;
  }
  // iterate all reduction vars corresponding to this char
  for (auto loc : einsumOp->reductionDims[currChar]) {
    if (einsumOp->inputs[loc.first].get() == inputs[inputInd].get())
      continue;
    // einsumOp->inputs[loc.first]->name << ")" << std::endl;
    inputSparsityVector &= einsumOp->inputs[loc.first]->sparsities[loc.second];
  }
  return inputSparsityVector;
}

// returns op output sparsity for the corresponding dimension or empty set
// if not in output
SparsityVector Einsum::op_output_sparsity_einsum(Einsum *einsumOp, int inputInd,
                                                 int inputDim) {
  char outputChar = '?';
  for (int i = 0; i < einsumOp->inputs.size(); ++i) {
    auto einsumInputTensor = einsumOp->inputs[i];
    if (einsumInputTensor.get() != inputs[inputInd].get())
      continue;
    outputChar = einsumOp->tensorIndicesVector[i][inputDim];
  }
  assert(outputChar != '?');
  int outputInd = -1;
  for (int i = 0; i < einsumOp->outputInds.length(); ++i) {
    if (einsumOp->outputInds[i] == outputChar) {
      outputInd = i;
      break;
    }
  }

  if (outputInd == -1)
    return SparsityVector(); // not in the output: return empty bitset
  return einsumOp->output->sparsities[outputInd];
}

SparsityVector Einsum::propagate_intra_multiop(OpNodePtr op, int inputInd,
                                               int inputDim) {
  SparsityVector inputSparsityVector; // start off 0
  OpNode *opPtr = op.get();
  if (typeid(*opPtr) == typeid(Add)) {
    Add *addPtr = dynamic_cast<Add *>(opPtr);
    auto addSparsityVectors = addPtr->get_input_sparsity_vectors(inputDim);
    for (int i = 0; i < addSparsityVectors.size(); ++i) {
      if (i == inputInd)
        continue;
      inputSparsityVector |= *addSparsityVectors[i];
    }
  } else if (typeid(*opPtr) == typeid(Einsum)) {
    Einsum *einsumOp = dynamic_cast<Einsum *>(op.get());
    inputSparsityVector |=
        and_all_operands_einsum(einsumOp, inputInd, inputDim);
    inputSparsityVector |=
        op_output_sparsity_einsum(einsumOp, inputInd, inputDim);
  }
  return inputSparsityVector;
}

SparsityVector Einsum::propagate_intra_dimension(int inputInd, int inputDim,
                                                 char indexChar) {
  SparsityVector inputSparsityVector;

  for (auto op : inputs[inputInd]->inputOps) {
    auto opPtr = op.get();
    inputSparsityVector |= compute_multiop_sparsity(opPtr, inputInd, inputDim);
  }

  return inputSparsityVector;
}

SparsityVector Einsum::compute_multiop_einsum_sparsity(Einsum *opPtr,
                                                       int inputInd,
                                                       int inputDim) {
  char indexVar = opPtr->tensorIndicesVector[inputInd][inputDim];
  SparsityVector inputSparsityVector;

  if (opPtr->outputDims.find(indexVar) != opPtr->outputDims.end()) {
    int ind = get_tensor_char_ind(opPtr->output, indexVar);
    inputSparsityVector = opPtr->output->sparsities[ind];
  } else {
    assert(opPtr->reductionDims.find(indexVar) != opPtr->reductionDims.end());
    auto pairs = opPtr->reductionDims[indexVar];
    inputSparsityVector.set();
    for (auto p : pairs) {
      int otherInputInd = p.first;  // which of the inputs
      int otherInputDim = p.second; // which dimension
      inputSparsityVector &=
          opPtr->inputs[otherInputInd]->sparsities[otherInputDim];
    }
  }

  return inputSparsityVector;
}

SparsityVector Einsum::compute_multiop_add_sparsity(Add *opPtr, int inputInd,
                                                    int inputDim) {
  return opPtr->output
      ->sparsities[inputDim]; // addition can only have output dimension
}

SparsityVector Einsum::compute_multiop_sparsity(OpNode *opPtr, int inputInd,
                                                int inputDim) {
  SparsityVector inputSparsityVector;

  if (typeid(*opPtr) == typeid(Add)) {
    Add *addPtr = dynamic_cast<Add *>(opPtr);
    inputSparsityVector =
        compute_multiop_add_sparsity(addPtr, inputInd, inputDim);
  } else if (typeid(*opPtr) == typeid(Einsum)) {
    Einsum *einPtr = dynamic_cast<Einsum *>(opPtr);
    inputSparsityVector =
        compute_multiop_einsum_sparsity(einPtr, inputInd, inputDim);
  }

  return inputSparsityVector;
}

void Einsum::print() {
  std::cout << "->Einsum[" << this->expression << "](";
  for (int i = 0; i < this->inputs.size(); ++i) {
    std::cout << this->inputs[i]->name;
    if (i != this->inputs.size() - 1)
      std::cout << ", ";
  }
  std::cout << ", out=" << this->output->name << ")";
}

void Einsum::print_sparsity() {
  for (int i = 0; i < this->inputs.size(); ++i) {
    this->inputs[i]->print_full_sparsity();
    if (i != this->inputs.size() - 1)
      std::cout << ",";
  }
  std::cout << " = " << std::endl;
  this->output->print_full_sparsity();
  std::cout << std::endl;
}

std::string Einsum::op_type() const { return "Einsum"; }

void Einsum::compute() {
  this->output->data->assemble();
  this->output->data->compute();
}
