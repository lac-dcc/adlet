#pragma once
#include "dot.hpp"
#include "graph.hpp"
#include "taco.h"
#include "taco/format.h"
#include "utils.hpp"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::pair<int, int>> readIndices(const std::string &filename) {
  std::vector<std::pair<int, int>> result;
  std::ifstream file(filename);

  std::string line;
  std::getline(file, line);
  line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

  size_t pos = 0;
  while ((pos = line.find('(', pos)) != std::string::npos) {
    size_t comma = line.find(',', pos);
    size_t close = line.find(')', pos);

    int first = std::stoi(line.substr(pos + 1, comma - pos - 1));
    int second = std::stoi(line.substr(comma + 1, close - comma - 1));
    result.emplace_back(first, second);
    pos = close + 1;
  }

  for (auto p : result)
    std::cout << p.first << ", " << p.second << " ";
  std::cout << std::endl;

  file.close();
  return result;
}

std::vector<std::string> readContractionStrings(const std::string &filename) {
  std::vector<std::string> result;
  std::ifstream file(filename);

  std::cout << file.is_open() << std::endl;

  std::string line;
  std::getline(file, line);
  line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

  std::cout << line << std::endl;
  size_t pos = 0;
  while ((pos = line.find('\'', pos)) != std::string::npos) {
    size_t close = line.find('\'', pos + 1);
    std::string einsumString = line.substr(pos + 1, close - pos - 1);
    std::cout << pos << " " << close << " " << einsumString << std::endl;
    result.emplace_back();
    pos = close + 1;
  }

  return result;
}

std::string extractOutputs(std::string const &einsumString) {
  int arrowPos = einsumString.find("->");
  return einsumString.substr(arrowPos + 2);
}

std::vector<std::string> extractInputs(std::string const &einsumString) {
  std::vector<std::string> inputs;
  int arrowPos = einsumString.find("->");
  std::string lhs = einsumString.substr(0, arrowPos);

  std::stringstream ss(lhs);
  std::string token;
  while (std::getline(ss, token, ','))
    inputs.push_back(token);

  return inputs;
}

std::unordered_map<char, int>
constructSizeMap(std::vector<std::string> const &inputs,
                 std::vector<std::vector<int>> const &tensorSizes) {
  std::unordered_map<char, int> sizeMap;

  for (int i = 0; i < tensorSizes.size(); ++i) {
    for (int j = 0; j < tensorSizes.size(); ++j) {
      auto indVar = inputs[i][j];
      auto dimSize = tensorSizes[i][j];
      sizeMap[indVar] = dimSize;
    }
  }
  return sizeMap;
}

std::vector<int> deduceOutputDims(std::string const &einsumString,
                                  std::vector<int> const &sizes1,
                                  std::vector<int> const &sizes2) {
  std::string output = extractOutputs(einsumString);
  std::vector<std::string> inputs = extractInputs(einsumString);
  std::unordered_map<char, int> sizeMap =
      constructSizeMap(inputs, {sizes1, sizes2});
  std::vector<int> outputSizes;

  for (auto indVar : output)
    outputSizes.push_back(sizeMap[indVar]);

  return outputSizes;
}

Graph buildTree(std::vector<std::vector<int>> tensorSizes,
                std::vector<std::string> const &contractionStrings,
                std::vector<std::pair<int, int>> const &contractionInds) {
  std::vector<TensorPtr> tensors;
  std::vector<TensorPtr> tensorStack;
  std::vector<OpNodePtr> ops;

  // construct tensors based on tensorSizes
  int ind = 0;
  for (auto dims : tensorSizes) {
    auto denseSparsityVector = generate_sparsity_vector(0.0, size);
    std::vector<bitset> sparsityVectors;
    for (auto dim : dims) {
      sparsityVectors.push_back(generate_sparsity_vector(0.0, dim));
    }
    auto newTensor =
        std::make_shared<Tensor>(dims, sparsityVectors, std::to_string(ind++));
    tensors.push_back(newTensor);
    tensorStack.push_back(newTensor);
  }

  // add einsum to list of ops and new tensor to tensors vector
  for (int i = 0; i < contractionStrings.size(); ++i) {
    int ind1 = contractionInds[i].first < contractionInds[i].second
                   ? contractionInds[i].first
                   : contractionInds[i].second;
    int ind2 = contractionInds[i].first > contractionInds[i].second
                   ? contractionInds[i].first
                   : contractionInds[i].second;
    std::vector<int> outputDims =
        deduceOutputDims(contractionStrings[i], tensorStack[ind1]->sizes,
                         tensorStack[ind2]->sizes);

    std::vector<bitset> sparsityVectors;
    for (auto dim : outputDims)
      sparsityVectors.push_back(generate_sparsity_vector(0.0, dim));

    auto newTensor = std::make_shared<Tensor>(outputDims, sparsityVectors,
                                              std::to_string(ind++));

    tensors.push_back(newTensor);
    ops.push_back(std::make_shared<Einsum>(
        std::vector<TensorPtr>{tensorStack[ind1], tensorStack[ind2]},
        tensors.back(), contractionStrings[i]));
    tensorStack.erase(tensorStack.begin() + ind2);
    tensorStack.erase(tensorStack.begin() + ind1);
    tensorStack.push_back(tensors.back());
  }

  return Graph::build_graph(tensors, tensorStack[0], ops);
}
