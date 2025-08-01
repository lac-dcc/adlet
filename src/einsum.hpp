#pragma once

#include <regex>

#include "dot.hpp"
#include "graph.hpp"
#include "taco.h"
#include "taco/format.h"
#include <unordered_map>

std::vector<std::pair<int, int>> getContractionPath(const std::string &line) {
  std::vector<std::pair<int, int>> result;

  size_t pos = 0;
  while ((pos = line.find('(', pos)) != std::string::npos) {
    size_t comma = line.find(',', pos);
    size_t close = line.find(')', pos);

    int first = std::stoi(line.substr(pos + 1, comma - pos - 1));
    int second = std::stoi(line.substr(comma + 1, close - comma - 1));
    result.emplace_back(first, second);
    pos = close + 1;
  }
  return result;
}

std::vector<std::string> getContractionStrings(const std::string &line) {
  std::vector<std::string> result;

  size_t pos = 0;
  while ((pos = line.find('\'', pos)) != std::string::npos) {
    size_t close = line.find('\'', pos + 1);
    std::string einsumString = line.substr(pos + 1, close - pos - 1);
    result.emplace_back(einsumString);
    pos = close + 1;
  }
  return result;
}

std::vector<std::vector<int>> getTensorSizes(const std::string &line) {
  std::vector<std::vector<int>> result;
  std::regex tupleRegex(R"(\(([^()]*)\))");
  std::string s = line;

  std::smatch match;
  while (std::regex_search(s, match, tupleRegex)) {
    std::string content = match[1].str();
    std::vector<int> sizes;
    std::stringstream ss(content);
    std::string number;

    while (std::getline(ss, number, ',')) {
      number.erase(remove_if(number.begin(), number.end(), isspace),
                   number.end());
      sizes.push_back(std::stoi(number));
    }

    result.push_back(sizes);
    s = match.suffix();
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
    for (int j = 0; j < tensorSizes[i].size(); ++j) {
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

taco::Format getFormat(const int size) {
  std::vector<taco::ModeFormat> modes;
  for (int i = 0; i < size; i++) {
    modes.push_back(taco::Dense);
  }
  const taco::ModeFormatPack modeFormatPack(modes);
  std::vector<taco::ModeFormatPack> modeFormatPackVector{modeFormatPack};
  return taco::Format(modeFormatPackVector);
}

Graph buildTree(const std::vector<std::vector<int>> &tensorSizes,
                const std::vector<std::string> &contractionStrings,
                const std::vector<std::pair<int, int>> &contractionInds) {
  std::vector<TensorPtr> tensors;
  std::vector<TensorPtr> tensorStack;
  std::vector<OpNodePtr> ops;

  // construct tensors based on tensorSizes
  int ind = 0;
  for (auto dims : tensorSizes) {
    std::vector<bitset> sparsityVectors;
    for (auto dim : dims) {
      sparsityVectors.push_back(generate_sparsity_vector(0.0, dim));
    }
    auto newTensor = std::make_shared<Tensor>(dims, sparsityVectors,
                                              "T" + std::to_string(ind++));
    newTensor->create_data(getFormat(dims.size()));
    newTensor->initialize_data();
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
                                              "O" + std::to_string(ind++));
    newTensor->create_data(getFormat(outputDims.size()));

    tensors.push_back(newTensor);
    ops.push_back(std::make_shared<Einsum>(
        std::vector<TensorPtr>{tensorStack[ind2], tensorStack[ind1]},
        tensors.back(), contractionStrings[i]));
    tensorStack.erase(tensorStack.begin() + ind2);
    tensorStack.erase(tensorStack.begin() + ind1);
    tensorStack.push_back(tensors.back());
  }

  return Graph::build_graph(tensors, tensorStack[0], ops);
}

void readEinsumBenchmark(const std::string &filename) {

  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Failed to open file.\n";
    return;
  }
  std::string path;
  std::string contractions;
  std::string sizes;
  std::getline(file, path);
  std::getline(file, contractions);
  std::getline(file, sizes);
  auto contractionPath = getContractionPath(path);
  auto contractionStrings = getContractionStrings(contractions);
  auto tensorSizes = getTensorSizes(sizes);
  file.close();
  auto g = buildTree(tensorSizes, contractionStrings, contractionPath);
  /*print_dot(g, "teste.dot");*/
  /*g.compile();*/
  /*g.compute();*/
}
