#include "../include/einsum.hpp"
#include "../include/graph.hpp"
#include "../include/node.hpp"
#include "../include/tensor.hpp"
#include <fstream>
#include <regex>

#include "taco/format.h"

std::vector<std::pair<int, int>> get_contraction_path(const std::string &line) {
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

std::vector<std::string> get_contraction_strings(const std::string &line) {
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

std::vector<std::vector<int>> get_tensor_sizes(const std::string &line) {
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

std::string extract_outputs(std::string const &einsumString) {
  int arrowPos = einsumString.find("->");
  return einsumString.substr(arrowPos + 2);
}

std::vector<std::string> extract_inputs(std::string const &einsumString) {
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
construct_size_map(std::vector<std::string> const &inputs,
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
  std::string output = extract_outputs(einsumString);
  std::vector<std::string> inputs = extract_inputs(einsumString);
  std::unordered_map<char, int> sizeMap =
      construct_size_map(inputs, {sizes2, sizes1});
  std::vector<int> outputSizes;

  for (auto indVar : output)
    outputSizes.push_back(sizeMap[indVar]);

  return outputSizes;
}

std::vector<taco::ModeFormatPack>
generate_modes(int order, std::vector<int> sizes,
               std::vector<SparsityVector> sparsities, bool sparse) {
  std::vector<taco::ModeFormatPack> modes;
  for (int j = 0; j < order; ++j) {
    if (!sparse)
      modes.push_back(taco::Dense);
    else {
      modes.push_back(count_bits(sparsities[j], sizes[j]) != sizes[j]
                          ? taco::Dense
                          : taco::Sparse);
    }
  }
  return modes;
}

Graph build_tree(const std::vector<std::vector<int>> &tensorSizes,
                 const std::vector<std::string> &contractionStrings,
                 const std::vector<std::pair<int, int>> &contractionInds,
                 const double sparsity) {
  std::vector<TensorPtr> inputTensors;
  std::vector<TensorPtr> tensorStack;
  std::vector<OpNodePtr> ops;
  // construct tensors based on tensorSizes
  int ind = 1;
  for (auto dims : tensorSizes) {
    auto newTensor =
        std::make_shared<Tensor>(dims, "T" + std::to_string(ind++));
    inputTensors.push_back(newTensor);
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
    auto t1 = tensorStack[ind1];
    auto t2 = tensorStack[ind2];

    bool prune = false;
    if (!t1->outputTensor) {
      std::vector<SparsityVector> sparsityVectors;
      for (auto dim : t1->sizes) {
        sparsityVectors.push_back(generate_sparsity_vector(sparsity, dim));
      }
      t1->sparsities = sparsityVectors;
      prune = true;
    } else if (!t2->outputTensor && !prune) {
      std::vector<SparsityVector> sparsityVectors;
      for (auto dim : t2->sizes) {
        sparsityVectors.push_back(generate_sparsity_vector(sparsity, dim));
      }
      t2->sparsities = sparsityVectors;
    }

    std::vector<int> outputDims =
        deduceOutputDims(contractionStrings[i], tensorStack[ind1]->sizes,
                         tensorStack[ind2]->sizes);

    std::vector<SparsityVector> sparsityVectors;
    for (auto dim : outputDims) {
      sparsityVectors.push_back(generate_sparsity_vector(0.0, dim));
    }

    auto newTensor = std::make_shared<Tensor>(
        outputDims, sparsityVectors, "O" + std::to_string(ind++), true);

    ops.push_back(std::make_shared<Einsum>(
        std::vector<TensorPtr>{tensorStack[ind2], tensorStack[ind1]}, newTensor,
        contractionStrings[i]));
    tensorStack.erase(tensorStack.begin() + ind2);
    tensorStack.erase(tensorStack.begin() + ind1);
    tensorStack.push_back(newTensor);
  }

  return Graph::build_graph(inputTensors, tensorStack[0], ops);
}

EinsumBenchmark read_einsum_benchmark(const std::string &filename) {

  EinsumBenchmark result;
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Failed to open file.\n";
    return result;
  }
  std::string path;
  std::string contractions;
  std::string sizes;
  std::getline(file, path);
  std::getline(file, contractions);
  std::getline(file, sizes);
  file.close();
  auto contractionPath = get_contraction_path(path);
  auto contractionStrings = get_contraction_strings(contractions);
  auto tensorSizes = get_tensor_sizes(sizes);
  result.sizes = tensorSizes;
  result.path = contractionPath;
  result.strings = contractionStrings;
  return result;
}
