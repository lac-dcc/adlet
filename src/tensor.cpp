#include "../include/tensor.hpp"

void Tensor::create_data(const double threshold) {
  taco::ModeFormat sparse = taco::Sparse;
  taco::ModeFormat dense = taco::Dense;
  std::vector<taco::ModeFormatPack> modes;
  for (size_t dim = 0; dim < this->numDims; dim++) {
    int dimSize = this->sizes[dim];
    size_t bits = count_bits(this->sparsities[dim], dimSize);
    if (static_cast<float>(static_cast<float>(dimSize - bits) / dimSize) >
        threshold)
      modes.push_back(sparse);
    else {
      modes.push_back(dense);
    }
  }
  this->data = std::make_shared<taco::Tensor<float>>(
      taco::Tensor<float>(this->name, this->sizes, modes));
}

// constructor from sparsity vector (doesn't initialize tensor)
Tensor::Tensor(std::vector<int> sizes, std::vector<SparsityVector> sparsities,
               const std::string &n, const bool outputTensor)
    : name(n), sizes(sizes), sparsities(sparsities) {
  numDims = sizes.size();
  this->outputTensor = outputTensor;
}
// constructor for empty output tensors
Tensor::Tensor(std::vector<int> sizes, const std::string &n)
    : name(n), sizes(sizes) {
  numDims = sizes.size();
  for (int i = 0; i < numDims; ++i) {
    sparsities.push_back(SparsityVector());
    sparsities[i].set();
  }
}

Tensor::Tensor(std::vector<int> sizes, const std::string &n,
               taco::Format format)
    : data(std::make_shared<taco::Tensor<float>>(n, sizes, format)), name(n),
      sizes(sizes) {
  numDims = sizes.size();
  for (int i = 0; i < numDims; ++i) {
    sparsities.push_back(SparsityVector());
    sparsities[i].set();
  }
}

Tensor::Tensor(std::vector<int> sizes, std::vector<float> sparsityRatios,
               const std::string &n, taco::Format format)
    : data(std::make_shared<taco::Tensor<float>>(n, sizes, format)), name(n),
      sizes(sizes) {
  numDims = sizes.size();
  // Initialize sparsity bitsets to 1 (active)
  for (int i = 0; i < numDims; ++i) {
    sparsities.push_back(SparsityVector());
    sparsities[i].set();
  }

  // number of dimensions can vary: compute indices for each one
  for (int i = 0; i < numDims; ++i) {
    int zeroCount = static_cast<int>(sizes[i] * sparsityRatios[i]);

    std::vector<int> indices(sizes[i]);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{SEED});

    for (int j = 0; j < zeroCount; ++j)
      sparsities[i].set(indices[j], 0);
  }

  initialize_data();
}

void Tensor::create_data(taco::Format format) {
  this->data = std::make_shared<taco::Tensor<float>>(
      taco::Tensor<float>(this->name, this->sizes, format));
}

void Tensor::initialize_data() {
  // number of dimensions can vary so compute num elements
  assert(numDims > 0);
  taco::Format dense({taco::Dense, taco::Dense});
  int numElements = 1;
  for (auto size : sizes)
    numElements *= size;

  for (int numElement = 0; numElement < numElements; ++numElement) {
    auto indices = get_indices(sizes, numElement);
    bool isZero = false;

    for (int i = 0; i < numDims; ++i) {
      if (sparsities[i][indices[i]] == 0) {
        isZero = true;
        break;
      }
    }
    if (isZero) {
      if (this->data->getFormat() == dense) {
        data->insert(indices, 0.0f);
      }
      continue;
    }

    float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    data->insert(indices, val);
  }
  data->pack();
}

void Tensor::print_matrix() {
  assert(numDims == 2 && "Tensor must be a matrix to call this method");
  std::vector<std::vector<float>> tmp(sizes[0],
                                      std::vector<float>(sizes[1], 0.0));
  for (auto entry : *data) {
    tmp[entry.first[0]][entry.first[1]] = entry.second;
  }
  for (int i = 0; i < sizes[0]; ++i) {
    for (int j = 0; j < sizes[1]; ++j) {
      std::cout << tmp[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

void Tensor::print_full_sparsity() {
  for (int i = 0; i < numDims; ++i) {
    std::cout << "dim " << i << std::endl;
    for (int j = 0; j < sizes[i]; j++) {
      if (sparsities[i][j] == 0)
        std::cout << '0';
      else
        std::cout << '1';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

float Tensor::get_sparsity_ratio() {
  size_t total = 1;
  size_t nnz = 1;
  for (int dim = 0; dim < this->numDims; dim++) {
    int dimSize = this->sizes[dim];
    total *= dimSize;
    int bits = count_bits(this->sparsities[dim], dimSize);
    if (bits > 0)
      nnz *= bits;
  }
  int zero_elements = total - nnz;
  return static_cast<float>(zero_elements) / total;
}

size_t Tensor::get_nnz() {
  size_t nnz = 1;

  size_t numElements = 1;

  std::vector<size_t> dimNnz;
  for (int i = 0; i < sparsities.size(); ++i) {
    std::cout << count_bits(sparsities[i], sizes[i]) << " ";
    nnz *= count_bits(sparsities[i], sizes[i]);
  }

  std::cout << std::endl;
  
  return nnz;
}

void Tensor::print_shape() {
  std::cout << "(";
  for (auto size : this->sizes) {
    std::cout << size << ", ";
  }
  std::cout << ")" << std::endl;
}

size_t Tensor::compute_size_in_bytes() {
  size_t size{ 0 };
  auto index = data->getStorage().getIndex();
  auto format = data->getFormat().getModeFormats();

  int nnz{ 1 };
  std::vector<int> dimNnz;

  for (int i = 0; i < sparsities.size(); ++i)
    dimNnz.push_back(count_bits(sparsities[i], sizes[i]));

  bool hadSparse = false;
  for (int i = format.size() - 1; i >= 0; --i) {
    if (format[i] == taco::Sparse)
      hadSparse = true;
    if (hadSparse) {
      nnz *= dimNnz[i];
    } else {
      nnz *= sizes[i];
    }
  }

  int currDims{ 1 };
  int currSparseDims{ 1 };
  int prevDims{ -1 };

  for (int i = 0; i < format.size(); ++i) {
    if (format[i] == taco::Dense) {
      std::cout << i << " " << 1 << std::endl;
      size += 1;
      currDims *= sizes[i];
      currSparseDims *= dimNnz[i];
      prevDims = prevDims == -1 ? currDims : prevDims * sizes[i];
      continue;
    }
    size += prevDims != -1 ? prevDims + 1 : currSparseDims + 1;
    currDims *= dimNnz[i];
    currSparseDims *= dimNnz[i];
    size += currSparseDims;
    prevDims = currSparseDims;
  }
  
  return (size + nnz) * sizeof(float);
}
