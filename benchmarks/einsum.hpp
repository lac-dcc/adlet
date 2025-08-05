#include "../src/einsum.hpp"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/tensor.h"

void run(const bool propagate, const std::string &file_path) {

  auto benchmark = readEinsumBenchmark(file_path);

  if (benchmark.path.empty() || benchmark.strings.empty() ||
      benchmark.sizes.empty()) {
    std::cerr << "Could not parse einsum benchmark.\n";
    return;
  }
  const auto buildStart = begin();
  auto g = buildTree(benchmark.sizes, benchmark.strings, benchmark.path);
  end(buildStart, "create graph = ");

  /**/
  /*for (auto node : g.inputs) {*/
  /*  if (node->outputTensor) {*/
  /*    node->create_data(generateDenseModes(node->numDims));*/
  /*  } else {*/
  /*    node->create_data(generateDenseModes(node->numDims));*/
  /*    node->initialize_data();*/
  /*  }*/
  /*}*/

  g.run_propagation(FORWARD);
  std::cout << "ratio before = " << g.get_sparsity_ratio() << std::endl;
  if (propagate) {
    const auto startPropagation = begin();
    g.run_propagation();
    end(startPropagation, "analysis = ");
  } else {
    std::cout << "analysis = " << 0 << std::endl;
  }

  const auto startComp = begin();
  g.compile();
  end(startComp, "compilation = ");
  const auto startRun = begin();
  auto result = g.compute();
  end(startRun, "runtime = ");
}

int benchmark_einsum(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " einsum <file_path> <sparsity> <format> "
                 "<propagate> \n ";
    return 1;
  }
  int param = 1;
  const std::string file_path = argv[++param];
  double sparsity = std::stod(argv[++param]);
  std::string format = argv[++param];
  bool propagate = std::stoi(argv[++param]);
  /*run(propagate, file_path);*/

  taco::Format format2(
      taco::Format({taco::Sparse, taco::Sparse, taco::Sparse, taco::Sparse}));

  taco::Tensor<float> d("T", {4, 4, 4, 4}, format2);
  d.insert({0, 0, 0, 0}, 1.0f);
  d.insert({0, 0, 1, 0}, 1.0f);
  d.insert({2, 0, 3, 0}, 1.0f);
  d.pack();

  taco::Format format3(
      taco::Format({taco::Sparse, taco::Sparse, taco::Sparse, taco::Sparse}));
  taco::Tensor<float> e("R", {4, 4, 4, 4}, format3);
  e.insert({0, 0, 0, 0}, 1.0f);
  e.insert({0, 0, 1, 0}, 1.0f);
  e.insert({1, 0, 3, 0}, 1.0f);
  e.pack();
  std::vector<taco::TensorBase> tensors;
  tensors.push_back(e);
  tensors.push_back(d);

  taco::Format format4(
      taco::Format({taco::Dense, taco::Dense, taco::Sparse, taco::Sparse}));

  taco::parser::EinsumParser parser("bacd,afbe->cfed", tensors, format4,
                                    taco::Datatype::Float32);
  parser.parse();
  taco::Tensor<float> result = parser.getResultTensor();
  result.compile();
  result.assemble();
  result.compute();

  return 0;
}
