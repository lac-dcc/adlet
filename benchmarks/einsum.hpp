#include "../src/dot.hpp"
#include "../src/einsum.hpp"
#include "../src/utils.hpp"
#include "taco/format.h"

void run(const std::string &file_path, const bool propagate,
         const double sparsity, const double chanceToPrune,
         const bool sparse = true) {

  auto benchmark = readEinsumBenchmark(file_path);

  if (benchmark.path.empty() || benchmark.strings.empty() ||
      benchmark.sizes.empty()) {
    std::cerr << "Could not parse einsum benchmark.\n";
    return;
  }
  const auto buildStart = begin();
  auto g = buildTree(benchmark.sizes, benchmark.strings, benchmark.path,
                     sparsity, chanceToPrune);
  end(buildStart, "create graph = ");

  g.run_propagation(FORWARD);
  std::cout << "ratio before = " << g.get_sparsity_ratio() << std::endl;
  if (propagate) {
    const auto startPropagation = begin();
    g.run_propagation();
    end(startPropagation, "analysis = ");
  } else {
    std::cout << "analysis = " << 0 << std::endl;
  }

  auto startLoad = begin();
  for (auto t : g.inputs) {
    if (!t->outputTensor) {
      t->create_data(generateModes(t->numDims, sparse));
      t->initialize_data();
    } else
      t->create_data(generateModes(t->numDims, false));
  }
  end(startLoad, "load graph = ");

  print_memory_usage();
  g.get_tensor_sizes();
  std::cout << "ratio after = " << g.get_sparsity_ratio() << std::endl;
  const auto startComp = begin();
  g.compile();
  end(startComp, "compilation = ");
  const auto startRun = begin();
  auto result = g.compute();
  end(startRun, "runtime = ");
  print_dot(g, "teste.dot");
}

int benchmark_einsum(int argc, char *argv[]) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0]
              << " einsum <file_path> <sparsity> <chance_to_prune> "
                 "<sparse> <propagate> <random_seed>\n ";
    return 1;
  }
  int param = 1;
  const std::string file_path = argv[++param];
  const std::string sparseStr = argv[++param];
  const bool sparse = sparseStr == "sparse";
  const double sparsity = std::stod(argv[++param]);
  const double chanceToPrune = std::stod(argv[++param]);
  const bool propagate = std::stoi(argv[++param]);
  SEED = std::stoi(argv[++param]);
  run(file_path, propagate, sparsity, chanceToPrune, sparse);

  return 0;
}
