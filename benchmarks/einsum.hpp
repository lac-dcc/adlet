#include "../src/dot.hpp"
#include "../src/einsum.hpp"
#include "../src/utils.hpp"
#include "taco/format.h"

void run(const std::string &file_path, const bool propagate,
         const double sparsity, const double chanceToPrune) {

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
      /*t->create_data(taco::Format({taco::Sparse, taco::Sparse}));*/
      t->create_data(generateModes(t->numDims, true));
      t->initialize_data();
    } else
      t->create_data(0.5);
    /*t->create_data(taco::Format({taco::Sparse, taco::Sparse}));*/
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
  /*std::cout << *(result->data) << std::endl;*/
  end(startRun, "runtime = ");
  /*print_dot(g, "teste.dot");*/
}

int benchmark_einsum(int argc, char *argv[]) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " einsum <file_path> <sparsity> <chance_to_prune> "
                 "<propagate> <random_seed>\n ";
    return 1;
  }
  int param = 1;
  const std::string file_path = argv[++param];
  double sparsity = std::stod(argv[++param]);
  double chanceToPrune = std::stod(argv[++param]);
  bool propagate = std::stoi(argv[++param]);
  SEED = std::stoi(argv[++param]);
  run(file_path, propagate, sparsity, chanceToPrune);

  return 0;
}
