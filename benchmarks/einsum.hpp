#include "../src/dot.hpp"
#include "../src/einsum.hpp"
#include "../src/utils.hpp"
#include "taco/format.h"

void run(const std::string &file_path, const bool propagate,
         const double sparsity, const bool sparse = true) {

  auto benchmark = readEinsumBenchmark(file_path);

  if (benchmark.path.empty() || benchmark.strings.empty() ||
      benchmark.sizes.empty()) {
    std::cerr << "Could not parse einsum benchmark.\n";
    return;
  }
  const auto buildStart = begin();
  auto g =
      buildTree(benchmark.sizes, benchmark.strings, benchmark.path, sparsity);
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
    } else {
      /*t->create_data(generateModes(t->numDims, t->sizes, t->sparsities, true));*/
      /*t->initialize_data();*/
      t->create_data(generateModes(t->numDims, false));
    }
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

void run_prop(const std::string &file_path, const double sparsity,
        bool run_fw, bool run_lat, bool run_bw) {
  auto benchmark = readEinsumBenchmark(file_path);

  if (benchmark.path.empty() || benchmark.strings.empty() ||
      benchmark.sizes.empty()) {
    std::cerr << "Could not parse einsum benchmark.\n";
    return;
  }
  const auto buildStart = begin();
  auto g =
      buildTree(benchmark.sizes, benchmark.strings, benchmark.path, sparsity);
  end(buildStart, "create graph = ");

  if (run_fw) {
    g.run_propagation(FORWARD);
    std::cout << "fw_ratio = " << g.get_sparsity_ratio() << std::endl;
  } 
  if (run_lat) {
    g.run_propagation(INTRA);
    std::cout << "lat_ratio = " << g.get_sparsity_ratio() << std::endl;
  } 
  if (run_bw) {
    g.run_propagation(BACKWARD);
    std::cout << "bw_ratio = " << g.get_sparsity_ratio() << std::endl;
  }
}

int benchmark_einsum(int argc, char *argv[]) {
  if (argc != 7 && argc != 9) {
    std::cerr << "Usage for runtime/memory: " << argv[0]
              << " einsum <file_path> <sparsity> "
                 "<format> <propagate> <random_seed>\n ";
    std::cerr << "Usage for analysis: " << argv[0]
              << " einsum prop <file_path> <sparsity> "
                 "<run_fw> <run_lat> <run_bw> <random_seed>\n ";
    return 1;
  }
  
  int param = 1;
  if (argc == 7) {
    const std::string file_path = argv[++param];
    const std::string sparseStr = argv[++param];
    const bool format = sparseStr == "sparse";
    const double sparsity = std::stod(argv[++param]);
    const bool propagate = std::stoi(argv[++param]);
    SEED = std::stoi(argv[++param]);
    run(file_path, propagate, sparsity, format);
  } else {
    const std::string propStr = argv[++param];
    if (propStr != "prop") {
      std::cerr << "Usage for analysis: " << argv[0]
                << " einsum prop <file_path> <sparsity> "
                   "<run_fw> <run_lat> <run_bw> <random_seed>\n ";
      return 1;
    }
    const std::string file_path = argv[++param];
    const double sparsity = std::stod(argv[++param]);
    const bool run_fw = std::stoi(argv[++param]);
    const bool run_lat = std::stoi(argv[++param]);
    const bool run_bw = std::stoi(argv[++param]);
    SEED = std::stoi(argv[++param]);
    run_prop(file_path, sparsity, run_fw, run_lat, run_bw);
  }

  return 0;
}
