#include "../src/dot.hpp"
#include "../src/einsum.hpp"

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

  print_memory_usage();
  g.run_propagation(FORWARD);
  std::cout << "ratio before = " << g.get_sparsity_ratio() << std::endl;
  if (propagate) {
    const auto startPropagation = begin();
    g.run_propagation();
    end(startPropagation, "analysis = ");
  } else {
    std::cout << "analysis = " << 0 << std::endl;
  }

  std::cout << "ratio after = " << g.get_sparsity_ratio() << std::endl;
  const auto startComp = begin();
  g.compile();
  end(startComp, "compilation = ");
  const auto startRun = begin();
  /*auto result = g.compute();*/
  end(startRun, "runtime = ");
  print_dot(g, "teste.dot");
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
  run(propagate, file_path);

  return 0;
}
