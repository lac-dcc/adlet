#include "../include/einsum.hpp"
#include "../include/utils.hpp"

void run(const std::string &file_path, const double sparsity, bool run_fw,
              bool run_lat, bool run_bw) {
  auto benchmark = read_einsum_benchmark(file_path);

  if (benchmark.path.empty() || benchmark.strings.empty() ||
      benchmark.sizes.empty()) {
    std::cerr << "Could not parse einsum benchmark.\n";
    return;
  }
  const auto buildStart = begin();
  auto g =
      build_tree(benchmark.sizes, benchmark.strings, benchmark.path, sparsity);
  end(buildStart, "create graph = ");

  std::cout << "initial_ratio = " << g.get_sparsity_ratio() << std::endl;
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
  int param = 1;
  const std::string file_path = argv[++param];
  const double sparsity = std::stod(argv[++param]);
  const bool run_fw = std::stoi(argv[++param]);
  const bool run_lat = std::stoi(argv[++param]);
  const bool run_bw = std::stoi(argv[++param]);
  SEED = std::stoi(argv[++param]);
  run(file_path, sparsity, run_fw, run_lat, run_bw);

  return 0;
}
