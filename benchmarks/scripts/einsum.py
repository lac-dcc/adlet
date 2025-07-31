import opt_einsum as oe
import einsum_benchmark

def generate_lists(einsum_string, tensors):
    _, path_info = oe.contract_path(einsum_string, *tensors)
    indices = [pi[0] for pi in path_info.contraction_list]
    einsum_strings = [pi[2] for pi in path_info.contraction_list]
    shape_list = [t.shape for t in tensors]

    return (indices, einsum_strings, shape_list)

def write_benchmark(benchmark_name, indices, einsum_strings, tensor_sizes):
    with open(benchmark_name + ".txt", "w") as file:
        file.write(str(indices) + "\n")
        file.write(str(einsum_strings) + "\n")
        file.write(str(tensor_sizes) + "\n")

if __name__ == "__main__":
    instance = einsum_benchmark.instances["qc_circuit_n49_m14_s9_e6_pEFGH_simplified"]
    einsum_string = instance.format_string
    tensors = instance.tensors
    write_benchmark("qc_circuit_n49_m14_s9_e6_pEFGH_simplified", *generate_lists(einsum_string, tensors))
