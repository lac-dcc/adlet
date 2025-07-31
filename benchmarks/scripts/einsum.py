import os
import opt_einsum as oe
import einsum_benchmark

def generate_lists(einsum_string, tensors):
    _, path_info = oe.contract_path(einsum_string, *tensors)
    indices = [pi[0] for pi in path_info.contraction_list]
    einsum_strings = [pi[2] for pi in path_info.contraction_list]
    
    return (indices, einsum_strings)

def write_benchmark(root, benchmark_name, indices, einsum_strings):
    os.mkdir(root + benchmark_name)
    with open(root + benchmark_name + "/indices.txt", "w") as file:
        file.write(str(indices))
    with open(root + benchmark_name + "/einsum_strings.txt", "w") as file:
        file.write(str(einsum_strings))

if __name__ == "__main__":
    instance = einsum_benchmark.instances["qc_circuit_n49_m14_s9_e6_pEFGH_simplified"]
    einsum_string = instance.format_string
    tensors = instance.tensors
    write_benchmark("./", "qc_circuit_n49_m14_s9_e6_pEFGH_simplified", *generate_lists(einsum_string, tensors))

