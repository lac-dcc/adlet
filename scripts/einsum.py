import opt_einsum as oe
import pickle
import numpy as np
import os

def generate_lists(einsum_string, tensors):
    path, path_info = oe.contract_path(einsum_string, *tensors)
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
    with open(
        "./instances/qc_circuit_n49_m14_s9_e6_pEFGH_simplified.pkl", "rb"
    ) as file:
        einsum_string, tensors, path_meta, sum_output = pickle.load(file)

    write_benchmark("./", "qc_circuit_n49_m14_s9_e6_pEFGH_simplified", *generate_lists(einsum_string, tensors))
    

