from typing import List
import os
import einsum_benchmark
import opt_einsum as oe

def convert_einsum_string(einsum_string):
    char_mapping = {}
    new_string = []
    valid_chars = [chr(c) for c in range(ord('a'), ord('z') + 1)] \
            + [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [',', '-', '>']
    new_char = ord('a')
    for c in einsum_string:
        assert c != ' '
        if c == ',' or c == '-' or c == '>':
            new_string.append(c)
            continue
        if c in char_mapping:
            new_string.append(char_mapping[c])
            continue
        char_mapping[c] = chr(new_char)
        new_string.append(char_mapping[c])
        new_char += 1
        if new_char == ord('z') + 1:
            new_char = ord('A')
        elif new_char == ord('Z') + 1:
            assert all([ c in valid_chars for c in new_string]), "This expression has too many index variables!"
    return ''.join(new_string)

def generate_lists(einsum_string, tensors):
    _, path_info = oe.contract_path(einsum_string, *tensors, optimize='auto')
    indices = [pi[0] for pi in path_info.contraction_list]
    einsum_strings = [convert_einsum_string(pi[2]) for pi in path_info.contraction_list]
    shape_list = [t.shape for t in tensors]

    return (indices, einsum_strings, shape_list)

def write_benchmark(benchmark_name, indices, einsum_strings, tensor_sizes):
    with open(benchmark_name + ".txt", "w") as file:
        file.write(str(indices) + "\n")
        file.write(str(einsum_strings) + "\n")
        file.write(str(tensor_sizes) + "\n")

def get_small_benchmarks(threshold: int = 100):
    benchmarks = []
    sizes = {}
    for instance in einsum_benchmark.instances:
        if len(instance.tensors) <= threshold:
            benchmarks.append(instance)
            if len(instance.tensors) in sizes:
                sizes[len(instance.tensors)] +=1
            else:
                sizes[len(instance.tensors)] = 1

            t = 0
            for tensor in instance.tensors:
                for d in tensor.shape:
                    if d > t:
                        t = d
            print(f"max size for {instance.name} is {t}")

    print(sizes)


    print(f"saving {len(benchmarks)} benchmarks")
    os.makedirs(f"./sub{threshold}", exist_ok=True)
    for instance in benchmarks:
        write_benchmark(f"sub{threshold}/{instance.name}", *generate_lists(instance.format_string, instance.tensors))

def get_biggest_dim(names: List[str]) -> int:
    biggest_instance = 0
    biggest = 0
    for name in names:
        instance = einsum_benchmark.instances[name.split(".txt")[0]]
        _, path_info = oe.contract_path(instance.format_string, *instance.tensors, optimize='auto')
        for pi in path_info.contraction_list:
            lhs, rhs = pi[2].split("->")
            dim_list = lhs.split(",")
            dim_list.append(rhs)
            for s in dim_list:
                if len(s) > biggest:
                    biggest = len(s)
                    biggest_instance = instance
    print(biggest_instance.name, biggest)
    return biggest

if __name__ == "__main__":
    get_small_benchmarks(1000)

