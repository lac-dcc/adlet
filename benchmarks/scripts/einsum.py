import os
import pickle
import opt_einsum as oe
import glob

def generate_lists(einsum_string, tensors):
    _, path_info = oe.contract_path(einsum_string, *tensors, optimize='auto')
    indices = [pi[0] for pi in path_info.contraction_list]
    einsum_strings = [convert_einsum_string(pi[2]) for pi in path_info.contraction_list]
    sizes = [tuple(t.shape) for t in tensors]
    
    return (indices, einsum_strings, sizes)

def write_benchmark(root, benchmark_name, einsum_strings, indices, sizes):
    os.mkdir(root + benchmark_name)
    with open(root + benchmark_name + "/einsum_strings.txt", "w") as file:
        file.write(str(einsum_strings))
    with open(root + benchmark_name + "/indices.txt", "w") as file:
        file.write(str(indices))
    with open(root + benchmark_name + "/sizes.txt", "w") as file:
        file.write(str(sizes))

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
            
if __name__ == "__main__":
    instance_files = glob.glob("./instances/*.pkl")
    for instance_file in instance_files:
        if "mc" in instance_file:
            continue
        instance_name = instance_file.split(".pkl")[0].split("/")[-1]
        if os.path.isdir('converted/' + instance_name):
            continue
        with open(instance_file, "rb") as file:
            einsum_string, tensors, _, _ = pickle.load(file)

            try:
                indices, einsum_strings, sizes = generate_lists(einsum_string, tensors)
            except AssertionError as e:
                print("couldn't write", instance_file)

            write_benchmark("./converted/", instance_name, einsum_strings, indices, sizes)

