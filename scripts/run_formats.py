import subprocess


sizes = [1024, 2048, 4096]
sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
formats = ["D", "S"]


def run_matrix():
    rank = 2
    with open("matrix.csv", "wt") as file:
        for size in sizes:
            for sparsity in sparsities:
                for format in formats:
                    cmd = ["./benchmark",
                        "format",
                        "SS" if format == "S" else "DD",
                        str(rank),
                        str(size),
                        str(size),
                        str(sparsity),
                        str(sparsity),
                    ]
                    print(cmd)
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        process.wait()
                        file.write(process.stdout.read())
                        file.flush()

                    except Exception as e:
                        print(f"Error {e} running cmd {cmd}")

def run_3D():
    rank = 3
    with open("matrix.csv", "wt") as file:
        for size in sizes:
            for sparsity in sparsities:
                for format in formats:
                    cmd = ["./benchmark",
                        "format",
                        "SSS" if format == "S" else "DDD",
                        str(rank),
                        str(size),
                        str(size),
                        str(size),
                        str(sparsity),
                        str(sparsity),
                        str(sparsity),
                    ]
                    print(cmd)
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        process.wait()
                        file.write(process.stdout.read())
                        file.flush()

                    except Exception as e:
                        print(f"Error {e} running cmd {cmd}")
if __name__ == "__main__":
    run_matrix()
    run_3D()



