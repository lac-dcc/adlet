import subprocess
import csv
import os
import statistics

# Fixed matrix size
rows = 1024
cols = 1024

# Format combinations
left_formats = ["CSR", "CSC", "DCSR", "DCSC", "SparseDense"]
right_formats = ["CSR", "CSC", "DCSR", "DCSC", "SparseDense"]
out_format = "DD"

sparsities = [
    ((0.5, 0.5), (0.5, 0.5)),
    ((0.8, 0.8), (0.5, 0.5)),
    ((0.5, 0.5), (0.8, 0.8)),
    
    ((0.5, 0.8), (0.5, 0.5)),
    ((0.8, 0.5), (0.5, 0.5)),

    ((0.5, 0.5), (0.5, 0.8)),
    ((0.5, 0.5), (0.8, 0.5)),
] 
# Number of runs per configuration
num_runs = 1

# Output CSV file
output_file = "sparse_sparse.csv"

# CSV Header
header = [
    "rows", "cols", "out_format", "left_format", "right_format",
    "left_row_sparsity", "left_col_sparsity",
    "right_row_sparsity", "right_col_sparsity", "mean_exec_time"
]

# Write header only if file does not exist
write_header = not os.path.exists(output_file)

with open(output_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(header)

    for left_format in left_formats:
        for right_format in right_formats:
            for sparsity in sparsities:
                left_sparsity, right_sparsity = sparsity
                left_row_sparsity, left_col_sparsity = left_sparsity
                right_row_sparsity, right_col_sparsity = right_sparsity
                exec_times = []

                for run in range(num_runs):
                    cmd = [
                        "./benchmark",
                        str(rows),
                        str(cols),
                        out_format,
                        left_format,
                        right_format,
                        f"{left_row_sparsity:.2f}",
                        f"{left_col_sparsity:.2f}",
                        f"{right_row_sparsity:.2f}",
                        f"{right_col_sparsity:.2f}",
                    ]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        lines = result.stdout.strip().splitlines()

                        if len(lines) < 2:
                            raise ValueError("Unexpected benchmark output format.")

                        # Parse the second line for the execution time
                        data_line = lines[1].strip().split(",")
                        exec_time = float(data_line[-1])
                        exec_times.append(exec_time)

                    except Exception as e:
                        print(f"Error on run {run+1}/10 for {left_format} with sparsity {left_row_sparsity},{left_col_sparsity}: {e}")
                        exec_times.append(-1.0)

                # Compute average ignoring any failed (-1.0) runs
                valid_times = [t for t in exec_times if t >= 0.0]

                if valid_times:
                    mean_time = statistics.mean(valid_times)
                else:
                    mean_time = -1.0  # mark as invalid

                row = [
                    rows, cols, out_format, left_format, right_format,
                    left_row_sparsity, left_col_sparsity,
                    right_row_sparsity, right_col_sparsity,
                    mean_time
                ]

                writer.writerow(row)
                print("✔ Recorded:", row)
