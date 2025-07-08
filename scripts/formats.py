import subprocess
import itertools
import csv
import os

# Fixed matrix size
rows = 1024
cols = 1024

# Format combinations
left_formats = ["CSR", "CSC", "DCSR", "DCSC", "SparseDense"]
out_format = "DD"
right_format = "DD"

# Sparsity configurations for the left matrix
left_sparsities = [
    (0.50, 0.0), (0.70, 0.0), (0.90, 0.0),
    (0.0, 0.50), (0.0, 0.70), (0.0, 0.90),
    (0.50, 0.50), (0.50, 0.70), (0.50, 0.90),
    (0.70, 0.50), (0.90, 0.50),
]

# Fixed right matrix sparsity
right_row_sparsity = 0.0
right_col_sparsity = 0.50

# Output CSV file
output_file = "benchmark_results.csv"

# CSV Header expected from the benchmark output
expected_header = [
    "rows", "cols", "out_format", "left_format", "right_format",
    "left_row_sparsity", "left_col_sparsity",
    "right_row_sparsity", "right_col_sparsity", "exec_time"
]

# Write header only once
write_header = not os.path.exists(output_file)

with open(output_file, "wt", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(expected_header)

    for left_format in left_formats:
        for left_row_sparsity, left_col_sparsity in left_sparsities:
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

                header_line = lines[0].strip()
                data_line = lines[1].strip()

                # Optional: validate header matches
                if header_line.replace(" ", "") != ",".join(expected_header).replace(" ", ""):
                    raise ValueError("Header mismatch in benchmark output.")

                # Parse result row
                data = data_line.split(",")
                if len(data) != len(expected_header):
                    raise ValueError("Data row length mismatch.")

                writer.writerow(data)
                print("Recorded:", data)

            except subprocess.CalledProcessError as e:
                print(f"Error running {' '.join(cmd)}")
                print("stderr:", e.stderr.strip())
            except Exception as ex:
                print(f"Error parsing output of {' '.join(cmd)}: {ex}")

