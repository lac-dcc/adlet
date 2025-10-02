## Sparsity Propagation Analysis

```ascii
██████ ██████ ██████
█····· █····█ █····█
██████ ██████ ██████
·····█ █····· █····█
██████ █····· █····█
```

## About
This repository implements the [Sparsity Propagation Analysis (SPA)](Link to the paper).

SPA is a static analysis capable of propagating structured sparsity in n-dimensional tensors within a computational graph, where nodes represent kernels such as general `einsum` expressions or `addition` operations.


For example, given a matrix multiplication with structured sparsity:
```text
0000   1011   0000
1110 * 1011 = ?0??
0000   1011   0000
1110   0000   ?0??
```
SPA can infer the sparsity of the resulting matrix without actually performing the multiplication.

Furthermore, the analysis is not limited to matrices or matrix multiplication. It can be applied to n-dimensional tensors, general `einsum` expressions, and `addition` operations.

## Dependencies
SPA is built on top of the [Tensor Algebra Compiler](https://github.com/tensor-compiler/taco/tree/0e79acb56cb5f3d1785179536256e206790b2a9e). The project expects the `taco` library to be installed in `../taco` relative to the root directory.

The project was implemented using [Clang 14](https://releases.llvm.org/14.0.0/tools/clang/docs/ReleaseNotes.html)
Build Dependencies:

- [CMake](https://cmake.org/download/)
- [Ninja](https://github.com/ninja-build/ninja)

## How to build

1. Clone and build `taco`
2. Clone SPA
```bash
$ git clone https://github.com/lac-dcc/adlet/tree/main
```

3. Build
```bash
$ mkdir build && cd build
$ cmake -G Ninja ../ && ninja
```

## Running tests

Once built, you can run the test files:
```bash
$ ./tests
```

## Artifact

Build the image:

```bash
$ docker build -t spa-artifact
```

Run the experiments:
```bash
$ docker run --rm -v $(pwd)/images:/workspace/images spa-artifact
```


## Contributing

Contributions are welcome! 

You'll need [`pre-commit`](https://pre-commit.com/) for contributing to the project.

After installation, run:

```bash
$ pre-commit install
```

