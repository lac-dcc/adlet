## Sparsity Propagation Analysis

```ascii
██████ ██████ ██████
█····· █····█ █····█
██████ ██████ ██████
·····█ █····· █····█
██████ █····· █····█
```

## About
This repository implements the Sparsity Propagation Analysis (SPA).

SPA is a static analysis able to propagate structured sparsity in n-dimensional tensors in a computational graph where nodes represent kernels such as general `einsum` expressions or `addition`.


## Dependencies
SPA is built on top of the [Tensor Algebra Compiler](https://github.com/tensor-compiler/taco/tree/0e79acb56cb5f3d1785179536256e206790b2a9e). The project expects the `taco` library to be installed in `../taco` relative to the root directory.


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

## Contributing

Contributions are welcome! 

You'll need [`pre-commit`](https://pre-commit.com/) for contributing to the project.

After installation, run:

```bash
$ pre-commit install
```

