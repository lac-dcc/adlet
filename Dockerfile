FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Base build dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    tk-dev \
    liblzma-dev \
    cmake \
    ninja-build \
    python3.12 \
    python3-pip \
    python3.12-venv \
    software-properties-common \
    gnupg && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install Clang 17 (pin version to avoid LLVM 18 crash)
# ---------------------------------------------------------------------------
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 17 && \
    apt-get install -y clang-17 clang-tools-17 libomp-17-dev && \
    ln -sf /usr/bin/clang-17 /usr/bin/clang && \
    ln -sf /usr/bin/clang++-17 /usr/bin/clang++

# ---------------------------------------------------------------------------
# Install Google Chrome (needed by Kaleido)
# ---------------------------------------------------------------------------
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb

# ---------------------------------------------------------------------------
# Set compiler environment
# ---------------------------------------------------------------------------
ENV CC=clang
ENV CXX=clang++

WORKDIR /app

# ---------------------------------------------------------------------------
# Clone TACO
# ---------------------------------------------------------------------------
RUN echo "Cloning TACO" && \
    git clone --depth 1 https://github.com/tensor-compiler/taco.git && \
    cd taco && \
    git checkout 0e79acb56cb5f3d1785179536256e206790b2a9e

# ---------------------------------------------------------------------------
# Clone SPA (adlet)
# ---------------------------------------------------------------------------
RUN echo "Cloning SPA" && \
    git clone https://github.com/lac-dcc/adlet.git && \
    cd adlet && \
    git checkout v1.0

# ---------------------------------------------------------------------------
# Build TACO
# ---------------------------------------------------------------------------
RUN echo "Building TACO" && \
    mkdir -p taco/build && \
    cd taco/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. && \
    make -j$(nproc)

# ---------------------------------------------------------------------------
# Build SPA
# ---------------------------------------------------------------------------
RUN echo "Building SPA" && \
    cp -r adlet/scripts /app/scripts/ && \
    cp -r adlet/einsum-dataset/ /app/einsum-dataset/ && \
    mkdir -p adlet/build && \
    cd adlet/build && \
    cmake -G Ninja ../ && \
    ninja

RUN mkdir -p /app/results/

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r /app/scripts/requirements.txt

ENV PATH="/venv/bin:$PATH"

# ---------------------------------------------------------------------------
# Environment variables for runtime
# ---------------------------------------------------------------------------
ENV EINSUM_DATASET="einsum-dataset/"
ENV BIN_PATH=/app/adlet/build/benchmark
ENV SPA_ROOT=/app/adlet/
ENV BUILD_PATH=/app/adlet/build
ENV TESA_BIN_PATH=/app/adlet/build/tesa-prop

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["python", "-u", "scripts/artifact.py"]
CMD []
