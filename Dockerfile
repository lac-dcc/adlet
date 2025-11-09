FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    clang \
    libomp-dev \
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
    clang-tools

RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y ./google-chrome-stable_current_amd64.deb

ENV CC=clang
ENV CXX=clang++

WORKDIR /app

# Clone TACO
RUN echo "Cloning TACO" && \
    git clone --depth 1 https://github.com/tensor-compiler/taco.git && \
    cd taco && \
    git checkout 0e79acb56cb5f3d1785179536256e206790b2a9e

# Clone SPA (adlet)
ARG GITHUB_TOKEN
#TODO: remove token parameter
RUN echo "Cloning SPA" && \
    git clone -b artifact --depth 1 https://$GITHUB_TOKEN@github.com/lac-dcc/adlet.git && \
    cd adlet && \
    git checkout a81b4a0e077b1e79d2e433d70980116dc220a6ba

RUN echo "Cloning C++ TeSA Prop" && \
    git clone -b artifact --depth 1 https://github.com/seliayeu/tesa-prop.git && \
    cd tesa-prop && \
    git checkout 60f3370e3f5e8282c01b2de10fac6337c3a8c63c && \
    mkdir build

# Build TACO
RUN echo "Building TACO" && \
    mkdir -p taco/build && \
    cd taco/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. && \
    make -j$(nproc)

# Build SPA
RUN echo "Building SPA" && \
    cp -r adlet/scripts /app/scripts/ && \
    cp -r adlet/einsum-dataset/ /app/einsum-dataset/ && \
    mkdir -p adlet/build && \
    cd adlet/build && \
    cmake -G Ninja ../ && \
    ninja

RUN mkdir -p /app/results/

# Create virtual environment and install Python dependencies
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r /app/scripts/requirements.txt


# Activate venv in PATH
ENV PATH="/venv/bin:$PATH"

# Environment variables for your script
ENV EINSUM_DATASET="einsum-dataset/"
ENV BIN_PATH=/app/adlet/build/benchmark
ENV SPA_ROOT=/app/adlet/
ENV BUILD_PATH=/app/adlet/build
ENV TESA_ROOT=/app/tesa-prop/
ENV TESA_BIN_PATH=/app/tesa-prop/build/tesa-prop

# Entry point
ENTRYPOINT ["python", "-u", "scripts/artifact.py"]
CMD []
