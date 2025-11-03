FROM ubuntu:24.04

RUN echo "SPA - Artifact"
RUN echo "Installing dependencies"

RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*


ENV CC=clang
ENV CXX=clang++

WORKDIR /app

RUN echo "Cloning TACO"
RUN git clone --depth 1 https://github.com/tensor-compiler/taco.git \
    && cd taco/ \
    && git checkout 0e79acb56cb5f3d1785179536256e206790b2a9e


RUN echo "Cloning SPA"
#TODO: remove when the repo become public
ARG GITHUB_TOKEN
RUN git clone --depth 1 https://$GITHUB_TOKEN@github.com/lac-dcc/adlet.git \
    && cd adlet \
    #TODO: use tags instead
    && git checkout 29a959e11e9d3a19d48cd0c4dcc23529945926f4

RUN echo "Building TACO"
RUN mkdir -p taco/build && cd taco/build && cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. && make -j$(nproc) && cd ../../

RUN echo "Building SPA"
RUN mkdir -p adlet/build && cd adlet/build && cmake -G Ninja ../ && ninja 


COPY scripts/ /app/scripts/

RUN python3 -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r scripts/requirements.txt

ENV PATH="/venv/bin:$PATH"

# generate einsum files
RUN echo "Generating einsum benchmarks"
RUN python3 scripts/einsum.py 


CMD ["python", "scripts/artifact.py"]
