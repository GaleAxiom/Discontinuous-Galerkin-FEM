# Dockerfile for DGFEM project with pre-installed dependencies
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    gmsh \
    libgmsh-dev \
    libeigen3-dev \
    libgtest-dev \
    libgmock-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install Google Test and Google Mock
WORKDIR /tmp/gtest-build
RUN cmake /usr/src/googletest && \
    make && \
    cp lib/*.a /usr/lib/ 2>/dev/null || cp ./*.a /usr/lib/ 2>/dev/null || \
    find . -name "*.a" -exec cp {} /usr/lib/ \;

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
