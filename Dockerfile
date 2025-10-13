# Multi-stage Dockerfile for DGFEM project with pre-installed dependencies

# Base stage with common dependencies
FROM ubuntu:22.04 AS base

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgmsh4 \
    && rm -rf /var/lib/apt/lists/*

# Development stage with all build tools
FROM base AS development

# Install build tools and development dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    gmsh \
    libgmsh-dev \
    libeigen3-dev \
    libgtest-dev \
    libgmock-dev \
    ccache \
    ninja-build \
    clang-format \
    cppcheck \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Build and install Google Test and Google Mock
WORKDIR /tmp/gtest-build
RUN cmake /usr/src/googletest && \
    make && \
    cp lib/*.a /usr/lib/ 2>/dev/null || cp ./*.a /usr/lib/ 2>/dev/null || \
    find . -name "*.a" -exec cp {} /usr/lib/ \;

# Configure ccache
ENV PATH="/usr/lib/ccache:${PATH}"
ENV CCACHE_DIR=/workspace/.ccache

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]

# Builder stage for building the project
FROM development AS builder

# Copy source code
COPY dgfem /workspace/dgfem
COPY external /workspace/external

# Build the project
RUN cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON && \
    cmake --build build --config Release -j$(nproc)

# Production stage with minimal size
FROM base AS production

# Copy only the built binaries and required libraries
COPY --from=builder /workspace/build/libdgfem.a /usr/local/lib/
COPY --from=builder /workspace/build/examples/* /usr/local/bin/
COPY --from=builder /workspace/dgfem/include /usr/local/include/dgfem

# Set working directory
WORKDIR /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD test -f /usr/local/lib/libdgfem.a || exit 1

# Default command
CMD ["/bin/bash"]
