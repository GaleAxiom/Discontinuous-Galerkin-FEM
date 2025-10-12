#!/bin/bash
# Script to build and test the DGFEM project using Docker

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Docker image...${NC}"
docker build -t dgfem-dev .

echo -e "${GREEN}Docker image built successfully!${NC}"

echo -e "${BLUE}Running build inside Docker container...${NC}"
docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    dgfem-dev \
    bash -c "
        echo 'Configuring CMake...'
        cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release
        echo 'Building project...'
        cmake --build build --config Release
        echo 'Running tests...'
        cd build && ctest -C Release
    "

echo -e "${GREEN}Build and test completed successfully!${NC}"
