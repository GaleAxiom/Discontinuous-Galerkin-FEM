# Docker Setup for DGFEM Project

This project uses Docker to provide a consistent build environment with all dependencies pre-installed.

## Prerequisites

- Docker installed on your system
- For GitHub Actions: The Docker image is automatically built and published to GitHub Container Registry

## Building the Docker Image Locally

```bash
docker build -t dgfem-dev .
```

## Using the Docker Container

### Quick Build and Test

Use the provided script:

```bash
chmod +x build-docker.sh
./build-docker.sh
```

### Manual Docker Commands

**Run an interactive shell:**
```bash
docker run -it --rm -v "$(pwd):/workspace" -w /workspace dgfem-dev /bin/bash
```

**Build the project:**
```bash
docker run --rm -v "$(pwd):/workspace" -w /workspace dgfem-dev bash -c "
    cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
"
```

**Run tests:**
```bash
docker run --rm -v "$(pwd):/workspace" -w /workspace dgfem-dev bash -c "
    cd build && ctest -C Release
"
```

## GitHub Actions

The project has two workflows:

1. **`docker-publish.yml`**: Automatically builds and publishes the Docker image to GitHub Container Registry when:
   - The Dockerfile changes
   - Manually triggered via workflow_dispatch
   
2. **`cmake-single-platform.yml`**: Runs the build and tests using the pre-built Docker image

The Docker image includes:
- Ubuntu 22.04 base
- CMake and build tools
- GMSH and libgmsh-dev
- Eigen3
- Google Test (GTest) and Google Mock (GMock)

## Benefits

- **Consistency**: Same environment locally and in CI/CD
- **Speed**: Dependencies are pre-installed in the image
- **Reproducibility**: Anyone can build the project with the exact same dependencies
- **Isolation**: No need to install dependencies on your host system
