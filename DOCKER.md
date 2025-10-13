# Docker Setup for DG-FEM

This project uses Docker to provide a consistent build environment with all dependencies pre-installed.

## Docker Image

The project uses a custom Docker image published to GitHub Container Registry (GHCR):

```
ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest
```

### Included Dependencies

The Docker image includes:

- **Ubuntu 22.04** base system
- **CMake** (latest from Ubuntu repositories)
- **GCC/G++** build tools
- **GMSH** mesh generation library
- **Eigen3** linear algebra library
- **Google Test** and **Google Mock** testing frameworks

## Usage

### Local Development

Run the development environment:

```bash
docker run -it --rm -v $(pwd):/workspace ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest
```

Build the project inside the container:

```bash
cd /workspace
cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest
```

### Using Docker Compose (Recommended)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  dgfem:
    image: ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: /bin/bash
```

Then run:

```bash
docker-compose run --rm dgfem
```

### Building the Image Locally

If you need to modify the Docker image:

```bash
docker build -t dgfem-local .
docker run -it --rm -v $(pwd):/workspace dgfem-local
```

Or use the build script:

```bash
./build-docker.sh
```

## GitHub Actions Integration

The CI/CD pipeline uses the Docker image to ensure consistent builds:

### Automatic Image Building

The Docker image is automatically built and published when:

1. The `Dockerfile` is modified and pushed to main
2. The `.github/workflows/docker-publish.yml` is modified
3. Manually triggered via GitHub Actions UI (workflow_dispatch)

### CI Testing

The `cmake-single-platform.yml` workflow uses the Docker image:

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest
```

This eliminates the need to install dependencies in each CI run, speeding up the build process.

## Updating the Docker Image

1. Modify the `Dockerfile` as needed
2. Commit and push to main:
   ```bash
   git add Dockerfile
   git commit -m "Update Docker image dependencies"
   git push origin main
   ```
3. The GitHub Actions workflow will automatically build and publish the new image
4. The image will be tagged with:
   - `latest` (always points to the most recent build)
   - `main-<commit-sha>` (specific version)

## Troubleshooting

### Image Pull Issues

If you get permission errors pulling the image:

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

The image is public, so authentication should not be required.

### Cache Issues

Clear Docker cache:

```bash
docker system prune -a
docker pull ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest
```

### Building Without Docker

If you prefer not to use Docker, install dependencies manually:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake gmsh libgmsh-dev \
                        libeigen3-dev libgtest-dev libgmock-dev
```

**macOS (Homebrew):**
```bash
brew install cmake gmsh eigen googletest
```

## Performance Considerations

- The Docker image is ~500MB compressed, ~1.5GB uncompressed
- First pull will take time, but subsequent runs are fast
- CI builds are significantly faster using the pre-built image
- Local development can use volume mounts for live code updates

## Security

- The Docker image is built from official Ubuntu base
- All dependencies are from official Ubuntu 22.04 repositories
- The image is published to GHCR with automatic security scanning
- No secrets or sensitive data are included in the image

## Best Practices

1. **Pull latest before building:**
   ```bash
   docker pull ghcr.io/galeaxiom/discontinuous-galerkin-fem:latest
   ```

2. **Use volume mounts for development:**
   - Don't copy files into the container
   - Mount the project directory instead

3. **Clean up after testing:**
   ```bash
   docker-compose down
   docker system prune
   ```

4. **Tag specific versions for production:**
   - Use `main-<sha>` tags for reproducible builds
   - Only use `latest` for development

## Related Files

- `Dockerfile` - Image definition
- `.github/workflows/docker-publish.yml` - Image build workflow
- `.github/workflows/cmake-single-platform.yml` - CI testing workflow
- `build-docker.sh` - Local build script
