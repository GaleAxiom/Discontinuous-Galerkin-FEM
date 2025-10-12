# Discontinuous Galerkin Finite Element Method (DG-FEM)

[![CMake Build](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/cmake-single-platform.yml)

A high-performance C++20 implementation of the Discontinuous Galerkin Finite Element Method for solving partial differential equations (PDEs).

## 🎯 Features

### Numerical Methods
- **Discontinuous Galerkin (DG) formulation** for hyperbolic and elliptic PDEs
- **High-order polynomial basis functions**: Legendre, Dubiner, and monomial bases
- **Flexible quadrature rules**: Gauss-Legendre integration on triangles and quadrilaterals
- **Time-stepping schemes**: Explicit Runge-Kutta methods for time-dependent problems
- **Geometric mappings**: Support for curved and affine element transformations

### Supported Equations
- **Advection equation** (scalar transport)
- **Laplace/Poisson equation** (elliptic)
- **Euler equations** (compressible inviscid flow)
- **Navier-Stokes equations** (compressible viscous flow)
- **Acoustic wave propagation**

### Mesh & Visualization
- **GMSH integration** for complex mesh generation
- **VTK output** for visualization in ParaView
- Support for **triangular and quadrilateral** elements
- Arbitrary order p-refinement

## 🚀 Quick Start

### Prerequisites

- **C++20** compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake** 3.12 or higher
- **Eigen3** (bundled in `external/eigen` or system-installed)
- **GMSH** library and headers
- **Google Test** (for testing)

### Building the Project

```bash
# Clone the repository
git clone https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM.git
cd Discontinuous-Galerkin-FEM

# Configure and build
cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
cd build
ctest --output-on-failure
```

### Using Docker

For a consistent build environment with all dependencies pre-installed:

```bash
# Build and test using Docker
./build-docker.sh

# Or manually
docker build -t dgfem-dev .
docker run --rm -v "$(pwd):/workspace" -w /workspace dgfem-dev bash -c "
    cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    cd build && ctest
"
```

See [DOCKER.md](DOCKER.md) for detailed Docker usage.

## 📚 Examples

The `dgfem/examples/` directory contains several demonstration programs:

### Basic Examples
- **`simple_mesh_test.cpp`** - Basic mesh creation and DG space setup
- **`laplace_solver_test.cpp`** - Solving Poisson's equation
- **`advection_solver_test.cpp`** - Scalar advection with DG

### Advanced Examples
- **`euler_vortex_test.cpp`** - Isentropic vortex (Euler equations)
- **`taylor_green_test.cpp`** - Taylor-Green vortex flow
- **`shear_flow_example.cpp`** - Shear layer instability
- **`acoustic_wave_test.cpp`** - Acoustic wave propagation
- **`convergence_test.cpp`** - Convergence rate verification
- **`high_order_test.cpp`** - High-order (p > 3) accuracy demonstration

### Running Examples

```bash
cd build/examples

# Run Laplace solver
./laplace_solver_test

# Run Euler vortex test
./euler_vortex_test

# Visualize results in ParaView
paraview ../output/*.vtk
```

## 🧪 Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
cd build
ctest

# Run specific test suite
ctest -R "Basis"           # Basis function tests
ctest -R "Solver"          # Solver tests
ctest -R "Convergence"     # Convergence tests

# Verbose output on failure
ctest --output-on-failure
```

Test categories:
- **Basis functions**: Legendre, Dubiner, monomial polynomials
- **Reference elements**: Shape functions, mappings, Jacobians
- **Quadrature**: Integration accuracy
- **Solvers**: DG assembler, weak forms, time-stepping
- **Convergence**: h-refinement and p-refinement rates
- **Boundary conditions**: Dirichlet, Neumann, periodic
- **Utilities**: Mesh creation, VTK output

## 📖 Project Structure

```
Discontinuous-Galerkin-FEM/
├── dgfem/                      # Main library
│   ├── include/dgfem/          # Header files
│   │   ├── basis/              # Polynomial basis functions
│   │   ├── boundary/           # Boundary conditions
│   │   ├── core/               # Mesh, solution, space
│   │   ├── quadrature/         # Numerical integration
│   │   ├── reference/          # Reference elements & mappings
│   │   ├── solver/             # DG solver & assembler
│   │   ├── utils/              # Mesh creation, VTK writer
│   │   └── weak_forms/         # PDE weak formulations
│   ├── src/                    # Implementation files
│   ├── tests/                  # Unit tests
│   └── examples/               # Example programs
├── external/eigen/             # Eigen linear algebra library
├── output/                     # VTK output files
├── Dockerfile                  # Docker container setup
├── build-docker.sh             # Docker build script
└── README.md                   # This file
```

## 🛠️ CMake Options

```bash
# Build with examples (default: ON)
cmake -B build -S dgfem -DBUILD_EXAMPLES=ON

# Build with tests (default: ON)
cmake -B build -S dgfem -DBUILD_TESTS=ON

# Debug build with sanitizers
cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Debug

# Use system Eigen3 instead of bundled
cmake -B build -S dgfem -DEigen3_DIR=/path/to/eigen3
```

## 🔬 Theory & Method

The Discontinuous Galerkin method combines features of finite element and finite volume methods:

- **Local element-wise basis**: Polynomials are discontinuous across element boundaries
- **Numerical fluxes**: Interface values computed via Riemann solvers
- **High-order accuracy**: Arbitrary polynomial order p achieves O(h^(p+1)) convergence
- **Flexibility**: Natural handling of complex geometries and adaptive refinement

### Key References

1. Cockburn, B., & Shu, C. W. (1998). *The Runge-Kutta discontinuous Galerkin method for conservation laws*
2. Hesthaven, J. S., & Warburton, T. (2008). *Nodal Discontinuous Galerkin Methods*
3. Kopriva, D. A. (2009). *Implementing Spectral Methods for PDEs*

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is developed for academic and research purposes.

## 👤 Author

**Florent Distree**

- GitHub: [@GaleAxiom](https://github.com/GaleAxiom)

## 🙏 Acknowledgments

- **Eigen** - High-performance linear algebra library
- **GMSH** - Mesh generation and processing
- **Google Test** - C++ testing framework