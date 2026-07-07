# Discontinuous Galerkin Finite Element Method (DG-FEM)

[![CMake Build](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/cmake-single-platform.yml)
[![Code Coverage](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/code-coverage.yml/badge.svg)](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/code-coverage.yml)
[![Format Check](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/format-check.yml/badge.svg)](https://github.com/GaleAxiom/Discontinuous-Galerkin-FEM/actions/workflows/format-check.yml)

A high-performance C++20 implementation of the Discontinuous Galerkin Finite Element Method for solving partial differential equations (PDEs).

## 🎯 Features

### Numerical Methods
- **Discontinuous Galerkin (DG) formulation** for hyperbolic and elliptic PDEs
- **High-order polynomial basis functions**: Legendre, Dubiner, and monomial bases
- **Flexible quadrature rules**: Gauss-Legendre integration on triangles and quadrilaterals
- **Time-stepping schemes**: Explicit Runge-Kutta methods for time-dependent problems
- **Geometric mappings**: Support for curved and affine element transformations
- **Shock capturing** (order-1 quad, x-direction, opt-in): pluggable
  `TroubledCellIndicator`/`ReconstructionTechnique` strategy classes on
  `CompressibleDGSolverBase`, defaulting to a Persson-Peraire troubled-cell indicator + WENO
  reconstruction (see `dgfem/include/dgfem/solver/dg_solver.hpp`'s `set_limiter_enabled()` doc
  for what this measurably does and doesn't buy you -- it's a robustness tool for hard problems,
  not a free accuracy win on easy ones)

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
- **Trilinos** (built from source; see `TRILINOS.md` for the recipe)
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

- **`convergence_test.cpp`** - Convergence rate verification
- **`high_order_test.cpp`** - High-order (p > 3) accuracy demonstration
- **`sod_shock_tube_example.cpp`** - Sod shock tube vs. the exact Riemann solution, comparing
  unlimited / old always-on minmod / the default Persson-Peraire+WENO shock-capturing limiter

`dgfem/examples/archive/` also contains Euler/Navier-Stokes examples (isentropic vortex,
Taylor-Green vortex, shear layer, acoustic wave, von Karman vortex street) from earlier in the
Eigen-to-Trilinos migration. They aren't currently wired into `dgfem/examples/CMakeLists.txt`
and don't build -- treat them as reference code to port forward, not runnable examples.

### Running Examples

```bash
cd build/examples

# Run Laplace solver
./laplace_solver_test

# Run the convergence-rate study
./convergence_test

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

# Point CMake at a non-default Trilinos install prefix
cmake -B build -S dgfem -DCMAKE_PREFIX_PATH=/path/to/trilinos-install
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
4. Persson, P.-O., & Peraire, J. (2006). *Sub-cell shock capturing for discontinuous Galerkin methods* (the troubled-cell indicator used by the default shock-capturing limiter)
5. Qiu, J., & Shu, C.-W. (2005). *Runge-Kutta discontinuous Galerkin method using WENO limiters* (the reconstruction technique used by the default shock-capturing limiter)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Quick start:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. **Format your code** using clang-format:

   ```bash
   ./format-code.sh
   ```

4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Style

This project uses `clang-format` to ensure consistent code formatting. The configuration is defined in `.clang-format` at the root of the repository. Key style guidelines:

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters maximum
- **Pointer/Reference**: Left-aligned (`Type* ptr`, `Type& ref`)
- **Braces**: LLVM style (opening brace on same line)
- **Naming**: Snake_case for variables, PascalCase for classes

The CI pipeline automatically checks code formatting. Make sure to run `./format-code.sh` before committing!

For more details, see [CODE_STYLE.md](CODE_STYLE.md) and [CI_CD.md](CI_CD.md).

## 📄 License

This project is developed for academic and research purposes.

## 👤 Author

**Florent Distree**

- GitHub: [@GaleAxiom](https://github.com/GaleAxiom)

## 🙏 Acknowledgments

- **Trilinos** - High-performance linear algebra and solver library
- **GMSH** - Mesh generation and processing
- **Google Test** - C++ testing framework