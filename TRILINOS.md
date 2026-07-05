# Trilinos Setup for DG-FEM

This branch (`trilinos-migration`) replaces Eigen with Trilinos for all linear algebra:
Tpetra/Belos/Ifpack2/Amesos2/MueLu for global sparse assembly and solves, Kokkos for local
per-element dense math. There is no prebuilt Trilinos package for macOS ARM64 (no Homebrew
formula, no arm64 conda-forge build), so it must be built from source.

## Why these choices

- **No MPI** (`-DTPL_ENABLE_MPI=OFF`): this solver runs single-machine. Skipping MPI avoids
  `MPI_Init`/rank-map complexity everywhere; every executable still needs a `Kokkos::ScopeGuard`
  around `main()`, but nothing more.
- **No Fortran** (`-DTrilinos_ENABLE_Fortran=OFF`): Teuchos, Kokkos, KokkosKernels, Tpetra,
  Belos, Ifpack2, Amesos2, and MueLu are all pure C++ packages. Fortran is only pulled in by
  older Epetra/ML-era packages, which we don't enable.
- **OpenBLAS for both BLAS and LAPACK**: Apple's Accelerate framework has known ABI friction
  with some Trilinos TPL expectations; OpenBLAS (already available via Homebrew) is the safer,
  more predictable choice, and it ships both BLAS and LAPACK symbols in one library.
- **C++20**: Trilinos 17.x's build system *requires* `CMAKE_CXX_STANDARD` to be 20 or 23 (it
  hard-errors on 17) — this happens to match the project's own C++20 standard exactly, so no
  standard mismatch to worry about.
- **Kokkos Serial backend only**: no CUDA/HIP applicable on this hardware; OpenMP on macOS with
  system Clang is a known source of toolchain pain and isn't needed for a correctness-focused
  migration.
- **Minimal package set**: only the packages this project actually uses are enabled (see below).
  `Trilinos_ENABLE_TESTS`/`_EXAMPLES` are off to cut build time substantially.

## Prerequisites

```bash
brew install openblas cmake ninja
```

## Build (one-time, ~1-3+ hours depending on machine)

```bash
# 1. Clone a pinned release tag (shallow, single branch)
mkdir -p ~/trilinos-src ~/trilinos-build ~/trilinos-install
cd ~/trilinos-src
git clone --branch trilinos-release-17-1-1 --single-branch --depth 1 \
    https://github.com/trilinos/Trilinos.git .

# 2. Configure
cd ~/trilinos-build
cmake -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/trilinos-install \
  -DCMAKE_CXX_STANDARD=20 \
  -DTPL_ENABLE_MPI=OFF \
  -DTrilinos_ENABLE_Fortran=OFF \
  -DTrilinos_ENABLE_TESTS=OFF \
  -DTrilinos_ENABLE_EXAMPLES=OFF \
  -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
  -DTrilinos_ENABLE_ALL_PACKAGES=OFF \
  -DTrilinos_ENABLE_Teuchos=ON \
  -DTrilinos_ENABLE_Kokkos=ON \
  -DTrilinos_ENABLE_KokkosKernels=ON \
  -DTrilinos_ENABLE_Tpetra=ON \
  -DTrilinos_ENABLE_Belos=ON \
  -DTrilinos_ENABLE_Ifpack2=ON \
  -DTrilinos_ENABLE_Amesos2=ON \
  -DTrilinos_ENABLE_MueLu=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DTPL_ENABLE_BLAS=ON \
  -DTPL_BLAS_LIBRARIES=/opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  -DTPL_ENABLE_LAPACK=ON \
  -DTPL_LAPACK_LIBRARIES=/opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  $HOME/trilinos-src

# 3. Build (this is the long part)
ninja -j 8

# 4. Install
ninja install
```

Adjust `-j 8` to your core count. The install lands in `~/trilinos-install`, **not** inside this
repo (it's a multi-hundred-MB build artifact, unlike the small vendored Eigen submodule it
replaces).

## Building this project against Trilinos

```bash
cmake -B build -S dgfem -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$HOME/trilinos-install
cmake --build build -j 8
cd build && ctest
```

## Troubleshooting

- **`CMAKE_CXX_STANDARD=17 is not in the allowed set (20|23)`**: use `-DCMAKE_CXX_STANDARD=20`,
  not 17 — Trilinos 17.x's TriBITS build system enforces this itself regardless of what the
  *consuming* project's standard is.
- **Trilinos not found by `find_package`**: pass `-DCMAKE_PREFIX_PATH=$HOME/trilinos-install`
  (or `-DTrilinos_DIR=$HOME/trilinos-install/lib/cmake/Trilinos`) explicitly when configuring
  `dgfem`.
- **Build fails in `MueLu_Maxwell1_def.hpp` with "use 'template' keyword to treat 'get' as a
  dependent template name"**: a real strict-C++20-parsing issue in Trilinos 17.1.1's MueLu
  `Maxwell1` component (electromagnetics-specific multigrid, unrelated to this project, but
  compiled as part of the core `muelu` library regardless). Fix by adding `template` before the
  4 affected `.get<std::string>(...)` calls (lines ~357 x2, 540, 573 in that file) so they read
  `.template get<std::string>(...)`. This is a source patch to the local Trilinos checkout, not
  to this project.
- **`find_package(Trilinos REQUIRED COMPONENTS ...)` fails with "Galeri::all_libs ... target was
  not found"**: Xpetra/MueLu pull in Galeri as a transitive dependency at Trilinos's own build
  time, but restricting `COMPONENTS` to only the packages *we* asked for omits Galeri's own
  config from being processed. Simplest fix: drop the `COMPONENTS` list entirely and use
  `find_package(Trilinos REQUIRED)` — the enable/disable decisions were already made once, at
  Trilinos's own build/configure step; re-restricting components at `find_package` time in the
  consuming project isn't necessary and just reintroduces this kind of transitive-dependency gap.
- **Amesos2 solver creation fails to compile with "no type named 'scalar_t' in
  `Amesos2::MultiVecAdapter<Tpetra::Vector<>>`"**: Amesos2's adapter isn't specialized for plain
  `Tpetra::Vector`. Use `Tpetra::MultiVector` (with 1 column) for the RHS/solution vectors passed
  to `Amesos2::create(...)` instead — this is the standard, portable pattern Trilinos examples
  use, not a `Tpetra::Vector`-specific limitation you need to work around otherwise.
