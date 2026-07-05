# Handoff: Trilinos migration (Eigen -> Trilinos/Kokkos)

Branch: `trilinos-migration` (checked out, NOT pushed). Plan file (already approved by the user,
do not re-litigate the approach): `/Users/florentdistree/.claude/plans/okay-but-i-want-drifting-lampson.md`

**Read that plan file first.** This document is the concrete, current-state supplement to it.

## Goal, in one paragraph

Remove Eigen entirely from this C++20 DG-FEM codebase (library AND tests — zero
`#include <Eigen/...>` anywhere when done), replacing it with Trilinos: Tpetra/Belos/Ifpack2/
Amesos2/MueLu for global (distributed) sparse linear algebra, and Kokkos::View/Kokkos::Array for
local per-element dense math. The proximate motivation: a hand-rolled BiCGStab+block-Jacobi
preconditioner (written earlier this session, still in `compressible_solver_base.cpp`) works but
isn't strong enough at scale — Belos+MueLu should replace it. That replacement is itself part of
this migration (Milestone 4), not yet done.

## Current state: uncommitted, mid-flight, does NOT compile yet

`git status` on this branch shows every file below as modified, plus one new untracked file
(`dgfem/include/dgfem/kokkos_math.hpp`). **Nothing has been committed on this branch yet.** The
library target (`dgfem`) does not currently build. Do not assume anything compiles without
checking — verify with the commands in "How to check progress" below.

## What's actually done (verified compiling, file-by-file)

Trilinos itself: built from source, installed at `~/trilinos-install` (see `TRILINOS.md` in repo
root for the exact recipe, including a real Trilinos-17.1.1 MueLu bug we had to patch in the
*local Trilinos source checkout* at `~/trilinos-src` — `MueLu_Maxwell1_def.hpp` needed `template`
keywords added before 4 `.get<std::string>(...)` calls; already done, don't redo it, but if you
ever re-clone Trilinos from scratch you'll hit it again — it's documented in TRILINOS.md's
Troubleshooting section).

CMake: `dgfem/CMakeLists.txt` has `find_package(Trilinos REQUIRED)` wired in (no COMPONENTS list
— restricting it breaks on a Galeri transitive-dependency gap, see TRILINOS.md). Eigen3
find_package block is still present (not yet removed — that's Milestone 5).

Files confirmed compiling clean (verified individually via the command in the next section):

- `dgfem/include/dgfem/kokkos_math.hpp` — **new file, read this in full before doing anything
  else.** This is the shared toolkit every other file leans on. Documented in detail below.
- `dgfem/src/basis/{legendre,dubiner,monomial}.cpp` + their headers + `orthogonal.hpp`
- `dgfem/src/reference/{elements,mapping}.cpp` + headers
- `dgfem/src/quadrature/factory.cpp` + header (the big Dunavant-table file — ported via mostly
  mechanical `sed` for the repetitive `.resize(N, 2)` calls, worked cleanly)
- `dgfem/src/core/{solution,space,mesh}.cpp` + headers
- `dgfem/src/boundary/conditions.cpp` + header
- `dgfem/src/solver/trilinos_types.cpp` + header (see note below — API changed from Milestone 2)
- `dgfem/src/solver/time_stepping.cpp` + header (BlockMassMatrix class was **deleted** — it was
  fully dead code, zero callers anywhere in src/tests/examples, confirmed via grep before
  deleting; don't resurrect it)
- `dgfem/src/solver/laplace_solver.cpp`
- `dgfem/src/solver/advection_solver.cpp`
- `dgfem/src/weak_forms/weak_formulation_base.cpp` + header
- `dgfem/src/weak_forms/laplace_weak_formulation.cpp` + header
- `dgfem/src/weak_forms/advection_weak_formulation.cpp` + header
- `dgfem/src/solver/euler_solver.cpp` (thin wrapper, no dense math of its own — compiled clean
  without changes needed beyond what cascaded from headers)
- `dgfem/src/solver/navier_stokes_solver.cpp` (same — thin wrapper)

**Important API change from Milestone 2's original plan**, discovered mid-work: the plan said
"Eigen<->Tpetra conversion helpers" (`eigen_to_tpetra`/`tpetra_to_eigen`). Those have been
**renamed and retyped** to `view_to_tpetra`/`tpetra_to_view` (DView1<->Tpetra, not
Eigen::VectorXd<->Tpetra), because by the time `AdvectionDGSolver`/`LaplaceDGSolver` needed them
again, `DGSolution::set_global_coeffs()` etc. had already moved to `DView1`. If you see any
lingering references to `eigen_to_tpetra`/`tpetra_to_eigen` anywhere (there may be leftover ones
in test files), they need the same rename. Same for `tpetra_to_dense` — still exists, now returns
`DView2` instead of `Eigen::MatrixXd`.

**Important design correction discovered mid-work**: `Eigen::VectorXd`/`Eigen::MatrixXd` in the
original code was used for TWO semantically different things that a blanket find-replace
conflates: (a) local per-element blocks (small, n_basis-ish size) — these correctly become
`DView1`/`DView2`; (b) *global* whole-mesh vectors (size = n_elements*n_basis, e.g. thousands of
entries) — these should become `Teuchos::RCP<TpetraMultiVector>`, NOT `DView1`, because
`SSP_RK::step_rk3<T>` needs vector arithmetic and Tpetra::MultiVector already has `.update()`/
`.scale()` for that, whereas bare `Kokkos::View` has none (no operator overloading at all).
`advection_solver.cpp` got this wrong on the first automated pass (a blanket sed) and had to be
manually rewritten to fix it — see that file for the corrected pattern (global vector = Tpetra
RCP, local blocks in `M_inv_blocks_` = `std::vector<DView2>`). **When you port
`compressible_solver_base.cpp`'s `StateVector` usage and `euler_solver.cpp`/
`navier_stokes_solver.cpp`, double check whether each vector-like variable is local-per-element
(→ DView1/DView2, fine as `StateVector = std::vector<DView2>` already is) or whole-mesh-global
(→ Tpetra) before mechanically renaming.** StateVector itself (`std::vector<DView2>`, one block
per element) is correctly the *local* pattern already — that one's fine as-is.

## The `kokkos_math.hpp` toolkit — read this before writing any new porting code

This header (`dgfem/include/dgfem/kokkos_math.hpp`) is the single shared math layer. Do not
reinvent any of these — grep for them first if you think you need something similar:

```cpp
using Vec2 = Kokkos::Array<double, 2>;      // physical/reference 2D points
using Vec4 = Kokkos::Array<double, 4>;      // conserved/primitive-variable tuples [rho,u,v,p] etc
using DView1 = Kokkos::View<double*, Kokkos::HostSpace>;   // dynamic vector (local OR flattened)
using DView2 = Kokkos::View<double**, Kokkos::HostSpace>;  // dynamic matrix
using IView1 = Kokkos::View<int*, Kokkos::HostSpace>;
using IView2 = Kokkos::View<int**, Kokkos::HostSpace>;

// Vec2/Vec4 arithmetic: operator+, operator-, operator*(scalar), unary operator-, +=, -=, /=
// dot(Vec2,Vec2), dot(Vec4,Vec4), norm(Vec2), norm(Vec4), normalized(Vec2)
// norm(const DView1&) -- separate overload, arbitrary-length L2 norm (NOT a member function!)

struct Mat2 { double m00,m01,m10,m11; det(); transpose(); inverse(); apply(Vec2) -> Vec2; };
// Used for the 2x2 Jacobian/inverse-Jacobian in MappingData (reference/mapping.hpp).

// Row/column extraction (replaces Eigen's .row()/.col()/.block()):
Vec2 row2(const DView2&, int i);            // row i of an (n,2) view, as Vec2
void set_row2(DView2&, int i, const Vec2&); // write a Vec2 into row i of an (n,2) view
DView1 row_of(const DView2&, int i);        // row i of an (n,m) view, arbitrary width, as DView1
DView1 col_of(const DView2&, int j);        // column j of an (n,m) view, as DView1
Vec2 to_vec2(const DView2& col);            // read a (2,1)-shaped view (e.g. stored "normal") as Vec2
double scalar_of(const DView2& m);          // read a (1,1)-shaped view (e.g. stored "length") as double

// Outer-product accumulate (replaces Eigen's `v * u.transpose()` pattern): M(i,j) += alpha*v(i)*u(j)
void outer_add(DView2& M, double alpha, const DView1& v, const DView1& u);

// Real matrix algebra -- via KokkosBlas (Trilinos-provided, not hand-rolled; user explicitly
// asked to use Trilinos's own coded matrix ops here rather than write new ones or switch to
// Teuchos::SerialDenseMatrix):
void gemm(char transA, char transB, double alpha, const DView2& A, const DView2& B,
         double beta, DView2& C);   // C = alpha*op(A)*op(B) + beta*C
void gemv(char trans, double alpha, const DView2& A, const DView1& x,
         double beta, DView1& y);   // y = alpha*op(A)*x + beta*y
DView2 invert_dense(const DView2& A);  // dense inverse via KokkosBatched::SerialInverseLU
                                        // (used for mass-matrix block inversion, n_basis x n_basis,
                                        // NOT always 2x2 so Mat2 doesn't apply)
```

**Element/face data maps**: `DGSpace::compute_element_data()` and `compute_face_data()` were
*unified* to both return `std::map<std::string, DView2>` (previously the codebase had a split
between `map<string,MatrixXd>` for element data and `map<string,VectorXd>` for face data, plus a
lazy conversion cache in `mesh.cpp` bridging them — that whole cache mechanism, including
`face_data_matrix_`, was **deleted** as part of this unification; it's genuinely simpler now, one
type everywhere). Consequence: every stored "vector-like" quantity (J_det_vol, length,
neighbor_elem, is_boundary, etc.) is stored as an (N,1)-shaped DView2, not a flat DView1 — access
via `.at("key")(i, 0)` or `scalar_of(...)` for single values, not `[i]`. `DGMesh::get_face_data()`
is now just an alias for `get_element_face_data()` (see `mesh.hpp`) — both return the same unified
type; no more separate raw/converted variants.

**One correctness bonus discovered from this unification**: the old code flattened face
`quad_points` (an (n_quad,2) matrix) into a flat VectorXd via a column-major `.data()` reinterpret,
which was *separately* re-interpreted elsewhere (`navier_stokes_weak_formulation.cpp`, the
lift-force code from earlier in the session) as row-major (n_quad,2) — a latent
byte-layout mismatch that likely produced wrong values whenever `quad_points` was actually used
for physical coordinates (needs checking once NS weak form is fixed — see below). Under the new
unified scheme, `quad_points` is stored as a genuine (n_quad,2) `DView2` from the start
(`DGSpace::compute_face_data()` in `space.cpp`) — no flatten/reinterpret needed at all, so this
bug class is structurally gone. `euler_weak_formulation.cpp`'s `boundary_face_residual()` (not
yet rewritten — see below) has an `Eigen::Map<...RowMajor...>` reinterpret hack for exactly this
that should simply be deleted and replaced with `row2(quad_points_mat, q)` once you rewrite that
function — I had this analysis done but hadn't written it to disk when interrupted.

**Mesh helper added**: `DGMesh::get_element_vertices(int elem_id) const -> DView2` — gathers one
element's physical vertex coordinates into a fresh (n_verts, 2) view. This exact "gather vertices"
loop was duplicated ~5+ times across the original codebase (laplace_solver, advection_solver,
compressible_solver_base, mesh.cpp itself, weak_formulation_base assemble loops) — always use
this helper instead of re-writing the gather loop; several already-fixed files were simplified to
use it (`mesh.cpp`'s own `initialize_dg_space`/`build_precomputed_faces`,
`laplace_weak_formulation.cpp`, `laplace_solver.cpp`).

## An UNVERIFIED assumption load-bearing across every already-written file

Every file so far relies on `Kokkos::View`'s constructor (`DView2("label", n, m)`) **value/zero-
initializing** its contents by default, the same way `Eigen::MatrixXd::Zero(n,m)` used to be
explicit about. This is standard, well-documented Kokkos behavior (default View construction
zero-initializes for arithmetic types; only `Kokkos::view_alloc(Kokkos::WithoutInitializing, ...)`
skips it, which nothing here uses) — but I was in the middle of empirically double-checking this
in *this exact build* when interrupted, and hadn't gotten a clean run (kept hitting environment
issues: wrong Kokkos on the default include path, forgetting `-c` produces an object file not an
executable, permissions on a redirected scratch path — none of these were signal about the actual
answer, just tooling friction). **First thing to do: settle this for certain**, e.g. by writing a
2-line gtest (`Kokkos::View<double**> v("t",3,3); EXPECT_EQ(v(1,1), 0.0);`) and running it via the
normal test build (reuses working include paths, no more manual compile-command hacking needed).
If it turns out Views do NOT zero-init by default in this Kokkos build for some reason, every
`DView2("label", n, m)` accumulator pattern in every file listed above needs an explicit
`Kokkos::deep_copy(view, 0.0)` after construction — that would be a mechanical but wide-reaching
fix. High confidence this is a non-issue, but confirm before trusting any numerical output.

## What's left, in order

### 1. Finish the weak_forms layer (in progress, analysis done, not yet written)

- **`dgfem/src/weak_forms/euler_weak_formulation.cpp`** — I had fully read and analyzed this file
  and knew the exact rewrite needed, but had not yet written it when interrupted. Key points for
  whoever does it:
  - `struct FluxAndPrimitive` / `get_fluxes_and_primitive` / `get_fluxes` / `rusanov_flux`: mostly
    fine already (Vec4 arithmetic all supported), except Eigen's comma-initializer
    (`F << rho_u, ...;`) needs to become `F = Vec4{rho_u, ...};` (or construct directly at
    declaration).
  - `volume_residual`: `u_coeffs_elem.cols()` → `.extent(1)`. The `Eigen::ArrayXd w_phys =
    weights.array() * J_det.array().abs()` precompute has no direct Kokkos analog — just inline
    `weights[q] * std::abs(J_det(q,0))` per-iteration instead (J_det is now DView2 (n,1), see
    unification note above). `const DView2 U_quad_points = phi * u_coeffs_elem;` is a genuine
    matrix-matrix product (phi: n_quad x n_basis times u_coeffs_elem: n_basis x n_vars) — use
    `gemm('N','N', 1.0, phi, u_coeffs_elem, 0.0, U_quad_points)` (allocate U_quad_points first).
    `U_quad_points.row(q).transpose()` → build a `Vec4` directly:
    `Vec4{U_quad_points(q,0), U_quad_points(q,1), U_quad_points(q,2), U_quad_points(q,3)}`. The
    `grad_phi_q.block(...).col(0)*F_q.transpose() + ...col(1)*G_q.transpose()` outer-product-sum
    is simplest rewritten as a direct double loop (see pattern already used successfully in
    `laplace_weak_formulation.cpp`'s `outer_add` calls, or just inline
    `R_vol(i,v) += w*(dphi_dx(q*n_basis+i,0)*F_q[v] + dphi_dx(q*n_basis+i,1)*G_q[v])` over i,v —
    simpler than forcing it through `outer_add` since F_q/G_q are Vec4 not DView1).
  - `interior_face_residual`/`boundary_face_residual`: mostly index-based already (uses manual
    loops per the file's own "OPTIMIZATION" comments), just needs: `face_data.at("weights")` is
    now DView2 not DView1 (`.extent(0)`, `(q,0)` indexing); `face_data.at("normal").col(0)` →
    `to_vec2(face_data.at("normal"))` (returns by value, can't bind to `const Vec2&`, use
    `Vec2 normal = ...`); `face_data.at("length")(0,0)` already matches DView2 directly, no change
    needed there; delete the `Eigen::Map<...RowMajor...>` quad_points reinterpret entirely and use
    `Vec2 x_q = row2(quad_points_mat, q)` (see the correctness-bonus note above);
    `Vec2 u_L(W_L_q[1], W_L_q[2])` → `Vec2 u_L{W_L_q[1], W_L_q[2]}`; `W_R_q << a,b,c,d` →
    `Vec4 W_R_q{a,b,c,d}`.
  - `primitive_to_conserved`/`conserved_to_primitive`: already fine as pure scalar/Vec4 element
    access, shouldn't need changes.
  - `viscous_*_residual` stub methods (real implementation lives in `navier_stokes_weak_formulation.cpp`
    which overrides them): just `DView2::Zero(n,m)` → `DView2("label", n, m)`.

- **`dgfem/src/weak_forms/navier_stokes_weak_formulation.cpp`** — not started at all. This is the
  file with genuine viscous-flux tensor math (stress tensor, heat flux, SIPG penalty for viscous
  terms) — expect it to be the most involved of the four weak-form files. Same toolkit applies
  (row2/row_of/dot/outer_add/gemm/gemv/Mat2). Its header
  (`navier_stokes_weak_formulation.hpp`) already got the safe bulk rename
  (`Eigen::Vector2d`→`Vec2` etc.) but the `.cpp` has real matrix-algebra call sites (`.transpose()`,
  `.block()`, `.dot()`) needing the same treatment as the other three weak-form files.

### 2. `compressible_solver_base.cpp` — the actual motivating fix

`StateVector = std::vector<DView2>` (type already correct). The file still has the OLD hand-rolled
`bicgstab()` + graph-coloring block-Jacobi preconditioner (written earlier this session, before
the Trilinos decision) using Eigen-flavored calls (`.row()`, `DView2::Zero()`, `.inverse()`,
`.cols()`, `.transpose()` on Vec4) that need the same mechanical fixes as everywhere else — but
the REAL task per the plan is to **replace** that hand-rolled solver with Belos+MueLu, not just
port its syntax:
- Wrap the existing matrix-free `A_op` lambda as a `Tpetra::Operator` subclass.
- Extend the *already-correct* graph-coloring FD-probing technique (verified mathematically
  correct earlier this session, don't redo that verification, just reuse the coloring +
  probing logic) to insert probed values into a global `Tpetra::CrsMatrix` instead of per-element
  local blocks only — this gives MueLu a real (approximate/lagged) Jacobian to build a multigrid
  hierarchy from.
- Keep a safety-fallback (reject non-converged/blown-up corrections) — Belos exposes convergence
  status directly, use that instead of the hand-rolled residual tracking.
- This is flagged in the plan as the single biggest source of schedule uncertainty — it's new
  integration work, not mechanical porting. Budget real time for it.

### 3. `dgfem/src/solver/euler_solver.cpp`, `navier_stokes_solver.cpp`

Already compile clean as of this handoff — but that was BEFORE `compressible_solver_base.cpp`'s
API potentially changes shape from the Belos/MueLu work in step 2. Re-check after step 2 lands.

### 4. `dgfem/src/utils/mesh_creation.cpp` (+ header), `dgfem/src/utils/vtk_writer.cpp` (+ header),
   `dgfem/include/dgfem/utils/example_helpers.hpp`

Not started. `mesh_creation.cpp` is the GMSH ingestion glue (~400 lines) — GMSH itself is
untouched (not Eigen-based), only the code copying GMSH's raw arrays into `DView2`/`IView2` needs
work. Known errors already surfaced: `.resize()` calls on views (same fix as everywhere —
reassign via constructor instead), `.row()` calls, and two "out-of-line definition does not match
declaration" errors for `create_cylinder_channel_mesh`/`extract_mesh_data` — check the header
signature was updated to match wherever the .cpp got a mechanical type-rename but the .hpp
declaration didn't (or vice versa) before doing anything more invasive.

### 5. Full-build iteration across `dgfem/examples/*.cpp`

Every example program (`taylor_green_test.cpp`, `von_karman_test.cpp`, the two viscous variants
added this session, etc.) also constructs `Eigen::Vector2d`/`Vector4d`/etc. directly in their
`main()` — they'll all need the same `Vec2`/`Vec4` treatment once the library itself compiles.
Not started; will surface as a normal build-error list once the library target succeeds.

### 6. Test suite (170 tests, ~20 files) — likely the largest remaining chunk

Only `dgfem/tests/solver/assembler_test.cpp` has been touched so far (partially, via the same
bulk `eigen_to_tpetra`→`view_to_tpetra` sed — NOT fully reworked; its `MOCK_METHOD` signatures
still declare `Eigen::MatrixXd`/`Eigen::Vector2d` return/param types that no longer match the real
(now `DView2`/`Vec2`) virtual methods they're supposed to override — gmock mocks need their
signatures updated to match exactly, same pattern as the "override hides virtual function" errors
seen earlier when the weak-form headers were mid-migration). Every other test file is completely
untouched. Expect heavy use of Eigen for constructing fixtures/expected values
(`Eigen::MatrixXd::Identity(3,3)`, hard-coded numeric matrix literals compared element-by-element,
etc.) — this was flagged in the plan as needing the most manual attention of any milestone.
Do this LAST, after the library itself compiles and links, so you're fixing tests against a
stable API rather than a moving target.

### 7. Milestone 5 cleanup (only after everything above is green)

- Remove `external/eigen` git submodule + `.gitmodules` entry + the Eigen3 `find_package`
  fallback block in `dgfem/CMakeLists.txt`.
- `grep -r "Eigen" dgfem/` should return zero hits.
- Also worth a final grep for `Eigen::Triplet` specifically — `DGAssembler`'s `triplets_` member
  still uses `std::vector<Eigen::Triplet<double>>` (a tiny (row,col,value) POD holder) even though
  the surrounding assembly logic is fully Tpetra now; this is trivial to replace with a local
  3-field struct but hasn't been done yet since it's harmless glue, not blocking anything.
- Full `ctest` run, all green.
- Re-run `taylor_green_viscous_example` at the exact configs that diverged earlier this session
  (`dt=1e-2` and `dt=1e-3` on the 3844-element mesh) as the real regression check for why this
  migration started — confirm zero solver-failure warnings and tight KE-decay-rate agreement with
  theory through the full run. This is the actual acceptance test for the whole migration, not
  just "does ctest pass."

## How to check progress (commands that work right now)

Check one file's compile status without a full rebuild (fast iteration loop used throughout):

```bash
cd /Users/florentdistree/Documents/Uni/Discontinuous-Galerkin-FEM/build
make -f CMakeFiles/dgfem.dir/build.make CMakeFiles/dgfem.dir/src/<path>.cpp.o 2>&1 | head -100
```

Full library build (will stop after ~20 errors per file by default, `-ferror-limit`):

```bash
cd /Users/florentdistree/Documents/Uni/Discontinuous-Galerkin-FEM
cmake --build build --target dgfem -j 8 2>&1 | grep -E "error:"
```

CMake was already reconfigured with Trilinos on this branch (`-DCMAKE_PREFIX_PATH=$HOME/trilinos-install`
baked into the existing `build/` dir's CMakeCache) — you should NOT need to reconfigure from
scratch unless CMakeLists.txt itself changes. If you do need to reconfigure, see `TRILINOS.md`.

Per-file status as of this handoff (re-run the loop below to refresh; some may have shifted if
headers changed since):

```bash
cd build
for f in basis/legendre basis/dubiner basis/monomial reference/elements reference/mapping \
         quadrature/factory core/solution core/space core/mesh boundary/conditions \
         solver/trilinos_types solver/time_stepping solver/laplace_solver solver/advection_solver \
         solver/assembler weak_forms/weak_formulation_base weak_forms/laplace_weak_formulation \
         weak_forms/advection_weak_formulation weak_forms/euler_weak_formulation \
         weak_forms/navier_stokes_weak_formulation solver/compressible_solver_base \
         solver/euler_solver solver/navier_stokes_solver utils/mesh_creation utils/vtk_writer; do
  n=$(make -f CMakeFiles/dgfem.dir/build.make CMakeFiles/dgfem.dir/src/${f}.cpp.o 2>&1 | grep -c "error:")
  [ "$n" = "0" ] && echo "OK   $f" || echo "FAIL $f ($n errors)"
done
```

At last check: everything OK except `solver/assembler` (12 errors — see the StateVector/
`Eigen::MatrixXd`-leftover notes above), `weak_forms/euler_weak_formulation` (20, capped by
error-limit, real count higher), `weak_forms/navier_stokes_weak_formulation` (20, not started),
`solver/compressible_solver_base` (20, needs the Belos/MueLu work), `utils/mesh_creation` (20),
`utils/vtk_writer` (20).

## General lessons learned this session (apply going forward)

- **Don't blanket-sed `Eigen::MatrixXd`/`VectorXd` → `DView2`/`DView1` and assume you're done.**
  It's a safe *type-name* rename but leaves behind every Eigen *method call* (`.row()`,
  `.transpose()`, `.block()`, `::Zero()`, `.inverse()`, `.dot()`, `.cols()`, `.norm()`,
  `.squaredNorm()`, `.normalize()`, `<<` comma-init) that Kokkos::View/Array simply don't have.
  The safe bulk-sed is a fine FIRST pass to shrink the diff, but always compile-check immediately
  after and expect a second, manual pass per file for the method-call fixes. This was the
  single biggest source of rework this session (see the `advection_solver.cpp` global-vs-local
  vector confusion above, which came from exactly this).
- **Compile after every file, not after every milestone.** The pattern that worked well: write/
  fix one `.cpp` (+ its `.hpp` if needed), immediately run the single-file compile check above,
  fix, repeat. Don't batch multiple files before checking — errors compound and get harder to
  isolate.
- When a type change in one file ripples into a virtual-function signature mismatch elsewhere
  ("non-virtual member function marked 'override' hides virtual member function" /  "has a
  different return type than the function it overrides"), that's the compiler telling you a
  *derived* class header didn't get the same treatment as its base class — fix the derived
  header's signature to match exactly, don't fight the compiler on this one.
- If you find genuinely dead code during porting (like `BlockMassMatrix` was), grep for callers
  across `src/`, `tests/`, `examples/` to confirm zero usage, then delete rather than port it —
  confirmed with the user this session that minimizing new/ported code is a real priority, not
  just correctness.
