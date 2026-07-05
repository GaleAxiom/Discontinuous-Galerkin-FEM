# Handoff: High-order Euler/NS instability — RESOLVED

## Summary

The Euler/NS DG solver's divergence at polynomial order >= 3 (`dgfem/examples/archive/euler_vortex_test.cpp`
diverging to exactly `DBL_MAX` around timestep 170) is **fixed**. Root cause: `DGSpace::create_basis`
(dgfem/src/core/space.cpp) used `MonomialBasisTriangle` for triangle elements — raw monomials
(`x^i*y^j`) are textbook ill-conditioned at order >= 3. Swapping to the existing, already-tested
`DubinerBasis` (orthogonal, Jacobi-polynomial-based) fully resolves it.

## What changed

- `dgfem/src/core/space.cpp`: `DGSpace::create_basis` now returns `DubinerBasis` instead of
  `MonomialBasisTriangle` for triangles.
- `DGSpace`'s constructor gained an optional third parameter,
  `DGSpace(element_type, order, basis_override = "")`, so tests can force `"monomial"` or
  `"dubiner"` explicitly (dgfem/include/dgfem/core/space.hpp,
  dgfem/src/core/space.cpp:`create_basis`). Quads are unaffected (`basis_override` is ignored,
  only `LegendreBasis` exists there).
- No quadrature-order change was needed in the end — `DGSpace::DGSpace`'s existing
  `quad_order = 2*order` (volume) / `order+1` (face) sizing is untouched. An earlier
  `4*order`/`2*(order+1)` over-integration experiment (see git history if curious) delayed but
  didn't fix the divergence on its own; it turned out to be unnecessary once the basis was fixed
  and was reverted before this fix landed.
- No limiter was needed. The Zhang-Shu positivity-preserving limiter design notes from the
  original investigation (injection via `SSP_RK::step_rk3`, `dgfem/include/dgfem/solver/time_stepping.hpp`)
  are **not implemented** and turned out to be unnecessary at the tested resolution/order — see
  "Possible follow-up" below if this needs revisiting at higher order or coarser tolerance.

## Verification performed

Standalone repro (dx=0.5, 946 elements, T_final=0.5, dt=0.0025, 200 steps, order 3), compiled via
the same command documented below, run for all of:
- Periodic BCs + Euler (the original archived example's config) — stable, density stays in
  [0.494, 1.000] for the full 200 steps (previously diverged to `DBL_MAX` at step ~170).
- Far-field BCs + Euler — stable, same result.
- Far-field BCs + Navier-Stokes (`mu=1.0e-3`) — stable, same result.

**Also verified at a larger, longer-running case** (dx=0.2 instead of 0.5, order 3, periodic BCs,
matching the archived example's domain/IC): 5,830 elements, 8,000 timesteps (T_final=20 vs the
quick repro's T_final=0.5/200 steps — i.e. ~40x more steps, run for a duration ~47x longer than
where the original bug diverged). Density stayed in `[0.494, 1.000]` for all 801 logged
checkpoints, CFL stayed bounded (max 0.0484), no `DBL_MAX`/NaN. Runtime ~645s. VTK output in
`/tmp/dgfem_examples/output/euler_vortex_dx02_*.vtk` (41 frames), source at
`/tmp/dgfem_examples/euler_vortex_dx02.cpp`. This is meaningfully stronger evidence than the
original dx=0.5 quick repro, though still short of the full dx=0.1/T_final=100 archived
original — see "Possible follow-up" below.

Full `ctest` suite: **216/217 passing** (up from a 197/197 baseline before this investigation
started; test count grew due to concurrent unrelated work plus 9 new tests added here). The one
remaining failure, `NavierStokesWeakFormulationTest.DetectsDivergenceFromNonFiniteState`, is
**pre-existing and unrelated** to this basis change (confirmed: it fails identically whether the
triangle basis is Monomial or Dubiner) — not investigated further, out of scope for this handoff.

### Tests fixed after the basis swap (all now pass, hardcoded per-basis, no generic/basis-agnostic abstractions added)

The basis swap broke 10 previously-passing tests that baked in Monomial-specific assumptions.
Each existing test was pinned to `"monomial"` explicitly (so Monomial correctness stays covered
regardless of DGSpace's default), and a sibling `...DubinerBasis` test was added with
independently-derived (and numerically cross-checked) hardcoded expected values:

- `dgfem/tests/solver/weak_form_test.cpp`: `LaplaceWeakFormulationTest.InteriorFaceIntegral`,
  `AdvectionWeakFormulationTest.MassIntegral`/`StiffnessIntegral`/`BoundaryFaceIntegralInflow`/
  `BoundaryFaceIntegralOutflow` — Dubiner siblings added with either captured-and-symmetry-checked
  golden matrices (Laplace) or hand-derived analytic values using Dubiner's known order-1 closed
  form (`phi_0=2`, `phi_1=sqrt(6)*(4*xi+2*eta-2)`, `phi_2=2*sqrt(2)*(3*eta-1)` — verified against
  `DubinerBasis::evaluate` directly before use).
- `dgfem/tests/solver/dg_solver_test.cpp`: `LaplaceDGSolverTest.SystemMatrixTest` — same treatment.
- `dgfem/tests/solver/navier_stokes_weak_formulation_test.cpp`:
  `UniformFlowHasZeroResidual`/`ShearFlowVolumeResidualMatchesAnalytic`/
  `ShearFlowInteriorFaceResidualMatchesAnalytic` — root cause was two shared test helpers
  (`make_constant_coeffs`, and `solve_coeffs` inside `make_shear_flow_setup`) hardcoding
  Monomial-specific nodal-to-modal coefficient formulas (assuming `phi_0==1`). Both now take an
  explicit basis-kind selector and branch to a separate hardcoded Dubiner-specific formula
  (hand-derived by inverting the order-1 Dubiner Vandermonde matrix at the reference vertices) —
  not a generic/basis-agnostic quadrature projection.
- `dgfem/tests/utils/gmsh_vtk_writer_test.cpp`: `GMSHVTKWriterTest.MultiVariableOutput` — same
  `phi_0==1` assumption; fixed by dividing the target constant by the actual (hardcoded)
  `DubinerBasis` mode-0 value of `2.0`. This one wasn't duplicated per-basis since it's testing
  VTK I/O, not basis math, and only ever exercised whichever basis is DGSpace's default.

## Possible follow-up (not required, not started)

- **Verification still doesn't cover the full dx=0.1 original or the other archived examples.**
  The dx=0.2/T_final=20 run above is meaningfully harder than the original dx=0.5 quick repro
  (6x more elements, 40x more steps) and adds real confidence, but it's still not the actual
  full-resolution `dgfem/examples/archive/euler_vortex_test.cpp` as shipped (dx=0.1, 23,272
  elements, T_final=100) — nor any of the other archived examples that are plausibly
  harder/more demanding: `taylor_green_test.cpp`, `von_karman_vortices_example.cpp`,
  `shear_flow_example.cpp`, `acoustic_wave_test.cpp` (all in `dgfem/examples/archive/`, none
  currently wired into CMake). Before treating the Dubiner-basis fix as fully validated, run the
  actual unmodified `euler_vortex_test.cpp` (dx=0.1, T_final=100 — will take considerably longer
  than the ~11 min the dx=0.2 run took, likely on the order of a few hours given ~4x the
  elements and ~5x the T_final), and ideally the other archived examples too, at order 3 (and
  probably order 4+, since that's untested territory for the compressible solver).
- The Zhang-Shu limiter design from the original investigation is still a reasonable defense
  if higher orders (4+), coarser tolerances, or more extreme flow conditions ever reintroduce
  instability — the injection point (`SSP_RK::step_rk3`) and the "always use quadrature-based
  cell averages, don't assume a raw coefficient index is the mean" caveat are still valid advice
  if this is revisited, but nothing here currently requires it.
- Order >= 3 still has no *dedicated regression test* for the compressible solver's stability
  itself (only the basis/weak-form unit tests above were extended) — the two existing NS
  accuracy tests (`NavierStokesShearFlowAccuracyTest`/`NavierStokesBlasiusAccuracyTest`) still
  hardcode `order = 2`. Consider extending one of them to order 3 so this divergence can't
  silently regress unnoticed.
- `euler_vortex_test.cpp` is still archived/not wired into CMake (`dgfem/examples/CMakeLists.txt`)
  and isn't built by anything — could be un-archived now that it's stable, so it doesn't bit-rot
  again.
- Two unrelated real bugs were found *earlier* in this session (before this investigation began)
  and spawned as separate background tasks — not verified as fixed or still open as of this
  writing, check their status independently if relevant:
  - `task_8808b8a2` — periodic-boundary interior-face dedup bug. **Partially addressed**: a fix
    (dedup by `(elem,face)` pairs via `std::minmax`, not just `(elem_L,elem_R)`) landed in
    `TimeIndependentWeakFormulation::assemble` (dgfem/src/weak_forms/weak_formulation_base.cpp)
    from a concurrent session, but the same bug still exists as originally reported in
    `DGMesh::build_precomputed_faces()` (dgfem/src/core/mesh.cpp) — that function still dedups
    interior faces by `(elem_L, elem_R)` only. Worth fixing there too for consistency.
  - `task_f3a94d87` — Laplace Neumann/Robin BCs. **Confirmed fixed** by a concurrent session:
    both BC types are now implemented in `LaplaceWeakFormulation::compute_boundary_face_integral`/
    `compute_boundary_rhs_integral` (dgfem/src/weak_forms/laplace_weak_formulation.cpp).

## Reproduction commands (kept from the original investigation, still valid)

The archived example isn't wired into CMake, so it's compiled standalone. Scratch artifacts are
in `/tmp/dgfem_examples/` (may be cleaned by the OS — regenerate if gone). Compile command
(works for any `.cpp` variant there):

```bash
cd /Users/florentdistree/Documents/Uni/Discontinuous-Galerkin-FEM
/Library/Developer/CommandLineTools/usr/bin/c++ -std=c++20 -arch arm64 \
  -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX15.0.sdk -mmacosx-version-min=14.5 \
  -O3 -DNDEBUG -DKOKKOS_DEPENDENCE \
  -I dgfem/include -I /opt/homebrew/include \
  -isystem /Users/florentdistree/trilinos-install/include \
  -isystem /Users/florentdistree/trilinos-install/include/kokkos \
  <YOUR_FILE>.cpp -o <OUTPUT_NAME> \
  -Wl,-search_paths_first -Wl,-headerpad_max_install_names \
  build/libdgfem.a \
  /Users/florentdistree/trilinos-install/lib/libmuelu-adapters.a \
  /Users/florentdistree/trilinos-install/lib/libmuelu.a \
  /Users/florentdistree/trilinos-install/lib/libifpack2.a \
  /Users/florentdistree/trilinos-install/lib/libamesos2.a \
  /Users/florentdistree/trilinos-install/lib/libbelosxpetra.a \
  /Users/florentdistree/trilinos-install/lib/libbelostpetra.a \
  /Users/florentdistree/trilinos-install/lib/libbelos.a \
  /Users/florentdistree/trilinos-install/lib/libgaleri-xpetra.a \
  /Users/florentdistree/trilinos-install/lib/libxpetra.a \
  /Users/florentdistree/trilinos-install/lib/libtrilinosss.a -lm \
  /Users/florentdistree/trilinos-install/lib/libtpetraext.a \
  /Users/florentdistree/trilinos-install/lib/libtpetrainout.a \
  /Users/florentdistree/trilinos-install/lib/libtpetra.a \
  /Users/florentdistree/trilinos-install/lib/libtpetraclassic.a \
  /Users/florentdistree/trilinos-install/lib/libkokkostsqr.a \
  /Users/florentdistree/trilinos-install/lib/libteuchoskokkoscomm.a \
  /Users/florentdistree/trilinos-install/lib/libteuchoskokkoscompat.a \
  /Users/florentdistree/trilinos-install/lib/libteuchosremainder.a \
  /Users/florentdistree/trilinos-install/lib/libteuchosnumerics.a \
  /Users/florentdistree/trilinos-install/lib/libteuchoscomm.a \
  /Users/florentdistree/trilinos-install/lib/libteuchosparameterlist.a \
  /Users/florentdistree/trilinos-install/lib/libteuchosparser.a \
  /Users/florentdistree/trilinos-install/lib/libteuchoscore.a \
  /Users/florentdistree/trilinos-install/lib/libkokkoskernels.a \
  /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  /Users/florentdistree/trilinos-install/lib/libkokkoscontainers.a \
  /Users/florentdistree/trilinos-install/lib/libkokkosalgorithms.a \
  /Users/florentdistree/trilinos-install/lib/libkokkossimd.a \
  /Users/florentdistree/trilinos-install/lib/libkokkoscore.a \
  /opt/homebrew/lib/libgmsh.dylib
```

Run from a directory containing an `output/` subdirectory. Build type must be `Release`
(`cmake -DCMAKE_BUILD_TYPE=Release build/ && cmake --build build/ --target dgfem -j`).
