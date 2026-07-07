# Examples CMakeLists.txt

# Simple mesh test example
add_executable(simple_mesh_test simple_mesh_test.cpp)
target_link_libraries(simple_mesh_test dgfem)

# Laplace solver test example
add_executable(laplace_solver_test laplace_solver_test.cpp)
target_link_libraries(laplace_solver_test dgfem)

# Convergence test example
add_executable(convergence_test convergence_test.cpp)
target_link_libraries(convergence_test dgfem)

# Advection solver test example
add_executable(advection_solver_test advection_solver_test.cpp)
target_link_libraries(advection_solver_test dgfem)

# High-order elements test example
add_executable(high_order_test high_order_test.cpp)
target_link_libraries(high_order_test dgfem)

# Acoustic wave propagation test (dispersion/dissipation analysis for the Euler solver)
add_executable(acoustic_wave_test acoustic_wave_test.cpp)
target_link_libraries(acoustic_wave_test dgfem)

# Viscous shear flow example (Navier-Stokes solver against an analytic steady profile)
add_executable(shear_flow_example shear_flow_example.cpp)
target_link_libraries(shear_flow_example dgfem)

# Viscous Couette flow example (no-slip wall BCs instead of far-field ghost states)
add_executable(couette_flow_example couette_flow_example.cpp)
target_link_libraries(couette_flow_example dgfem)

# Sod shock tube (inviscid Euler against the exact Riemann solution; robustness check for
# the lack of a shock-capturing limiter)
add_executable(sod_shock_tube_example sod_shock_tube_example.cpp)
target_link_libraries(sod_shock_tube_example dgfem)

# Isentropic vortex test (smooth exact Euler solution, stationary vortex on periodic domain)
add_executable(euler_vortex_test euler_vortex_test.cpp)
target_link_libraries(euler_vortex_test dgfem)

# Remaining archived examples (taylor_green_test, von_karman_vortices_example) live under
# examples/archive/ and are intentionally not built.
