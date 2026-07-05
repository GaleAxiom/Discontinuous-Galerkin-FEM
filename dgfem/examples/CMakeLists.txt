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

# Archived examples (euler_vortex_test, shear_flow_example, taylor_green_test,
# acoustic_wave_test, von_karman_vortices_example) live under examples/archive/
# and are intentionally not built.
