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

# Euler vortex test example
add_executable(euler_vortex_test euler_vortex_test.cpp)
target_link_libraries(euler_vortex_test dgfem)

# Navier-Stokes shear flow example
add_executable(shear_flow_example shear_flow_example.cpp)
target_link_libraries(shear_flow_example dgfem)

# Taylor-Green vortex test example
add_executable(taylor_green_test taylor_green_test.cpp)
target_link_libraries(taylor_green_test dgfem)

# Acoustic wave test example
add_executable(acoustic_wave_test acoustic_wave_test.cpp)
target_link_libraries(acoustic_wave_test dgfem)

# High-order elements test example
add_executable(high_order_test high_order_test.cpp)
target_link_libraries(high_order_test dgfem)

# von Kármán vortex street example
add_executable(von_karman_vortices_example von_karman_vortices_example.cpp)
target_link_libraries(von_karman_vortices_example dgfem)
