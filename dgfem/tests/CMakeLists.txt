cmake_minimum_required(VERSION 3.10)

include(GoogleTest)

# Google Test should already be found in parent CMakeLists.txt
include_directories(${GTEST_INCLUDE_DIRS} .)

# Add test executable
add_executable(dgfem_tests
    basis/legendre_test.cpp
    basis/orthogonal_test.cpp
    basis/dubiner_test.cpp
    basis/monomial_test.cpp
    reference/mapping_test.cpp
    reference/elements_test.cpp
    reference/quad_test.cpp
    reference/face_integral_test.cpp
    quadrature/factory_test.cpp
    solver/assembler_test.cpp
    solver/weak_form_test.cpp
    solver/dg_solver_test.cpp
    solver/convergence_test.cpp
    solver/p_convergence_test.cpp
    solver/time_dependent_test.cpp
    solver/navier_stokes_weak_formulation_test.cpp
    solver/shear_flow_accuracy_test.cpp
    solver/compressible_blasius_accuracy_test.cpp
    boundary/boundary_conditions_test.cpp
    utils/mesh_creation_test.cpp
    utils/visualization_test.cpp
    utils/vtk_writer_test.cpp
    utils/gmsh_vtk_writer_test.cpp
)

# Link against Google Test and our library
target_link_libraries(dgfem_tests
    PRIVATE
    dgfem
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    GTest::gmock_main
)

# Automatically create individual ctest entries for every GoogleTest
gtest_discover_tests(
    dgfem_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_TIMEOUT 300
    PROPERTIES LABELS "dgfem"
)

# # Convenience aggregate test for running the full suite from ctest
# add_test(NAME dgfem_tests_all COMMAND dgfem_tests)
# set_tests_properties(dgfem_tests_all PROPERTIES LABELS "dgfem")