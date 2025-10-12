#include <gmsh.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <dgfem/solver/weak_form.hpp>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/utils/mesh_creation.hpp>
#include <dgfem/solver/assembler.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <Eigen/Dense>
#include <cmath>
#include "test_helpers.h"

using namespace dgfem;
using namespace testing;

TEST(LaplaceWeakFormulationTest, PenaltyParameter) {
    LaplaceWeakFormulation weak_form(10.0);
    
    // Test penalty parameter scaling with order and mesh size
    double h = 0.1;
    int p = 2;
    double sigma = weak_form.compute_penalty_parameter(p, h);
    
    // Should scale like (p+1)²/h
    EXPECT_NEAR(sigma, 10.0 * (p + 1) * (p + 1) / h, 1e-12);
    
    // Test handling of very small h
    EXPECT_NO_THROW(weak_form.compute_penalty_parameter(p, 1e-13));
}

TEST(LaplaceWeakFormulationTest, VolumeIntegral) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    LaplaceWeakFormulation weak_form(10.0);
    
    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);

    auto K_vol = weak_form.compute_volume_integral(elem_data, space);
    
    // Matrix should be symmetric
    EXPECT_EQ(K_vol.rows(), K_vol.cols());
    for (int i = 0; i < K_vol.rows(); ++i) {
        for (int j = 0; j < K_vol.cols(); ++j) {
            EXPECT_NEAR(K_vol(i,j), K_vol(j,i), 1e-12);
        }
    }
}

TEST(LaplaceWeakFormulationTest, InteriorFaceIntegral) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0, 1, 0, 1);
    LaplaceWeakFormulation weak_form(10.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto assembler = std::make_shared<DGAssembler>(mesh, std::make_shared<LaplaceWeakFormulation>(10.0));
    assembler->assemble([](const Eigen::Vector2d&){ return 0.0; });
    auto system_matrix = assembler->get_system_matrix();


    // Create expected matrix
    Eigen::MatrixXd expected(12, 12);
    expected <<
        80,      39,    20.5,     -40,   -19.5,     -20,     -40,   -19.5,    -0.5,       0,       0,       0,
        39,  26.6667, 6.16667,   -19.5, -12.833, -6.4167,   -19.5, -12.833,   -0.25,       0,       0,       0,
      20.5,  6.16667, 13.8333,    -0.5,   -0.25,   -0.25,     -20, -6.4167,   -0.25,       0,       0,       0,
       -40,    -19.5,    -0.5,      80,      39,    20.5,       0,       0,       0,     -40,   -19.5,     -20,
     -19.5,  -12.833,   -0.25,      39, 26.6667, 6.16667,       0,       0,       0,   -19.5, -12.833, -6.4167,
       -20,  -6.4167,   -0.25,    20.5, 6.16667, 13.8333,       0,       0,       0,    -0.5,   -0.25,   -0.25,
       -40,    -19.5,     -20,       0,       0,       0,      80,      39,    20.5,     -40,   -19.5,    -0.5,
     -19.5,  -12.833, -6.4167,       0,       0,       0,      39, 26.6667, 6.16667,   -19.5, -12.833,   -0.25,
      -0.5,    -0.25,   -0.25,       0,       0,       0,    20.5, 6.16667, 13.8333,     -20, -6.4167,   -0.25,
         0,        0,       0,     -40,   -19.5,    -0.5,     -40,   -19.5,     -20,      80,      39,    20.5,
         0,        0,       0,   -19.5, -12.833,   -0.25,   -19.5, -12.833, -6.4167,      39, 26.6667, 6.16667,
         0,        0,       0,     -20, -6.4167,   -0.25,    -0.5,   -0.25,   -0.25,    20.5, 6.16667, 13.8333;

    // Convert sparse matrix to dense for comparison
    Eigen::MatrixXd computed = Eigen::MatrixXd(system_matrix);
    
    // Compare matrices
    double tolerance = 1e-3;  // Adjust tolerance as needed
    ASSERT_EQ(computed.rows(), expected.rows());
    ASSERT_EQ(computed.cols(), expected.cols());
    
    for (int i = 0; i < computed.rows(); ++i) {
        for (int j = 0; j < computed.cols(); ++j) {
            EXPECT_NEAR(computed(i,j), expected(i,j), tolerance) 
                << "Mismatch at position (" << i << "," << j << ")";
        }
    }

    // Verify matrix properties
    // Check symmetry
    for (int i = 0; i < computed.rows(); ++i) {
        for (int j = 0; j < computed.cols(); ++j) {
            EXPECT_NEAR(computed(i,j), computed(j,i), tolerance)
                << "Matrix not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(AdvectionWeakFormulationTest, Construction) {
    Eigen::Vector2d beta(1.0, 0.0);  // Advection in x-direction
    EXPECT_NO_THROW(AdvectionWeakFormulation weak_form(beta));
    
    Eigen::Vector2d beta2(0.5, 0.5);  // Diagonal advection
    EXPECT_NO_THROW(AdvectionWeakFormulation weak_form2(beta2));
}

TEST(AdvectionWeakFormulationTest, MassIntegral) {
    Eigen::Vector2d beta(1.0, 0.0);
    AdvectionWeakFormulation weak_form(beta);
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);
    auto M = weak_form.compute_mass_integral(elem_data, space);

    // Mass matrix should be symmetric positive definite
    EXPECT_EQ(M.rows(), M.cols());
    EXPECT_EQ(M.rows(), 3);  // 3 basis functions for order 1 triangle
    
    for (int i = 0; i < M.rows(); ++i) {
        EXPECT_GT(M(i,i), 0.0);  // Diagonal entries positive
        for (int j = 0; j < M.cols(); ++j) {
            EXPECT_NEAR(M(i,j), M(j,i), 1e-12);  // Symmetric
        }
    }
    
    // Check that mass matrix has reasonable magnitude (for unit triangle)
    double trace = M.trace();
    EXPECT_GT(trace, 0.0);
    EXPECT_LT(trace, 10.0);  // Should be on order of element area
}

TEST(AdvectionWeakFormulationTest, StiffnessIntegral) {
    Eigen::Vector2d beta(1.0, 0.0);
    AdvectionWeakFormulation weak_form(beta);
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);
    auto L = weak_form.compute_volume_integral(elem_data, space);

    // Stiffness matrix should be square
    EXPECT_EQ(L.rows(), L.cols());
    EXPECT_EQ(L.rows(), 3);
    
    // For advection, the stiffness matrix is generally NOT symmetric
    // (unlike Laplace) due to the directional nature of advection
}

TEST(AdvectionWeakFormulationTest, InteriorFaceIntegralUpwind) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0, 1, 0, 1);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    // Advection in x-direction (to the right)
    Eigen::Vector2d beta(1.0, 0.0);
    AdvectionWeakFormulation weak_form(beta);
    
    // Find an interior face between two elements
    int elem_L = 0;
    const auto& neighbors = mesh->get_element_neighbors(elem_L);
    
    int elem_R = -1;
    int face_L = -1;
    int face_R = -1;
    
    for (int f = 0; f < 4; ++f) {
        if (neighbors[f].first >= 0) {
            elem_R = neighbors[f].first;
            face_L = f;
            face_R = neighbors[f].second;
            break;
        }
    }
    
    ASSERT_GE(elem_R, 0) << "No interior face found";
    
    // Compute face permutation
    int n_verts = mesh->get_elements().cols();
    Eigen::MatrixXd v_L(n_verts, 2);
    Eigen::MatrixXd v_R(n_verts, 2);
    for (int i = 0; i < n_verts; ++i) {
        v_L.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem_L, i));
        v_R.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem_R, i));
    }
    
    Eigen::VectorXi perm = space->compute_face_permutation(face_L, v_L, face_R, v_R);
    
    auto [L_LL, L_LR, L_RL, L_RR] = weak_form.compute_interior_face_integral(
        elem_L, face_L, elem_R, face_R, mesh, perm);
    
    // Check that matrices have correct size
    EXPECT_EQ(L_LL.rows(), 3);
    EXPECT_EQ(L_LL.cols(), 3);
    EXPECT_EQ(L_LR.rows(), 3);
    EXPECT_EQ(L_LR.cols(), 3);
    EXPECT_EQ(L_RL.rows(), 3);
    EXPECT_EQ(L_RL.cols(), 3);
    EXPECT_EQ(L_RR.rows(), 3);
    EXPECT_EQ(L_RR.cols(), 3);
    
    // For upwind flux, at least one of the matrices should be non-zero
    double norm_total = L_LL.norm() + L_LR.norm() + L_RL.norm() + L_RR.norm();
    EXPECT_GT(norm_total, 0.0);
}

TEST(AdvectionWeakFormulationTest, BoundaryFaceIntegralInflow) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    // Advection in x-direction (to the right)
    Eigen::Vector2d beta(1.0, 0.0);
    AdvectionWeakFormulation weak_form(beta);
    
    // Left boundary face (inflow for beta pointing right)
    int elem_id = 0;
    int face_id = 2;  // Left face of element 0
    
    // Boundary condition function (inflow value)
    auto bc_func = [](const Eigen::Vector2d& x) { return 1.0; };
    
    auto [L_bc, F_bc] = weak_form.compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);
    
    // Check dimensions
    EXPECT_EQ(L_bc.rows(), 3);
    EXPECT_EQ(L_bc.cols(), 3);
    EXPECT_EQ(F_bc.size(), 3);
    
    // For inflow, F_bc should be non-zero (contains BC data)
    EXPECT_GT(F_bc.norm(), 0.0);
}

TEST(AdvectionWeakFormulationTest, BoundaryFaceIntegralOutflow) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    // Advection in x-direction (to the right)
    Eigen::Vector2d beta(1.0, 0.0);
    AdvectionWeakFormulation weak_form(beta);
    
    // Right boundary face (outflow for beta pointing right)
    int elem_id = 0;
    int face_id = 1;  // Right face of element 0
    
    auto bc_func = [](const Eigen::Vector2d& x) { return 0.0; };
    
    auto [L_bc, F_bc] = weak_form.compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);
    
    // Check dimensions
    EXPECT_EQ(L_bc.rows(), 3);
    EXPECT_EQ(L_bc.cols(), 3);
    EXPECT_EQ(F_bc.size(), 3);
    
    // For outflow, L_bc should be non-zero, F_bc should be zero (no external BC)
    EXPECT_GT(L_bc.norm(), 0.0);
    EXPECT_NEAR(F_bc.norm(), 0.0, 1e-12);
}

TEST(AdvectionWeakFormulationTest, MassMatrixAssembly) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0, 1, 0, 1);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    Eigen::Vector2d beta(1.0, 0.5);
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(beta);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);
    
    auto M = assembler->assemble_mass_matrix();
    
    // Mass matrix should be symmetric and positive definite
    Eigen::MatrixXd M_dense = Eigen::MatrixXd(M);
    
    // Check symmetry
    for (int i = 0; i < M_dense.rows(); ++i) {
        for (int j = 0; j < M_dense.cols(); ++j) {
            EXPECT_NEAR(M_dense(i,j), M_dense(j,i), 1e-10);
        }
    }
    
    // Check that diagonal entries are positive
    for (int i = 0; i < M_dense.rows(); ++i) {
        EXPECT_GT(M_dense(i,i), 0.0);
    }
}

TEST(AdvectionWeakFormulationTest, AdvectionOperatorAssembly) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, false, 0, 1, 0, 1);  // Use quads instead
    auto space = std::make_shared<DGSpace>("quad", 1);
    mesh->initialize_dg_space(space);
    
    Eigen::Vector2d beta(1.0, 0.0);
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(beta);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);
    
    // Inflow BC function
    auto bc_func = [](const Eigen::Vector2d& x) { return 1.0; };
    
    assembler->assemble(nullptr, bc_func);
    auto L = assembler->get_system_matrix();
    auto F = assembler->get_rhs();

    
    // Operator should have correct size
    EXPECT_EQ(L.rows(), L.cols());
    EXPECT_EQ(F.size(), L.rows());
    
    // L should be non-zero (contains stiffness and face terms)
    EXPECT_GT(L.nonZeros(), 0);
}

TEST(AdvectionWeakFormulationTest, ZeroVelocityProperty) {
    // For zero velocity field, advection operator should give zero residual
    // (no transport means steady state for any initial condition)
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, false, 0, 1, 0, 1);
    auto space = std::make_shared<DGSpace>("quad", 1);
    mesh->initialize_dg_space(space);
    
    Eigen::Vector2d beta(0.0, 0.0);  // Zero velocity
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(beta);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);
    
    auto M = assembler->assemble_mass_matrix();
    auto bc_func = [](const Eigen::Vector2d& x) { return 0.0; };
    assembler->assemble(nullptr, bc_func);
    auto L = assembler->get_system_matrix();
    auto F = assembler->get_rhs();

    
    // For zero velocity, L should be essentially zero (only numerical errors)
    // and F should also be zero (no boundary forcing with zero BC)
    EXPECT_LT(L.norm(), 1e-10);
    EXPECT_LT(F.norm(), 1e-10);
}

// TEST(EulerWeakFormulationTest, Construction) {
//     EXPECT_NO_THROW(EulerWeakFormulation weak_form(1.4));
// }

// TEST(EulerWeakFormulationTest, ConservedPrimitiveConversion) {
//     // Test conversion between conserved and primitive variables
//     Eigen::Vector4d U;
//     U << 1.0, 2.0, 3.0, 10.0;  // rho, rhou, rhov, E
    
//     auto W = conserved_to_primitive(U);
//     auto U_back = primitive_to_conserved(W);
    
//     for (int i = 0; i < 4; ++i) {
//         EXPECT_NEAR(U(i), U_back(i), 1e-12);
//     }
// }

// TEST(EulerWeakFormulationTest, RusanovFlux) {
//     EulerWeakFormulation weak_form(1.4);
    
//     // Test states
//     Eigen::Vector4d U_L, U_R;
//     U_L << 1.0, 0.0, 0.0, 2.5;  // rho=1, u=v=0, p=1
//     U_R << 0.125, 0.0, 0.0, 0.25;  // rho=0.125, u=v=0, p=0.1
    
//     Eigen::Vector2d n(1.0, 0.0);  // x-direction normal
    
//     auto flux = weak_form.rusanov_flux(U_L, U_R, n);
    
//     EXPECT_EQ(flux.size(), 4);
//     // Check conservation
//     auto neg_flux = weak_form.rusanov_flux(U_R, U_L, -n);
//     for (int i = 0; i < 4; ++i) {
//         EXPECT_NEAR(flux(i), -neg_flux(i), 1e-12);
//     }
// }
