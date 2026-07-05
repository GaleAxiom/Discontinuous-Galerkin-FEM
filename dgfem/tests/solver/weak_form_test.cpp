#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/assembler.hpp>
#include <dgfem/solver/weak_form.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <gmock/gmock.h>
#include <gmsh.h>
#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace dgfem;
using namespace testing;

namespace {
double norm_of(const DView1& v) {
    return norm(v);
}
double norm_of(const DView2& m) {
    return frobenius_norm(m);
}
}  // namespace

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
    EXPECT_EQ(K_vol.extent(0), K_vol.extent(1));
    for (int i = 0; i < K_vol.extent(0); ++i) {
        for (int j = 0; j < K_vol.extent(1); ++j) {
            EXPECT_NEAR(K_vol(i, j), K_vol(j, i), 1e-12);
        }
    }
}

TEST(LaplaceWeakFormulationTest, InteriorFaceIntegral) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0, 1, 0, 1);
    LaplaceWeakFormulation weak_form(10.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto assembler =
        std::make_shared<DGAssembler>(mesh, std::make_shared<LaplaceWeakFormulation>(10.0));
    assembler->assemble([](const Vec2&) { return 0.0; });
    auto system_matrix = assembler->get_system_matrix();

    // Create expected matrix
    double expected_flat[144] = {
        80,      39,      20.5,    -40,     -19.5,   -20,     -40,     -19.5,   -0.5,    0,
        0,       0,       39,      26.6667, 6.16667, -19.5,   -12.833, -6.4167, -19.5,   -12.833,
        -0.25,   0,       0,       0,       20.5,    6.16667, 13.8333, -0.5,    -0.25,   -0.25,
        -20,     -6.4167, -0.25,   0,       0,       0,       -40,     -19.5,   -0.5,    80,
        39,      20.5,    0,       0,       0,       -40,     -19.5,   -20,     -19.5,   -12.833,
        -0.25,   39,      26.6667, 6.16667, 0,       0,       0,       -19.5,   -12.833, -6.4167,
        -20,     -6.4167, -0.25,   20.5,    6.16667, 13.8333, 0,       0,       0,       -0.5,
        -0.25,   -0.25,   -40,     -19.5,   -20,     0,       0,       0,       80,      39,
        20.5,    -40,     -19.5,   -0.5,    -19.5,   -12.833, -6.4167, 0,       0,       0,
        39,      26.6667, 6.16667, -19.5,   -12.833, -0.25,   -0.5,    -0.25,   -0.25,   0,
        0,       0,       20.5,    6.16667, 13.8333, -20,     -6.4167, -0.25,   0,       0,
        0,       -40,     -19.5,   -0.5,    -40,     -19.5,   -20,     80,      39,      20.5,
        0,       0,       0,       -19.5,   -12.833, -0.25,   -19.5,   -12.833, -6.4167, 39,
        26.6667, 6.16667, 0,       0,       0,       -20,     -6.4167, -0.25,   -0.5,    -0.25,
        -0.25,   20.5,    6.16667, 13.8333};
    DView2 expected("expected", 12, 12);
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            expected(i, j) = expected_flat[i * 12 + j];
        }
    }

    // Convert sparse matrix to dense for comparison
    DView2 computed = tpetra_to_dense(*system_matrix);

    // Compare matrices
    double tolerance = 1e-3;  // Adjust tolerance as needed
    ASSERT_EQ(computed.extent(0), expected.extent(0));
    ASSERT_EQ(computed.extent(1), expected.extent(1));

    for (int i = 0; i < computed.extent(0); ++i) {
        for (int j = 0; j < computed.extent(1); ++j) {
            EXPECT_NEAR(computed(i, j), expected(i, j), tolerance)
                << "Mismatch at position (" << i << "," << j << ")";
        }
    }

    // Verify matrix properties
    // Check symmetry
    for (int i = 0; i < computed.extent(0); ++i) {
        for (int j = 0; j < computed.extent(1); ++j) {
            EXPECT_NEAR(computed(i, j), computed(j, i), tolerance)
                << "Matrix not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(AdvectionWeakFormulationTest, Construction) {
    Vec2 beta{1.0, 0.0};  // Advection in x-direction
    EXPECT_NO_THROW(AdvectionWeakFormulation weak_form(beta));

    Vec2 beta2{0.5, 0.5};  // Diagonal advection
    EXPECT_NO_THROW(AdvectionWeakFormulation weak_form2(beta2));
}

TEST(AdvectionWeakFormulationTest, MassIntegral) {
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);
    auto M = weak_form.compute_mass_integral(elem_data, space);

    // Mass matrix should be symmetric positive definite
    EXPECT_EQ(M.extent(0), M.extent(1));
    EXPECT_EQ(M.extent(0), 3);  // 3 basis functions for order 1 triangle

    for (int i = 0; i < M.extent(0); ++i) {
        EXPECT_GT(M(i, i), 0.0);  // Diagonal entries positive
        for (int j = 0; j < M.extent(1); ++j) {
            EXPECT_NEAR(M(i, j), M(j, i), 1e-12);  // Symmetric
        }
    }

    // Check that mass matrix has reasonable magnitude (for unit triangle)
    double tr = trace(M);
    EXPECT_GT(tr, 0.0);
    EXPECT_LT(tr, 10.0);  // Should be on order of element area
}

TEST(AdvectionWeakFormulationTest, StiffnessIntegral) {
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);
    auto L = weak_form.compute_volume_integral(elem_data, space);

    // Stiffness matrix should be square
    EXPECT_EQ(L.extent(0), L.extent(1));
    EXPECT_EQ(L.extent(0), 3);

    // For advection, the stiffness matrix is generally NOT symmetric
    // (unlike Laplace) due to the directional nature of advection
}

TEST(AdvectionWeakFormulationTest, InteriorFaceIntegralUpwind) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0, 1, 0, 1);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Advection in x-direction (to the right)
    Vec2 beta{1.0, 0.0};
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
    DView2 v_L = mesh->get_element_vertices(elem_L);
    DView2 v_R = mesh->get_element_vertices(elem_R);

    IView1 perm = space->compute_face_permutation(face_L, v_L, face_R, v_R);

    auto [L_LL, L_LR, L_RL, L_RR] =
        weak_form.compute_interior_face_integral(elem_L, face_L, elem_R, face_R, mesh, perm);

    // Check that matrices have correct size
    EXPECT_EQ(L_LL.extent(0), 3);
    EXPECT_EQ(L_LL.extent(1), 3);
    EXPECT_EQ(L_LR.extent(0), 3);
    EXPECT_EQ(L_LR.extent(1), 3);
    EXPECT_EQ(L_RL.extent(0), 3);
    EXPECT_EQ(L_RL.extent(1), 3);
    EXPECT_EQ(L_RR.extent(0), 3);
    EXPECT_EQ(L_RR.extent(1), 3);

    // For upwind flux, at least one of the matrices should be non-zero
    double norm_total = norm_of(L_LL) + norm_of(L_LR) + norm_of(L_RL) + norm_of(L_RR);
    EXPECT_GT(norm_total, 0.0);
}

TEST(AdvectionWeakFormulationTest, BoundaryFaceIntegralInflow) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Advection in x-direction (to the right)
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);

    // Left boundary face (inflow for beta pointing right)
    int elem_id = 0;
    int face_id = 2;  // Left face of element 0

    // Boundary condition function (inflow value)
    auto bc_func = [](const Vec2& x) { return 1.0; };

    auto [L_bc, F_bc] = weak_form.compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);

    // Check dimensions
    EXPECT_EQ(L_bc.extent(0), 3);
    EXPECT_EQ(L_bc.extent(1), 3);
    EXPECT_EQ(F_bc.size(), 3);

    // For inflow, F_bc should be non-zero (contains BC data)
    EXPECT_GT(norm_of(F_bc), 0.0);
}

TEST(AdvectionWeakFormulationTest, BoundaryFaceIntegralOutflow) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Advection in x-direction (to the right)
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);

    // Right boundary face (outflow for beta pointing right)
    int elem_id = 0;
    int face_id = 1;  // Right face of element 0

    auto bc_func = [](const Vec2& x) { return 0.0; };

    auto [L_bc, F_bc] = weak_form.compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);

    // Check dimensions
    EXPECT_EQ(L_bc.extent(0), 3);
    EXPECT_EQ(L_bc.extent(1), 3);
    EXPECT_EQ(F_bc.size(), 3);

    // For outflow, L_bc should be non-zero, F_bc should be zero (no external BC)
    EXPECT_GT(norm_of(L_bc), 0.0);
    EXPECT_NEAR(norm_of(F_bc), 0.0, 1e-12);
}

TEST(AdvectionWeakFormulationTest, MassMatrixAssembly) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0, 1, 0, 1);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    Vec2 beta{1.0, 0.5};
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(beta);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    auto M = assembler->assemble_mass_matrix();

    // Mass matrix should be symmetric and positive definite
    DView2 M_dense = tpetra_to_dense(*M);

    // Check symmetry
    for (int i = 0; i < M_dense.extent(0); ++i) {
        for (int j = 0; j < M_dense.extent(1); ++j) {
            EXPECT_NEAR(M_dense(i, j), M_dense(j, i), 1e-10);
        }
    }

    // Check that diagonal entries are positive
    for (int i = 0; i < M_dense.extent(0); ++i) {
        EXPECT_GT(M_dense(i, i), 0.0);
    }
}

TEST(AdvectionWeakFormulationTest, AdvectionOperatorAssembly) {
    gmsh::initialize();
    auto mesh =
        dgfem::MeshCreator::create_rectangular_mesh(1, false, 0, 1, 0, 1);  // Use quads instead
    auto space = std::make_shared<DGSpace>("quad", 1);
    mesh->initialize_dg_space(space);

    Vec2 beta{1.0, 0.0};
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(beta);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    // Inflow BC function
    auto bc_func = [](const Vec2& x) { return 1.0; };

    assembler->assemble(nullptr, bc_func);
    auto L = assembler->get_system_matrix();
    auto F = assembler->get_rhs();

    // Operator should have correct size
    EXPECT_EQ(L->getGlobalNumRows(), L->getGlobalNumCols());
    EXPECT_EQ(F->getGlobalLength(), L->getGlobalNumRows());

    // L should be non-zero (contains stiffness and face terms)
    EXPECT_GT(L->getGlobalNumEntries(), 0u);
}

TEST(AdvectionWeakFormulationTest, ZeroVelocityProperty) {
    // For zero velocity field, advection operator should give zero residual
    // (no transport means steady state for any initial condition)
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, false, 0, 1, 0, 1);
    auto space = std::make_shared<DGSpace>("quad", 1);
    mesh->initialize_dg_space(space);

    Vec2 beta{0.0, 0.0};  // Zero velocity
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(beta);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    auto M = assembler->assemble_mass_matrix();
    auto bc_func = [](const Vec2& x) { return 0.0; };
    assembler->assemble(nullptr, bc_func);
    auto L = assembler->get_system_matrix();
    auto F = assembler->get_rhs();

    // For zero velocity, L should be essentially zero (only numerical errors)
    // and F should also be zero (no boundary forcing with zero BC)
    EXPECT_LT(frobenius_norm(tpetra_to_dense(*L)), 1e-10);
    EXPECT_LT(norm(tpetra_to_view(*F)), 1e-10);
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

//     Vec2 n{1.0, 0.0};  // x-direction normal

//     auto flux = weak_form.rusanov_flux(U_L, U_R, n);

//     EXPECT_EQ(flux.size(), 4);
//     // Check conservation
//     auto neg_flux = weak_form.rusanov_flux(U_R, U_L, -n);
//     for (int i = 0; i < 4; ++i) {
//         EXPECT_NEAR(flux(i), -neg_flux(i), 1e-12);
//     }
// }
