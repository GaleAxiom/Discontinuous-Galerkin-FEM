#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/assembler.hpp>
#include <dgfem/solver/weak_form.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <stdexcept>

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

TEST(LaplaceWeakFormulationTest, RejectsNonPositivePenaltyParameter) {
    EXPECT_THROW(LaplaceWeakFormulation(0.0), std::invalid_argument);
    EXPECT_THROW(LaplaceWeakFormulation(-10.0), std::invalid_argument);
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
    // Explicit "monomial" override: the golden matrix below is hand/derivation-tied to
    // MonomialBasisTriangle's specific representation, independent of whichever basis
    // DGSpace::create_basis picks by default for triangles.
    auto space = std::make_shared<DGSpace>("triangle", 1, "monomial");
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

TEST(LaplaceWeakFormulationTest, InteriorFaceIntegralDubinerBasis) {
    gmsh::initialize();
    auto mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0, 1, 0, 1);
    // Same setup as InteriorFaceIntegral above, but forcing the orthogonal DubinerBasis
    // instead of MonomialBasisTriangle, so both bases get real regression coverage of this
    // assembled system matrix.
    auto space = std::make_shared<DGSpace>("triangle", 1, "dubiner");
    mesh->initialize_dg_space(space);

    auto assembler =
        std::make_shared<DGAssembler>(mesh, std::make_shared<LaplaceWeakFormulation>(10.0));
    assembler->assemble([](const Vec2&) { return 0.0; });
    auto system_matrix = assembler->get_system_matrix();

    // Golden matrix captured from running this exact setup with DubinerBasis (not
    // hand-derived like the MonomialBasisTriangle case above -- Dubiner's Jacobi-polynomial
    // closed form makes hand derivation impractical for a full 4-element assembled system).
    // Sanity-checked below: symmetric (required for this self-adjoint SIPG formulation,
    // regardless of basis).
    DView2 computed = tpetra_to_dense(*system_matrix);

    double expected_flat[144] = {320,
                                 0,
                                 209.3036072,
                                 -160,
                                 -191.0601999,
                                 -104.6518036,
                                 -160,
                                 191.0601999,
                                 -104.6518036,
                                 0,
                                 0,
                                 0,
                                 0,
                                 640,
                                 0,
                                 191.0601999,
                                 308,
                                 -13.85640646,
                                 -191.0601999,
                                 308,
                                 13.85640646,
                                 0,
                                 0,
                                 0,
                                 209.3036072,
                                 0,
                                 688,
                                 -104.6518036,
                                 13.85640646,
                                 -308,
                                 -104.6518036,
                                 -13.85640646,
                                 -308,
                                 0,
                                 0,
                                 0,
                                 -160,
                                 191.0601999,
                                 -104.6518036,
                                 320,
                                 0,
                                 209.3036072,
                                 0,
                                 0,
                                 0,
                                 -160,
                                 -191.0601999,
                                 -104.6518036,
                                 -191.0601999,
                                 308,
                                 13.85640646,
                                 0,
                                 640,
                                 0,
                                 0,
                                 0,
                                 0,
                                 191.0601999,
                                 308,
                                 -13.85640646,
                                 -104.6518036,
                                 -13.85640646,
                                 -308,
                                 209.3036072,
                                 0,
                                 688,
                                 0,
                                 0,
                                 0,
                                 -104.6518036,
                                 13.85640646,
                                 -308,
                                 -160,
                                 -191.0601999,
                                 -104.6518036,
                                 0,
                                 0,
                                 0,
                                 320,
                                 0,
                                 209.3036072,
                                 -160,
                                 191.0601999,
                                 -104.6518036,
                                 191.0601999,
                                 308,
                                 -13.85640646,
                                 0,
                                 0,
                                 0,
                                 0,
                                 640,
                                 0,
                                 -191.0601999,
                                 308,
                                 13.85640646,
                                 -104.6518036,
                                 13.85640646,
                                 -308,
                                 0,
                                 0,
                                 0,
                                 209.3036072,
                                 0,
                                 688,
                                 -104.6518036,
                                 -13.85640646,
                                 -308,
                                 0,
                                 0,
                                 0,
                                 -160,
                                 191.0601999,
                                 -104.6518036,
                                 -160,
                                 -191.0601999,
                                 -104.6518036,
                                 320,
                                 0,
                                 209.3036072,
                                 0,
                                 0,
                                 0,
                                 -191.0601999,
                                 308,
                                 13.85640646,
                                 191.0601999,
                                 308,
                                 -13.85640646,
                                 0,
                                 640,
                                 0,
                                 0,
                                 0,
                                 0,
                                 -104.6518036,
                                 -13.85640646,
                                 -308,
                                 -104.6518036,
                                 13.85640646,
                                 -308,
                                 209.3036072,
                                 0,
                                 688};
    DView2 expected("expected", 12, 12);
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            expected(i, j) = expected_flat[i * 12 + j];
        }
    }

    double tolerance = 1e-3;
    ASSERT_EQ(computed.extent(0), expected.extent(0));
    ASSERT_EQ(computed.extent(1), expected.extent(1));

    for (int i = 0; i < computed.extent(0); ++i) {
        for (int j = 0; j < computed.extent(1); ++j) {
            EXPECT_NEAR(computed(i, j), expected(i, j), tolerance)
                << "Mismatch at position (" << i << "," << j << ")";
        }
    }

    for (int i = 0; i < computed.extent(0); ++i) {
        for (int j = 0; j < computed.extent(1); ++j) {
            EXPECT_NEAR(computed(i, j), computed(j, i), tolerance)
                << "Matrix not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(LaplaceWeakFormulationTest, CornerElementIndependentBoundaryFaces) {
    // Element 0 of create_test_mesh() has two boundary faces meeting at the corner (1,0):
    // face 0 ("bottom") and face 1 ("right"). DG has no shared DOFs between faces -- each
    // element's coefficients are independent -- so this checks that the boundary RHS
    // integral genuinely depends only on the BC passed for that specific face, is linear in
    // the Dirichlet value (a real property of the SIPG boundary terms), and that two
    // differently-valued adjacent faces of the same corner element don't contaminate each
    // other's contribution.
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    LaplaceWeakFormulation weak_form(10.0);

    DView1 F_bottom = weak_form.compute_boundary_rhs_integral(0, 0, mesh, make_dirichlet_bc(2.0));
    DView1 F_right = weak_form.compute_boundary_rhs_integral(0, 1, mesh, make_dirichlet_bc(5.0));

    ASSERT_GT(norm(F_bottom), 1e-12);
    ASSERT_GT(norm(F_right), 1e-12);

    // Linearity in the Dirichlet value: doubling g should double the RHS contribution.
    DView1 F_bottom_doubled =
        weak_form.compute_boundary_rhs_integral(0, 0, mesh, make_dirichlet_bc(4.0));
    for (int i = 0; i < static_cast<int>(F_bottom.extent(0)); ++i) {
        EXPECT_NEAR(F_bottom_doubled(i), 2.0 * F_bottom(i), 1e-10);
    }

    // The two adjacent faces' contributions are independent, not a shared/averaged value.
    bool any_different = false;
    for (int i = 0; i < static_cast<int>(F_bottom.extent(0)); ++i) {
        if (std::abs(F_bottom(i) - F_right(i)) > 1e-10) {
            any_different = true;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(LaplaceWeakFormulationTest, NeumannBoundaryFaceIntegralHasNoMatrixContribution) {
    // Neumann is a natural BC: du/dn = g enters only through the RHS
    // (compute_boundary_rhs_integral). Unlike Dirichlet's SIPG penalty/consistency terms, the
    // matrix contribution must be exactly zero regardless of the flux value or which face.
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    LaplaceWeakFormulation weak_form(10.0);

    DView2 K_bottom = weak_form.compute_boundary_face_integral(0, 0, mesh, make_neumann_bc(7.0));
    DView2 K_right = weak_form.compute_boundary_face_integral(0, 1, mesh, make_neumann_bc(-3.0));

    EXPECT_NEAR(frobenius_norm(K_bottom), 0.0, 1e-14);
    EXPECT_NEAR(frobenius_norm(K_right), 0.0, 1e-14);
}

TEST(LaplaceWeakFormulationTest, NeumannBoundaryRhsIntegralIsLinearAndFaceSpecific) {
    // Mirrors CornerElementIndependentBoundaryFaces above, but for Neumann flux data: the RHS
    // contribution should be nonzero, linear in the flux value (it's a plain integral(g*v)
    // term), and genuinely per-face rather than shared/averaged between the two boundary faces
    // meeting at the corner.
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    LaplaceWeakFormulation weak_form(10.0);

    DView1 F_bottom = weak_form.compute_boundary_rhs_integral(0, 0, mesh, make_neumann_bc(2.0));
    DView1 F_right = weak_form.compute_boundary_rhs_integral(0, 1, mesh, make_neumann_bc(5.0));

    ASSERT_GT(norm(F_bottom), 1e-12);
    ASSERT_GT(norm(F_right), 1e-12);

    DView1 F_bottom_doubled =
        weak_form.compute_boundary_rhs_integral(0, 0, mesh, make_neumann_bc(4.0));
    for (int i = 0; i < static_cast<int>(F_bottom.extent(0)); ++i) {
        EXPECT_NEAR(F_bottom_doubled(i), 2.0 * F_bottom(i), 1e-10);
    }

    bool any_different = false;
    for (int i = 0; i < static_cast<int>(F_bottom.extent(0)); ++i) {
        if (std::abs(F_bottom(i) - F_right(i)) > 1e-10) {
            any_different = true;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(LaplaceWeakFormulationTest, RobinBoundaryFaceIntegralScalesLinearlyWithAlpha) {
    // Robin's matrix contribution is a pure alpha-scaled mass term (+alpha*integral(u*v) ds),
    // so it must be symmetric, scale linearly with alpha, and vanish when alpha == 0
    // (recovering Neumann's no-matrix-contribution property exactly).
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    LaplaceWeakFormulation weak_form(10.0);

    DView2 K_alpha1 = weak_form.compute_boundary_face_integral(0, 0, mesh, make_robin_bc(1.0, 1.0));
    DView2 K_alpha3 = weak_form.compute_boundary_face_integral(0, 0, mesh, make_robin_bc(1.0, 3.0));
    DView2 K_alpha0 = weak_form.compute_boundary_face_integral(0, 0, mesh, make_robin_bc(1.0, 0.0));

    ASSERT_GT(frobenius_norm(K_alpha1), 1e-12);
    EXPECT_NEAR(frobenius_norm(K_alpha0), 0.0, 1e-14);

    for (int i = 0; i < static_cast<int>(K_alpha1.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(K_alpha1.extent(1)); ++j) {
            EXPECT_NEAR(K_alpha1(i, j), K_alpha1(j, i), 1e-12) << "Robin matrix must be symmetric";
            EXPECT_NEAR(K_alpha3(i, j), 3.0 * K_alpha1(i, j), 1e-10)
                << "Robin matrix must scale linearly with alpha";
        }
    }
}

TEST(LaplaceWeakFormulationTest, RobinAndNeumannRhsIntegralsMatchForSameFluxData) {
    // The RHS contribution is exactly integral(g*v) ds for both Neumann and Robin -- alpha only
    // ever enters the matrix (compute_boundary_face_integral), never the RHS. So a Robin BC and
    // a Neumann BC sharing the same (position-dependent) flux datum g must produce identical F
    // vectors, no matter what alpha is.
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    LaplaceWeakFormulation weak_form(10.0);

    auto g = [](const Vec2& x) { return 1.0 + 2.0 * x[0] - 3.0 * x[1]; };

    DView1 F_neumann = weak_form.compute_boundary_rhs_integral(0, 0, mesh, make_neumann_bc(g));
    DView1 F_robin_small_alpha =
        weak_form.compute_boundary_rhs_integral(0, 0, mesh, make_robin_bc(g, 0.5));
    DView1 F_robin_large_alpha =
        weak_form.compute_boundary_rhs_integral(0, 0, mesh, make_robin_bc(g, 42.0));

    ASSERT_GT(norm(F_neumann), 1e-12);
    for (int i = 0; i < static_cast<int>(F_neumann.extent(0)); ++i) {
        EXPECT_NEAR(F_neumann(i), F_robin_small_alpha(i), 1e-10);
        EXPECT_NEAR(F_neumann(i), F_robin_large_alpha(i), 1e-10);
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
    // Explicit "monomial" override: the analytic derivation below is tied to
    // MonomialBasisTriangle's specific representation, independent of whichever basis
    // DGSpace::create_basis picks by default for triangles.
    auto space = std::make_shared<DGSpace>("triangle", 1, "monomial");
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

    // Exact analytic check, not just a loose magnitude bound: MonomialBasisTriangle order 1
    // evaluates {1, eta, xi} directly in reference coordinates (see monomial.cpp's basis
    // ordering), and element 0 of create_test_mesh() -- vertices (0,0),(1,0),(1,1) -- maps to
    // the reference triangle via x=xi+eta, y=eta, with constant |J|=1. So
    // M_ij = |J| * integral_ref(phi_i * phi_j), using the standard reference-triangle moments
    // integral(1)=1/2, integral(xi)=integral(eta)=1/6, integral(xi^2)=integral(eta^2)=1/12,
    // integral(xi*eta)=1/24.
    EXPECT_NEAR(M(0, 0), 1.0 / 2.0, 1e-12);
    EXPECT_NEAR(M(0, 1), 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(M(0, 2), 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(M(1, 1), 1.0 / 12.0, 1e-12);
    EXPECT_NEAR(M(2, 2), 1.0 / 12.0, 1e-12);
    EXPECT_NEAR(M(1, 2), 1.0 / 24.0, 1e-12);
}

TEST(AdvectionWeakFormulationTest, MassIntegralDubinerBasis) {
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1, "dubiner");
    mesh->initialize_dg_space(space);

    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);
    auto M = weak_form.compute_mass_integral(elem_data, space);

    EXPECT_EQ(M.extent(0), M.extent(1));
    EXPECT_EQ(M.extent(0), 3);

    for (int i = 0; i < M.extent(0); ++i) {
        EXPECT_GT(M(i, i), 0.0);
        for (int j = 0; j < M.extent(1); ++j) {
            EXPECT_NEAR(M(i, j), M(j, i), 1e-12);
        }
    }

    // Exact analytic check for the order-1 DubinerBasis, derived by hand from
    // dgfem/src/basis/dubiner.cpp's evaluate_impl and cross-checked numerically against the
    // actual implementation: on the reference triangle (0,0),(1,0),(0,1), the order-1 modes
    // (indexed per get_dubiner_indices as idx0=(i=0,j=0), idx1=(i=1,j=0), idx2=(i=0,j=1)) are
    // phi_0 = 2 (constant), phi_1 = sqrt(6)*(4*xi + 2*eta - 2), phi_2 = 2*sqrt(2)*(3*eta - 1).
    // Element 0 of create_test_mesh() maps via x=xi+eta, y=eta with constant |J|=1 (same
    // element/mapping as MassIntegral above). Dubiner is orthogonal on the reference triangle
    // (area 1/2) with ||phi_i||^2 = 2 for every mode (i.e. integral_ref(phi_i*phi_j) = 2 *
    // delta_ij -- note this is twice the naive "unit-normalized" value because phi_0 itself is
    // the constant 2, not 1: integral_ref(phi_0^2) = 4 * (1/2) = 2), so M = |J| * 2 * I = 2*I.
    EXPECT_NEAR(M(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(M(1, 1), 2.0, 1e-10);
    EXPECT_NEAR(M(2, 2), 2.0, 1e-10);
    EXPECT_NEAR(M(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(M(0, 2), 0.0, 1e-10);
    EXPECT_NEAR(M(1, 2), 0.0, 1e-10);
}

TEST(AdvectionWeakFormulationTest, StiffnessIntegral) {
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);
    auto mesh = create_test_mesh();
    // Explicit "monomial" override -- see MassIntegral above.
    auto space = std::make_shared<DGSpace>("triangle", 1, "monomial");
    mesh->initialize_dg_space(space);

    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);
    auto L = weak_form.compute_volume_integral(elem_data, space);

    // Stiffness matrix should be square
    EXPECT_EQ(L.extent(0), L.extent(1));
    EXPECT_EQ(L.extent(0), 3);

    // For advection, the stiffness matrix is generally NOT symmetric (unlike Laplace) due to
    // the directional nature of advection -- exact analytic check, not just a shape check.
    // With phi = {1, eta, xi} (reference monomial basis, see MassIntegral above) and the
    // physical-to-reference map for this element (x=xi+eta, y=eta, constant |J|=1), physical
    // gradients are grad(phi_0)=(0,0), grad(phi_1)=(0,1), grad(phi_2)=(1,-1), so with
    // beta=(1,0), beta.grad(phi) = [0, 0, 1] -- constant per basis function since the basis is
    // linear. L(i,j) = integral(beta.grad(phi_i) * phi_j) = beta.grad(phi_i) * integral(phi_j),
    // so only row i=2 (the only nonzero beta.grad entry) is nonzero, and it equals the phi_j
    // volume-integral row from MassIntegral's first row: [1/2, 1/6, 1/6].
    for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(L(0, j), 0.0, 1e-12);
        EXPECT_NEAR(L(1, j), 0.0, 1e-12);
    }
    EXPECT_NEAR(L(2, 0), 1.0 / 2.0, 1e-12);
    EXPECT_NEAR(L(2, 1), 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(L(2, 2), 1.0 / 6.0, 1e-12);
}

TEST(AdvectionWeakFormulationTest, StiffnessIntegralDubinerBasis) {
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1, "dubiner");
    mesh->initialize_dg_space(space);

    int elem_id = 0;
    auto elem_data = mesh->get_element_data(elem_id);
    auto L = weak_form.compute_volume_integral(elem_data, space);

    EXPECT_EQ(L.extent(0), L.extent(1));
    EXPECT_EQ(L.extent(0), 3);

    // Exact analytic check for order-1 DubinerBasis, using the phi_0/phi_1/phi_2 closed forms
    // from MassIntegralDubinerBasis above. Reference gradients are constant (basis is affine):
    // grad_ref(phi_0)=(0,0), grad_ref(phi_1)=sqrt(6)*(4,2), grad_ref(phi_2)=2*sqrt(2)*(0,3).
    // Pushing forward through this element's map (x=xi+eta, y=eta => xi=x-y, eta=y, so
    // grad_phys(f) = (df/dxi, df/deta - df/dxi), same map as MassIntegral/StiffnessIntegral
    // above) gives grad_phys(phi_0)=(0,0), grad_phys(phi_1)=(4*sqrt(6), -2*sqrt(6)),
    // grad_phys(phi_2)=(0, 6*sqrt(2)). With beta=(1,0): beta.grad(phi_0)=0,
    // beta.grad(phi_1)=4*sqrt(6), beta.grad(phi_2)=0 -- constant per basis function.
    // L(i,j) = beta.grad(phi_i) * integral_phys(phi_j); since phi_1, phi_2 are mean-zero
    // (orthogonal to the constant phi_0 mode) and integral_phys(phi_0) = |J| * 2 * (1/2) = 1
    // (phi_0=2 over reference area 1/2), only column 0 and only row 1 (the sole nonzero
    // beta.grad entry) are nonzero: L(1,0) = 4*sqrt(6), everything else 0.
    double four_sqrt6 = 4.0 * std::sqrt(6.0);
    for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(L(0, j), 0.0, 1e-10);
        EXPECT_NEAR(L(2, j), 0.0, 1e-10);
    }
    EXPECT_NEAR(L(1, 0), four_sqrt6, 1e-10);
    EXPECT_NEAR(L(1, 1), 0.0, 1e-10);
    EXPECT_NEAR(L(1, 2), 0.0, 1e-10);
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
    // Explicit "monomial" override -- see MassIntegral above.
    auto space = std::make_shared<DGSpace>("triangle", 1, "monomial");
    mesh->initialize_dg_space(space);

    // Advection in x-direction (to the right)
    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);

    // Face 2 of element 0 (edge_vertices(2) = {2,0}) is the diagonal from (1,1) to (0,0), not
    // literally the geometric "left" edge of the unit square -- but its outward normal is
    // (-1,1)/sqrt(2), so beta.n = -1/sqrt(2) < 0 (inflow for beta=(1,0)), which is what this
    // test actually exercises.
    int elem_id = 0;
    int face_id = 2;

    // Boundary condition function (inflow value)
    auto bc_func = [](const Vec2& x) { return 1.0; };

    auto [L_bc, F_bc] = weak_form.compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);

    // Check dimensions
    EXPECT_EQ(L_bc.extent(0), 3);
    EXPECT_EQ(L_bc.extent(1), 3);
    EXPECT_EQ(F_bc.size(), 3);

    // Exact analytic check: F_bc[i] = -beta_n * integral_face(phi_i) ds, with phi = {1, eta, xi}
    // (see MassIntegral above) parametrized along this face as (xi,eta)=(0,1-t), t in [0,1],
    // physical arc length ds = sqrt(2) dt. integral(phi_0)=sqrt(2), integral(phi_1)=sqrt(2)/2,
    // integral(phi_2)=0 (xi=0 identically on this face), and -beta_n = 1/sqrt(2).
    EXPECT_NEAR(F_bc[0], 1.0, 1e-12);
    EXPECT_NEAR(F_bc[1], 0.5, 1e-12);
    EXPECT_NEAR(F_bc[2], 0.0, 1e-12);
}

TEST(AdvectionWeakFormulationTest, BoundaryFaceIntegralInflowDubinerBasis) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1, "dubiner");
    mesh->initialize_dg_space(space);

    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);

    int elem_id = 0;
    int face_id = 2;  // Same face as BoundaryFaceIntegralInflow above.

    auto bc_func = [](const Vec2& x) { return 1.0; };
    auto [L_bc, F_bc] = weak_form.compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);

    EXPECT_EQ(L_bc.extent(0), 3);
    EXPECT_EQ(L_bc.extent(1), 3);
    EXPECT_EQ(F_bc.size(), 3);

    // Exact analytic check for order-1 DubinerBasis, using the phi_0/phi_1/phi_2 closed forms
    // from MassIntegralDubinerBasis above. Face 2 runs from ref (0,1) to ref (0,0) (xi=0
    // throughout), parametrized as (xi,eta)=(0,1-t), t in [0,1], physical arc length ds =
    // sqrt(2) dt (same face/parametrization as the monomial case above). Along this face:
    // phi_0=2, phi_1=2*sqrt(6)*(eta-1)=-2*sqrt(6)*t, phi_2=2*sqrt(2)*(3*eta-1)=2*sqrt(2)*(2-3t).
    // integral_face(phi_0) = 2*sqrt(2), integral_face(phi_1) = -2*sqrt(3),
    // integral_face(phi_2) = 2, and -beta_n = 1/sqrt(2) (same geometry as monomial case), so
    // F_bc[i] = (1/sqrt(2)) * integral_face(phi_i): F_bc[0]=2, F_bc[1]=-sqrt(6), F_bc[2]=sqrt(2).
    EXPECT_NEAR(F_bc[0], 2.0, 1e-10);
    EXPECT_NEAR(F_bc[1], -std::sqrt(6.0), 1e-10);
    EXPECT_NEAR(F_bc[2], std::sqrt(2.0), 1e-10);
}

TEST(AdvectionWeakFormulationTest, BoundaryFaceIntegralOutflow) {
    auto mesh = create_test_mesh();
    // Explicit "monomial" override -- see MassIntegral above.
    auto space = std::make_shared<DGSpace>("triangle", 1, "monomial");
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

    // F_bc should be zero (no external BC needed for outflow)
    EXPECT_NEAR(norm_of(F_bc), 0.0, 1e-12);

    // Exact analytic check for L_bc: face 1 (edge_vertices(1)={1,2}) is the real right edge
    // x=1 for this element, with beta_n=+1 (outflow) and h_F=1. L_bc(i,j) = -integral_0^1
    // phi_i(t)*phi_j(t) dt where, parametrizing this face as (xi,eta)=(1-t,t), phi_0=1,
    // phi_1=t, phi_2=1-t and physical arc length ds=dt exactly.
    EXPECT_NEAR(L_bc(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(L_bc(0, 1), -0.5, 1e-12);
    EXPECT_NEAR(L_bc(0, 2), -0.5, 1e-12);
    EXPECT_NEAR(L_bc(1, 1), -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(L_bc(1, 2), -1.0 / 6.0, 1e-12);
    EXPECT_NEAR(L_bc(2, 2), -1.0 / 3.0, 1e-12);
}

TEST(AdvectionWeakFormulationTest, BoundaryFaceIntegralOutflowDubinerBasis) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>("triangle", 1, "dubiner");
    mesh->initialize_dg_space(space);

    Vec2 beta{1.0, 0.0};
    AdvectionWeakFormulation weak_form(beta);

    int elem_id = 0;
    int face_id = 1;  // Same face as BoundaryFaceIntegralOutflow above.

    auto bc_func = [](const Vec2& x) { return 0.0; };
    auto [L_bc, F_bc] = weak_form.compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);

    EXPECT_EQ(L_bc.extent(0), 3);
    EXPECT_EQ(L_bc.extent(1), 3);
    EXPECT_EQ(F_bc.size(), 3);

    EXPECT_NEAR(norm_of(F_bc), 0.0, 1e-12);

    // Exact analytic check for order-1 DubinerBasis, using the phi_0/phi_1/phi_2 closed forms
    // from MassIntegralDubinerBasis above. Face 1 runs from ref (1,0) to ref (0,1),
    // parametrized as (xi,eta)=(1-t,t), t in [0,1], physical arc length ds=dt exactly (same
    // face/parametrization as the monomial case above). Along this face: phi_0=2,
    // phi_1=sqrt(6)*(4*(1-t)+2*t-2)=2*sqrt(6)*(1-t), phi_2=2*sqrt(2)*(3*t-1).
    // L_bc(i,j) = -integral_0^1 phi_i(t)*phi_j(t) dt.
    double s6 = std::sqrt(6.0);
    double s2 = std::sqrt(2.0);
    EXPECT_NEAR(L_bc(0, 0), -4.0, 1e-10);
    EXPECT_NEAR(L_bc(0, 1), -2.0 * s6, 1e-10);
    EXPECT_NEAR(L_bc(0, 2), -2.0 * s2, 1e-10);
    EXPECT_NEAR(L_bc(1, 1), -8.0, 1e-10);
    EXPECT_NEAR(L_bc(1, 2), 0.0, 1e-10);
    EXPECT_NEAR(L_bc(2, 2), -8.0, 1e-10);
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

TEST(EulerWeakFormulationTest, Construction) {
    EXPECT_NO_THROW(EulerWeakFormulation weak_form(1.4));
}

TEST(EulerWeakFormulationTest, ConservedPrimitiveConversion) {
    // Test conversion between conserved and primitive variables
    Vec4 U{1.0, 2.0, 3.0, 10.0};  // rho, rhou, rhov, E

    Vec4 W = conserved_to_primitive(U, 1.4);
    Vec4 U_back = primitive_to_conserved(W, 1.4);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(U[i], U_back[i], 1e-12);
    }
}

TEST(EulerWeakFormulationTest, RusanovFlux) {
    EulerWeakFormulation weak_form(1.4);

    // Sod shock tube left/right states: E = p/(gamma-1) since u=v=0.
    Vec4 U_L{1.0, 0.0, 0.0, 2.5};     // rho=1, u=v=0, p=1
    Vec4 U_R{0.125, 0.0, 0.0, 0.25};  // rho=0.125, u=v=0, p=0.1

    Vec2 n{1.0, 0.0};  // x-direction normal

    Vec4 flux = weak_form.rusanov_flux(U_L, U_R, n);

    // A consistent numerical flux must satisfy F(U_L, U_R, n) = -F(U_R, U_L, -n): swapping
    // which side is "left" and reversing the face normal describes the same physical face.
    Vec2 neg_n{-1.0, 0.0};
    Vec4 neg_flux = weak_form.rusanov_flux(U_R, U_L, neg_n);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(flux[i], -neg_flux[i], 1e-12);
    }
}
