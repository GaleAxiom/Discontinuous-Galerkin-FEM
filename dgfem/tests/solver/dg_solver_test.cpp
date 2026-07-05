#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <stdexcept>

#include <gmock/gmock.h>
#include <gmsh.h>
#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace dgfem;
using namespace testing;

class LaplaceDGSolverTest : public Test {
protected:
    void SetUp() override {
        // Clear any existing gmsh models and reinitialize
        // Note: gmsh::isInitialized() is not available in all GMSH versions
        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
            // GMSH not initialized, this is fine
        }
        gmsh::initialize();
        mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0.0, 1.0, 0.0, 1.0);
    }
    std::shared_ptr<DGMesh> mesh;
};

TEST_F(LaplaceDGSolverTest, Construction) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    EXPECT_NO_THROW(LaplaceDGSolver(mesh, 10.0));
}

TEST_F(LaplaceDGSolverTest, SolveWithoutSource) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);
    auto solution = solver->solve(nullptr);
    EXPECT_EQ(solution.size(), mesh->get_n_elements() * space->get_basis()->get_n_basis());
}

TEST_F(LaplaceDGSolverTest, AllNeumannWithZeroDataGivesTrivialSolution) {
    // Pure-Neumann Laplace problems are singular up to an additive constant (no Dirichlet
    // data pins the solution down). With zero flux data and a zero source term, u = 0 is a
    // valid particular solution, and Amesos2's direct solve on this mesh resolves to it
    // without throwing -- this pins that down as a regression check, since a different
    // singular-system handling policy could just as easily start throwing or returning
    // garbage here. See NeumannManufacturedSolution/RobinManufacturedSolution below for
    // coverage of nonzero, non-degenerate Neumann/Robin data.
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto bc_neumann = dgfem::make_neumann_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_neumann);
    mesh->set_boundary_condition("Top", bc_neumann);
    mesh->set_boundary_condition("Left", bc_neumann);
    mesh->set_boundary_condition("Right", bc_neumann);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);

    DView1 solution;
    EXPECT_NO_THROW(solution = solver->solve(nullptr));
    EXPECT_NEAR(norm(solution), 0.0, 1e-10);
}

TEST_F(LaplaceDGSolverTest, NeumannBoundaryRecoversLinearManufacturedSolution) {
    // u(x,y) = 2x + 3y + 1 is harmonic (zero source) and linear, so it lies exactly in the P1
    // DG space. Galerkin/Nitsche consistency means a correctly implemented Neumann BC should
    // let the solver reproduce it essentially exactly, not just approximately -- this exercises
    // the compute_boundary_rhs_integral NEUMANN branch (laplace_weak_formulation.cpp) with a
    // genuinely non-zero flux, unlike the all-zero-data regression test above.
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto exact_solution = [](const Vec2& x) -> double { return 2.0 * x[0] + 3.0 * x[1] + 1.0; };
    auto exact_gradient = [](const Vec2&) -> Vec2 { return Vec2{2.0, 3.0}; };

    // Dirichlet on Left/Right/Top using the exact trace; Neumann on Bottom using the exact
    // outward flux du/dn = grad(u)*(0,-1) = -3.
    mesh->set_boundary_condition("Left", dgfem::make_dirichlet_bc(exact_solution));
    mesh->set_boundary_condition("Right", dgfem::make_dirichlet_bc(exact_solution));
    mesh->set_boundary_condition("Top", dgfem::make_dirichlet_bc(exact_solution));
    mesh->set_boundary_condition("Bottom", dgfem::make_neumann_bc(-3.0));

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);
    solver->solve(nullptr);

    auto errors = solver->compute_error(exact_solution, exact_gradient);
    EXPECT_NEAR(errors["L2"], 0.0, 1e-8);
    EXPECT_NEAR(errors["H1"], 0.0, 1e-8);
}

TEST_F(LaplaceDGSolverTest, RobinBoundaryRecoversLinearManufacturedSolution) {
    // Same manufactured solution as above, but the Top boundary uses a Robin condition
    // alpha*u + du/dn = g with alpha = 2.0: g(x, 1) = alpha*(2x + 3 + 1) + 3, which varies
    // with x and so exercises the position-dependent BC evaluation path for the ROBIN branch.
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto exact_solution = [](const Vec2& x) -> double { return 2.0 * x[0] + 3.0 * x[1] + 1.0; };
    auto exact_gradient = [](const Vec2&) -> Vec2 { return Vec2{2.0, 3.0}; };

    const double alpha = 2.0;
    auto robin_data = [alpha, exact_solution](const Vec2& x) -> double {
        return alpha * exact_solution(x) + 3.0;  // alpha*u + du/dn, du/dn = 3 on Top
    };

    mesh->set_boundary_condition("Left", dgfem::make_dirichlet_bc(exact_solution));
    mesh->set_boundary_condition("Right", dgfem::make_dirichlet_bc(exact_solution));
    mesh->set_boundary_condition("Bottom", dgfem::make_dirichlet_bc(exact_solution));
    mesh->set_boundary_condition("Top", dgfem::make_robin_bc(robin_data, alpha));

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);
    solver->solve(nullptr);

    auto errors = solver->compute_error(exact_solution, exact_gradient);
    EXPECT_NEAR(errors["L2"], 0.0, 1e-8);
    EXPECT_NEAR(errors["H1"], 0.0, 1e-8);
}

TEST_F(LaplaceDGSolverTest, SolveWithSource) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);

    auto source_func = [](const Vec2& x) -> double {
        return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
    };

    auto solution = solver->solve(source_func);
    EXPECT_EQ(solution.size(), mesh->get_n_elements() * space->get_basis()->get_n_basis());
}

TEST_F(LaplaceDGSolverTest, ComputeError) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);

    // Test solution u(x,y) = x(1-x)y(1-y)
    auto exact_solution = [](const Vec2& x) -> double {
        return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
    };

    auto exact_gradient = [](const Vec2& x) -> Vec2 {
        return Vec2{(1.0 - 2.0 * x[0]) * x[1] * (1.0 - x[1]),
                    x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1])};
    };

    solver->solve(nullptr);  // Solve homogeneous problem first
    auto errors = solver->compute_error(exact_solution, exact_gradient);

    EXPECT_GT(errors["L2"], 0.0);
    EXPECT_GT(errors["H1"], errors["L2"]);  // H1 error includes gradient
}

TEST_F(LaplaceDGSolverTest, FullSolver) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Set up boundary conditions (homogeneous Dirichlet)
    std::cout << "\nSetting up boundary conditions..." << std::endl;
    auto bc_zero = dgfem::make_dirichlet_bc(0.0);

    // Print boundary tag information first
    const auto& boundary_tags = mesh->get_boundary_tags();
    std::cout << "Available boundary tags: ";
    for (const auto& bt : boundary_tags) {
        std::cout << bt.first << "=" << bt.second << " ";
    }
    std::cout << std::endl;

    // For simplicity, assign the same BC to all boundaries using the first boundary tag
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        for (int face_id = 0; face_id < mesh->get_n_faces_per_element(); ++face_id) {
            if (mesh->is_boundary_face(elem_id, face_id)) {
                auto bc = mesh->get_boundary_condition(elem_id, face_id);
                std::cout << "  Element " << elem_id << " Face " << face_id
                          << " BC Type: " << static_cast<int>(bc->get_type())
                          << " BC Value: " << bc->get_value() << std::endl;  // Expect 0 (DIRICHLET)

                EXPECT_EQ(bc->get_type(), BCType::DIRICHLET);
                EXPECT_EQ(bc->get_value(), 0.0);
            }
        }
    }
    // Define source function f(x,y) = 2*(y(1-y) + x(1-x))
    auto source_function = [](const Vec2& x) -> double {
        return 2.0 * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]));
    };

    // Define exact solution for error computation
    auto exact_solution = [](const Vec2& x) -> double {
        return x[0] * (1 - x[0]) * x[1] * (1 - x[1]);
    };

    // Define exact gradient
    auto exact_gradient = [](const Vec2& x) -> Vec2 {
        Vec2 grad;
        grad[0] = (1 - 2 * x[0]) * x[1] * (1 - x[1]);
        grad[1] = x[0] * (1 - x[0]) * (1 - 2 * x[1]);
        return grad;
    };

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);

    solver->solve(source_function);
    auto errors = solver->compute_error(exact_solution, exact_gradient);

    std::cout << "Error analysis:" << std::endl;
    std::cout << "  L² error: " << errors["L2"] << std::endl;
    std::cout << "  H¹ error: " << errors["H1"] << std::endl;

    EXPECT_LT(errors["L2"], 1.3e-2);
    EXPECT_LT(errors["H1"], 1.1e-1);
}

TEST_F(LaplaceDGSolverTest, SystemMatrixTest) {
    // Explicit "monomial" override: the golden matrix below is tied to
    // MonomialBasisTriangle's specific representation, independent of whichever basis
    // DGSpace::create_basis picks by default for triangles.
    auto space = std::make_shared<DGSpace>("triangle", 1, "monomial");
    mesh->initialize_dg_space(space);

    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);

    solver->solve(nullptr);

    auto system_matrix_sparse = solver->get_system_matrix();
    DView2 system_matrix_dense = tpetra_to_dense(*system_matrix_sparse);

    double expected_flat[144] = {
        120.,  41.,          39.5,        -40.,  -19.5,        -20.,
        -40.,  -19.5,        -0.5,        0.,    0.,           0.,
        41.,   26.66666667,  7.16666667,  -19.5, -12.83333333, -6.41666667,
        -19.5, -12.83333333, -0.25,       0.,    0.,           0.,
        39.5,  7.16666667,   26.16666667, -0.5,  -0.25,        -0.25,
        -20.,  -6.41666667,  -0.25,       0.,    0.,           0.,
        -40.,  -19.5,        -0.5,        120.,  41.,          39.5,
        0.,    0.,           0.,          -40.,  -19.5,        -20.,
        -19.5, -12.83333333, -0.25,       41.,   26.66666667,  7.16666667,
        0.,    0.,           0.,          -19.5, -12.83333333, -6.41666667,
        -20.,  -6.41666667,  -0.25,       39.5,  7.16666667,   26.16666667,
        0.,    0.,           0.,          -0.5,  -0.25,        -0.25,
        -40.,  -19.5,        -20.,        0.,    0.,           0.,
        120.,  41.,          39.5,        -40.,  -19.5,        -0.5,
        -19.5, -12.83333333, -6.41666667, 0.,    0.,           0.,
        41.,   26.66666667,  7.16666667,  -19.5, -12.83333333, -0.25,
        -0.5,  -0.25,        -0.25,       0.,    0.,           0.,
        39.5,  7.16666667,   26.16666667, -20.,  -6.41666667,  -0.25,
        0.,    0.,           0.,          -40.,  -19.5,        -0.5,
        -40.,  -19.5,        -20.,        120.,  41.,          39.5,
        0.,    0.,           0.,          -19.5, -12.83333333, -0.25,
        -19.5, -12.83333333, -6.41666667, 41.,   26.66666667,  7.16666667,
        0.,    0.,           0.,          -20.,  -6.41666667,  -0.25,
        -0.5,  -0.25,        -0.25,       39.5,  7.16666667,   26.16666667};
    DView2 expected_matrix("expected_matrix", 12, 12);
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            expected_matrix(i, j) = expected_flat[i * 12 + j];
        }
    }

    ASSERT_EQ(system_matrix_dense.extent(0), expected_matrix.extent(0));
    ASSERT_EQ(system_matrix_dense.extent(1), expected_matrix.extent(1));

    for (int i = 0; i < static_cast<int>(expected_matrix.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(expected_matrix.extent(1)); ++j) {
            EXPECT_NEAR(system_matrix_dense(i, j), expected_matrix(i, j), 1e-5);
        }
    }
}

TEST_F(LaplaceDGSolverTest, SystemMatrixTestDubinerBasis) {
    // Same setup as SystemMatrixTest above, but forcing the orthogonal DubinerBasis instead
    // of MonomialBasisTriangle, so both bases get real regression coverage of this solver's
    // assembled system matrix.
    auto space = std::make_shared<DGSpace>("triangle", 1, "dubiner");
    mesh->initialize_dg_space(space);

    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);

    solver->solve(nullptr);

    auto system_matrix_sparse = solver->get_system_matrix();
    DView2 system_matrix_dense = tpetra_to_dense(*system_matrix_sparse);

    // Golden matrix captured from running this exact setup with DubinerBasis (not hand-derived
    // like the MonomialBasisTriangle case above -- Dubiner's Jacobi-polynomial closed form
    // makes hand derivation impractical for a full 4-element assembled system). Sanity-checked
    // below: symmetric (required for this self-adjoint SIPG formulation, regardless of basis).
    double expected_flat[144] = {480,
                                 0,
                                 16.97056275,
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
                                 960,
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
                                 16.97056275,
                                 0,
                                 912,
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
                                 480,
                                 0,
                                 16.97056275,
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
                                 960,
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
                                 16.97056275,
                                 0,
                                 912,
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
                                 480,
                                 0,
                                 16.97056275,
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
                                 960,
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
                                 16.97056275,
                                 0,
                                 912,
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
                                 480,
                                 0,
                                 16.97056275,
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
                                 960,
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
                                 16.97056275,
                                 0,
                                 912};
    DView2 expected_matrix("expected_matrix", 12, 12);
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            expected_matrix(i, j) = expected_flat[i * 12 + j];
        }
    }

    ASSERT_EQ(system_matrix_dense.extent(0), expected_matrix.extent(0));
    ASSERT_EQ(system_matrix_dense.extent(1), expected_matrix.extent(1));

    for (int i = 0; i < static_cast<int>(expected_matrix.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(expected_matrix.extent(1)); ++j) {
            EXPECT_NEAR(system_matrix_dense(i, j), expected_matrix(i, j), 1e-5);
        }
    }

    for (int i = 0; i < static_cast<int>(system_matrix_dense.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(system_matrix_dense.extent(1)); ++j) {
            EXPECT_NEAR(system_matrix_dense(i, j), system_matrix_dense(j, i), 1e-5)
                << "Matrix not symmetric at (" << i << "," << j << ")";
        }
    }
}

// ============================================================================
// Advection Solver Tests
// ============================================================================

class AdvectionDGSolverTest : public Test {
protected:
    void SetUp() override {
        // Clear any existing gmsh models and reinitialize
        // Note: gmsh::isInitialized() is not available in all GMSH versions
        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
            // GMSH not initialized, this is fine
        }
        gmsh::initialize();
        mesh = dgfem::MeshCreator::create_rectangular_mesh(1, true, 0.0, 1.0, 0.0, 1.0);
    }
    std::shared_ptr<DGMesh> mesh;
};

TEST_F(AdvectionDGSolverTest, WeakFormConstruction) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    Vec2 velocity{1.0, 0.0};
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);
    EXPECT_NE(weak_form, nullptr);
}

TEST_F(AdvectionDGSolverTest, MassMatrixAssembly) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    Vec2 velocity{1.0, 0.0};

    // Access assembler to test mass matrix
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    DView2 M = tpetra_to_dense(*assembler->assemble_mass_matrix());

    int n_dofs = mesh->get_n_elements() * space->get_basis()->get_n_basis();
    EXPECT_EQ(M.extent(0), n_dofs);
    EXPECT_EQ(M.extent(1), n_dofs);

    // Mass matrix should be symmetric
    EXPECT_NEAR(asymmetry_norm(M), 0.0, 1e-10);

    // Mass matrix should be positive definite (diagonal dominant)
    for (int i = 0; i < static_cast<int>(M.extent(0)); ++i) {
        EXPECT_GT(M(i, i), 0.0);
    }
}

TEST_F(AdvectionDGSolverTest, ElementIntegralComputation) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    Vec2 velocity{1.0, 0.5};
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);

    // Test element 0
    int elem_id = 0;
    int n_basis = space->get_basis()->get_n_basis();
    const auto& elem_data = mesh->get_element_data(elem_id);

    // Test mass integral
    DView2 M_elem = weak_form->compute_mass_integral(elem_data, space);
    EXPECT_EQ(M_elem.extent(0), n_basis);
    EXPECT_EQ(M_elem.extent(1), n_basis);

    // Mass matrix should be symmetric
    EXPECT_NEAR(asymmetry_norm(M_elem), 0.0, 1e-12);

    // Test stiffness integral
    DView2 L_elem = weak_form->compute_volume_integral(elem_data, space);
    EXPECT_EQ(L_elem.extent(0), n_basis);
    EXPECT_EQ(L_elem.extent(1), n_basis);
}

TEST_F(AdvectionDGSolverTest, SolveRejectsNonPositiveDtOrTFinal) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    Vec2 velocity{1.0, 0.0};
    AdvectionDGSolver solver(mesh, velocity);
    auto ic = [](const Vec2&) { return 0.0; };

    EXPECT_THROW(solver.solve(ic, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(solver.solve(ic, 1.0, -0.1), std::invalid_argument);
    EXPECT_THROW(solver.solve(ic, 0.0, 0.1), std::invalid_argument);
    EXPECT_THROW(solver.solve(ic, -1.0, 0.1), std::invalid_argument);
}

TEST_F(AdvectionDGSolverTest, WeakFormVelocityFields) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Test different velocity fields
    Vec2 velocity_zero{0.0, 0.0};
    auto wf_zero = std::make_shared<AdvectionWeakFormulation>(velocity_zero);
    EXPECT_NE(wf_zero, nullptr);

    Vec2 velocity_x{1.0, 0.0};
    auto wf_x = std::make_shared<AdvectionWeakFormulation>(velocity_x);
    EXPECT_NE(wf_x, nullptr);

    Vec2 velocity_y{0.0, 1.0};
    auto wf_y = std::make_shared<AdvectionWeakFormulation>(velocity_y);
    EXPECT_NE(wf_y, nullptr);

    Vec2 velocity_diag{1.0, 1.0};
    auto wf_diag = std::make_shared<AdvectionWeakFormulation>(velocity_diag);
    EXPECT_NE(wf_diag, nullptr);
}