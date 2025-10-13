#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <gmsh.h>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/utils/mesh_creation.hpp>
#include <Eigen/Dense>
#include <cmath>
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

TEST_F(LaplaceDGSolverTest, SolveWithSource) {
    
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);

    auto source_func = [](const Eigen::Vector2d& x) -> double {
        return std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
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
    auto exact_solution = [](const Eigen::Vector2d& x) -> double {
        return x(0) * (1.0 - x(0)) * x(1) * (1.0 - x(1));
    };
    
    auto exact_gradient = [](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        return Eigen::Vector2d(
            (1.0 - 2.0*x(0)) * x(1) * (1.0 - x(1)),
            x(0) * (1.0 - x(0)) * (1.0 - 2.0*x(1))
        );
    };
    
    solver->solve(nullptr);  // Solve homogeneous problem first
    auto errors = solver->compute_error(exact_solution, exact_gradient);
    
    EXPECT_GT(errors["L2"], 0.0);
    EXPECT_GT(errors["H1"], errors["L2"]);  // H1 error includes gradient
}

TEST_F(LaplaceDGSolverTest, FullSolver)
{
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
                            << " BC Value: " << bc->get_value()
                            << std::endl; // Expect 0 (DIRICHLET)

                EXPECT_EQ(bc->get_type(), BCType::DIRICHLET);
                EXPECT_EQ(bc->get_value(), 0.0);
            }
        }
    }
    // Define source function f(x,y) = 2*(y(1-y) + x(1-x))
    auto source_function = [](const Eigen::Vector2d& x) -> double {
        return 2.0 * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]));
    };
    
    // Define exact solution for error computation
    auto exact_solution = [](const Eigen::Vector2d& x) -> double {
        return x[0] * (1 - x[0]) * x[1] * (1 - x[1]);
    };
    
    // Define exact gradient
    auto exact_gradient = [](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        Eigen::Vector2d grad;
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

TEST_F(LaplaceDGSolverTest, SystemMatrixTest)
{
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto solver = std::make_unique<LaplaceDGSolver>(mesh, 10.0);
    
    solver->solve(nullptr);

    const auto& system_matrix_sparse = solver->get_system_matrix();
    Eigen::MatrixXd system_matrix_dense = system_matrix_sparse;

    Eigen::MatrixXd expected_matrix(12, 12);
    expected_matrix << 
        120., 41., 39.5, -40., -19.5, -20., -40., -19.5, -0.5, 0., 0., 0.,
        41., 26.66666667, 7.16666667, -19.5, -12.83333333, -6.41666667, -19.5, -12.83333333, -0.25, 0., 0., 0.,
        39.5, 7.16666667, 26.16666667, -0.5, -0.25, -0.25, -20., -6.41666667, -0.25, 0., 0., 0.,
        -40., -19.5, -0.5, 120., 41., 39.5, 0., 0., 0., -40., -19.5, -20.,
        -19.5, -12.83333333, -0.25, 41., 26.66666667, 7.16666667, 0., 0., 0., -19.5, -12.83333333, -6.41666667,
        -20., -6.41666667, -0.25, 39.5, 7.16666667, 26.16666667, 0., 0., 0., -0.5, -0.25, -0.25,
        -40., -19.5, -20., 0., 0., 0., 120., 41., 39.5, -40., -19.5, -0.5,
        -19.5, -12.83333333, -6.41666667, 0., 0., 0., 41., 26.66666667, 7.16666667, -19.5, -12.83333333, -0.25,
        -0.5, -0.25, -0.25, 0., 0., 0., 39.5, 7.16666667, 26.16666667, -20., -6.41666667, -0.25,
        0., 0., 0., -40., -19.5, -0.5, -40., -19.5, -20., 120., 41., 39.5,
        0., 0., 0., -19.5, -12.83333333, -0.25, -19.5, -12.83333333, -6.41666667, 41., 26.66666667, 7.16666667,
        0., 0., 0., -20., -6.41666667, -0.25, -0.5, -0.25, -0.25, 39.5, 7.16666667, 26.16666667;

    ASSERT_EQ(system_matrix_dense.rows(), expected_matrix.rows());
    ASSERT_EQ(system_matrix_dense.cols(), expected_matrix.cols());

    for (int i = 0; i < expected_matrix.rows(); ++i) {
        for (int j = 0; j < expected_matrix.cols(); ++j) {
            EXPECT_NEAR(system_matrix_dense(i, j), expected_matrix(i, j), 1e-5);
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
    
    Eigen::Vector2d velocity(1.0, 0.0);
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);
    EXPECT_NE(weak_form, nullptr);
}

TEST_F(AdvectionDGSolverTest, MassMatrixAssembly) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    Eigen::Vector2d velocity(1.0, 0.0);
    
    // Access assembler to test mass matrix
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);
    
    Eigen::SparseMatrix<double> M = assembler->assemble_mass_matrix();
    
    int n_dofs = mesh->get_n_elements() * space->get_basis()->get_n_basis();
    EXPECT_EQ(M.rows(), n_dofs);
    EXPECT_EQ(M.cols(), n_dofs);
    
    // Mass matrix should be symmetric
    Eigen::SparseMatrix<double> M_transpose = M.transpose();
    EXPECT_NEAR((M - M_transpose).norm(), 0.0, 1e-10);
    
    // Mass matrix should be positive definite (diagonal dominant)
    for (int i = 0; i < M.rows(); ++i) {
        EXPECT_GT(M.coeff(i, i), 0.0);
    }
}

TEST_F(AdvectionDGSolverTest, ElementIntegralComputation) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    Eigen::Vector2d velocity(1.0, 0.5);
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);
    
    // Test element 0
    int elem_id = 0;
    int n_basis = space->get_basis()->get_n_basis();
    const auto& elem_data = mesh->get_element_data(elem_id);
    
    // Test mass integral
    Eigen::MatrixXd M_elem = weak_form->compute_mass_integral(elem_data, space);
    EXPECT_EQ(M_elem.rows(), n_basis);
    EXPECT_EQ(M_elem.cols(), n_basis);
    
    // Mass matrix should be symmetric
    EXPECT_NEAR((M_elem - M_elem.transpose()).norm(), 0.0, 1e-12);
    
    // Test stiffness integral
    Eigen::MatrixXd L_elem = weak_form->compute_volume_integral(elem_data, space);
    EXPECT_EQ(L_elem.rows(), n_basis);
    EXPECT_EQ(L_elem.cols(), n_basis);
}

TEST_F(AdvectionDGSolverTest, WeakFormVelocityFields) {
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    
    // Test different velocity fields
    Eigen::Vector2d velocity_zero(0.0, 0.0);
    auto wf_zero = std::make_shared<AdvectionWeakFormulation>(velocity_zero);
    EXPECT_NE(wf_zero, nullptr);
    
    Eigen::Vector2d velocity_x(1.0, 0.0);
    auto wf_x = std::make_shared<AdvectionWeakFormulation>(velocity_x);
    EXPECT_NE(wf_x, nullptr);
    
    Eigen::Vector2d velocity_y(0.0, 1.0);
    auto wf_y = std::make_shared<AdvectionWeakFormulation>(velocity_y);
    EXPECT_NE(wf_y, nullptr);
    
    Eigen::Vector2d velocity_diag(1.0, 1.0);
    auto wf_diag = std::make_shared<AdvectionWeakFormulation>(velocity_diag);
    EXPECT_NE(wf_diag, nullptr);
}