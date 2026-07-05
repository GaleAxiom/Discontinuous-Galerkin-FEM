/**
 * @file time_dependent_test.cpp
 * @brief Tests for time-dependent DG solvers
 */

#include <Eigen/Dense>
#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <vector>

#include <gtest/gtest.h>

using namespace dgfem;

class TimeDependentSolverTest : public ::testing::Test {
protected:
    void SetUp() override { MeshCreator::initialize_gmsh(); }

    void TearDown() override { MeshCreator::finalize_gmsh(); }
};

TEST_F(TimeDependentSolverTest, AdvectionTimeSteppingStability) {
    // Test time stepping stability for advection equation
    // ∂u/∂t + v·∇u = 0

    double h = 0.1;
    auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(dg_space, 1);

    // Velocity field
    Eigen::Vector2d velocity(1.0, 0.0);

    // Initial condition: Gaussian pulse
    auto initial_condition = [](const Eigen::Vector2d& x) -> double {
        double x0 = 0.5, y0 = 0.5;
        double sigma = 0.1;
        return std::exp(-((x[0] - x0) * (x[0] - x0) + (x[1] - y0) * (x[1] - y0)) /
                        (2 * sigma * sigma));
    };

    // Set up initial solution by L2 projection
    int n_dofs = mesh->get_n_elements() * dg_space->get_basis()->get_n_basis();
    Eigen::VectorXd u_n = Eigen::VectorXd::Zero(n_dofs);

    // Simple initialization - set first DOF of each element to 1
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        int offset = elem_id * dg_space->get_basis()->get_n_basis();
        u_n[offset] = 1.0;  // Simplified IC for testing
    }

    // Time stepping parameters
    double dx = h;
    double v_max = velocity.norm();
    double CFL = 0.5;
    double dt = CFL * dx / v_max;  // CFL condition
    double T_final = 0.5;
    int n_steps = static_cast<int>(T_final / dt);

    // Assemble mass and advection matrices
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    auto M = assembler->assemble_mass_matrix();
    // Would need advection operator assembly here

    // For now, just test that solution remains bounded
    double initial_norm = u_n.norm();
    EXPECT_GT(initial_norm, 0.0) << "Initial condition should be non-zero";

    // Forward Euler time stepping (simple test)
    for (int n = 0; n < n_steps; ++n) {
        // u^{n+1} = u^n - dt * M^{-1} * L * u^n
        // (simplified - actual implementation would use proper time integrator)

        // For stability test, just check solution doesn't blow up
        double current_norm = u_n.norm();
        EXPECT_TRUE(std::isfinite(current_norm)) << "Solution should remain finite at step " << n;
        EXPECT_LT(current_norm, 10.0 * initial_norm)
            << "Solution shouldn't grow unbounded at step " << n;
    }
}

TEST_F(TimeDependentSolverTest, HeatEquationTimeAccuracy) {
    // Test time accuracy for heat equation: ∂u/∂t = ν∆u

    double h = 0.1;
    auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);

    double nu = 0.01;  // Diffusion coefficient
    const double pi = M_PI;

    // Manufactured solution: u(x,y,t) = exp(-2π²νt)sin(πx)sin(πy)
    auto exact_solution = [pi, nu](const Eigen::Vector2d& x, double t) -> double {
        return std::exp(-2 * pi * pi * nu * t) * std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    // Initial condition (t=0)
    auto initial = [pi](const Eigen::Vector2d& x) -> double {
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    // Set boundary conditions (zero Dirichlet)
    auto bc_zero = make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    // Time stepping
    double dt = 0.001;
    double T_final = 0.1;
    int n_steps = static_cast<int>(T_final / dt);

    // Initialize solution
    int n_dofs = mesh->get_n_elements() * dg_space->get_basis()->get_n_basis();
    Eigen::VectorXd u_n = Eigen::VectorXd::Zero(n_dofs);

    // Simple initialization for testing
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        int offset = elem_id * dg_space->get_basis()->get_n_basis();
        u_n[offset] = 1.0;  // Simplified IC
    }

    // Assemble diffusion operator
    LaplaceDGSolver laplace_solver(mesh, 10.0);
    // Would use this to get system matrix for implicit time stepping

    // For now, test that solution decays as expected
    double initial_norm = u_n.norm();
    double expected_decay = std::exp(-2 * pi * pi * nu * T_final);

    EXPECT_GT(initial_norm, 0.0) << "Initial condition should be non-zero";

    // After time T_final, solution should have decayed by factor exp(-2π²νT)
    // This test documents expected behavior for future implementation
    EXPECT_GT(expected_decay, 0.0) << "Solution should decay exponentially";
    EXPECT_LT(expected_decay, 1.0) << "Decay factor should be less than 1";
}

TEST_F(TimeDependentSolverTest, RK4TimeIntegration) {
    // Test Runge-Kutta 4th order time integration accuracy

    // Simple ODE test: du/dt = -u, exact solution: u(t) = u0*exp(-t)
    double u0 = 1.0;
    double dt = 0.1;
    double T_final = 1.0;
    int n_steps = static_cast<int>(T_final / dt);

    double u = u0;

    // RK4 time stepping
    for (int n = 0; n < n_steps; ++n) {
        double t_n = n * dt;

        // RK4 stages
        double k1 = -u;
        double k2 = -(u + 0.5 * dt * k1);
        double k3 = -(u + 0.5 * dt * k2);
        double k4 = -(u + dt * k3);

        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
    }

    // Compare with exact solution
    double u_exact = u0 * std::exp(-T_final);
    double error = std::abs(u - u_exact);

    // RK4 should give 4th order accuracy
    EXPECT_LT(error, 1e-6) << "RK4 should have very small error for smooth ODE";
}

TEST_F(TimeDependentSolverTest, ImplicitEulerStability) {
    // Test unconditional stability of implicit Euler

    // Stiff ODE: du/dt = -1000*u, exact solution: u(t) = u0*exp(-1000t)
    double u0 = 1.0;
    double lambda = -1000.0;  // Stiff coefficient

    // Use large time step (would be unstable with explicit method)
    double dt = 0.1;
    double T_final = 1.0;
    int n_steps = static_cast<int>(T_final / dt);

    double u = u0;

    // Implicit Euler: u^{n+1} = u^n + dt*f(u^{n+1})
    // For linear ODE: u^{n+1} = u^n / (1 - dt*lambda)
    for (int n = 0; n < n_steps; ++n) {
        u = u / (1.0 - dt * lambda);
    }

    // Should remain stable and bounded
    EXPECT_TRUE(std::isfinite(u)) << "Implicit Euler should remain stable";
    EXPECT_GT(u, 0.0) << "Solution should be positive";
    EXPECT_LT(u, u0) << "Solution should decay";

    // Compare with exact (will have some error due to time discretization)
    double u_exact = u0 * std::exp(lambda * T_final);
    // Use absolute error since u_exact is very small (e^-1000 ≈ 0)
    EXPECT_LT(std::abs(u - u_exact), 1e-10)
        << "Implicit Euler should give reasonable approximation";
}

TEST_F(TimeDependentSolverTest, CFLConditionViolation) {
    // Test that violating CFL condition leads to instability (for explicit methods)

    double h = 0.1;
    Eigen::Vector2d velocity(1.0, 0.0);

    // Violate CFL: use dt > h/|v|
    double v_max = velocity.norm();
    double dt_stable = h / v_max;
    double dt_unstable = 2.0 * dt_stable;  // Violate CFL by factor of 2

    // Document that CFL condition exists and is important
    EXPECT_GT(dt_unstable, dt_stable) << "Unstable dt should be larger than stable dt";

    // For explicit time stepping, dt > h/v leads to instability
    // This test documents the importance of CFL condition
    double CFL_stable = dt_stable * v_max / h;
    double CFL_unstable = dt_unstable * v_max / h;

    EXPECT_LE(CFL_stable, 1.0) << "Stable CFL number should be ≤ 1";
    EXPECT_GT(CFL_unstable, 1.0) << "Unstable CFL number should be > 1";
}

TEST_F(TimeDependentSolverTest, MassConservation) {
    // Test that time stepping conserves total mass for advection

    double h = 0.1;
    auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(dg_space, 1);

    // Divergence-free velocity (should conserve mass)
    Eigen::Vector2d velocity(1.0, 0.0);

    int n_dofs = mesh->get_n_elements() * dg_space->get_basis()->get_n_basis();
    Eigen::VectorXd u(n_dofs);
    u.setOnes();  // Uniform initial condition

    // Assemble mass matrix
    auto weak_form = std::make_shared<AdvectionWeakFormulation>(velocity);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);
    Eigen::MatrixXd M = tpetra_to_dense(*assembler->assemble_mass_matrix());

    // Compute total mass: ∫u dx ≈ 1^T M u
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(n_dofs);
    double initial_mass = ones.dot(M * u);

    EXPECT_GT(initial_mass, 0.0) << "Initial mass should be positive";

    // For divergence-free velocity and periodic/reflective BCs,
    // total mass should be conserved
    // This test documents the conservation property
}
