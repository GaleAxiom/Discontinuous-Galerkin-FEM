/**
 * @file p_convergence_test.cpp
 * @brief Tests for p-refinement convergence rates of DG solver
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

class PRefinementTest : public ::testing::Test {
protected:
    void SetUp() override { MeshCreator::initialize_gmsh(); }

    void TearDown() override { MeshCreator::finalize_gmsh(); }

    /**
     * @brief Compute convergence rate from error at two polynomial orders
     * @param p1 First polynomial order
     * @param err1 Error at p1
     * @param p2 Second polynomial order
     * @param err2 Error at p2
     * @return Convergence rate
     */
    double compute_p_convergence_rate(int p1, double err1, int p2, double err2) {
        return std::log(err1 / err2) / std::log(static_cast<double>(p2) / p1);
    }
};

TEST_F(PRefinementTest, SmoothSolutionPConvergence) {
    // For smooth solutions, p-refinement should give exponential convergence
    // Manufactured solution: u = sin(πx)sin(πy)
    // -Δu = 2π²sin(πx)sin(πy)

    const double pi = M_PI;
    auto exact = [pi](const Eigen::Vector2d& x) -> double {
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_grad = [pi](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        Eigen::Vector2d grad;
        grad[0] = pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
        grad[1] = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
        return grad;
    };

    auto source = [pi](const Eigen::Vector2d& x) -> double {
        return 2.0 * pi * pi * std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    // Fixed mesh, varying polynomial order
    double h = 0.1;
    std::vector<int> orders = {1, 2, 3, 4};
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;

    double penalty = 10.0;

    for (int p : orders) {
        auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("triangle", p);
        mesh->initialize_dg_space(dg_space, 1);

        // Set zero Dirichlet BCs (exact solution is zero on boundary)
        auto bc_zero = make_dirichlet_bc(0.0);
        mesh->set_boundary_condition("Bottom", bc_zero);
        mesh->set_boundary_condition("Top", bc_zero);
        mesh->set_boundary_condition("Left", bc_zero);
        mesh->set_boundary_condition("Right", bc_zero);

        LaplaceDGSolver solver(mesh, penalty);
        Eigen::VectorXd solution = solver.solve(source);

        auto errors = solver.compute_error(exact, exact_grad);
        l2_errors.push_back(errors["L2"]);
        h1_errors.push_back(errors["H1"]);
    }

    // For smooth solutions, error should decrease exponentially with p
    // Check that each refinement reduces error
    for (size_t i = 1; i < l2_errors.size(); ++i) {
        EXPECT_LT(l2_errors[i], l2_errors[i - 1]) << "L2 error should decrease with increasing p";
        EXPECT_LT(h1_errors[i], h1_errors[i - 1]) << "H1 error should decrease with increasing p";
    }

    // Exponential convergence means error should drop by at least factor of 2 each time
    EXPECT_LT(l2_errors[1] / l2_errors[0], 0.5) << "p=2 should give at least 2x reduction from p=1";
    EXPECT_LT(l2_errors[2] / l2_errors[1], 0.5) << "p=3 should give at least 2x reduction from p=2";
}

TEST_F(PRefinementTest, PolynomialSolutionExactness) {
    // For polynomial solutions of degree ≤ p, DG method of order p should be exact

    std::vector<int> orders = {2, 3, 4};

    for (int p : orders) {
        // Test with polynomial of degree p
        auto exact_poly = [p](const Eigen::Vector2d& x) -> double {
            double result = 0.0;
            for (int i = 0; i <= p; ++i) {
                for (int j = 0; j <= p; ++j) {
                    if (i + j <= p) {
                        result += std::pow(x[0], i) * std::pow(x[1], j);
                    }
                }
            }
            return result;
        };

        // Compute Laplacian analytically (will be polynomial of degree p-2)
        auto source = [p](const Eigen::Vector2d& x) -> double {
            double result = 0.0;
            for (int i = 2; i <= p; ++i) {
                for (int j = 0; j <= p; ++j) {
                    if (i + j <= p) {
                        result -= i * (i - 1) * std::pow(x[0], i - 2) * std::pow(x[1], j);
                    }
                }
            }
            for (int i = 0; i <= p; ++i) {
                for (int j = 2; j <= p; ++j) {
                    if (i + j <= p) {
                        result -= j * (j - 1) * std::pow(x[0], i) * std::pow(x[1], j - 2);
                    }
                }
            }
            return result;
        };

        auto exact_grad = [p](const Eigen::Vector2d& x) -> Eigen::Vector2d {
            Eigen::Vector2d grad = Eigen::Vector2d::Zero();
            for (int i = 1; i <= p; ++i) {
                for (int j = 0; j <= p; ++j) {
                    if (i + j <= p) {
                        grad[0] += i * std::pow(x[0], i - 1) * std::pow(x[1], j);
                    }
                }
            }
            for (int i = 0; i <= p; ++i) {
                for (int j = 1; j <= p; ++j) {
                    if (i + j <= p) {
                        grad[1] += j * std::pow(x[0], i) * std::pow(x[1], j - 1);
                    }
                }
            }
            return grad;
        };

        double h = 0.1;
        auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("triangle", p);
        mesh->initialize_dg_space(dg_space, 1);

        // Set BC matching exact solution
        auto bc_func = make_dirichlet_bc(exact_poly);
        mesh->set_boundary_condition("Bottom", bc_func);
        mesh->set_boundary_condition("Top", bc_func);
        mesh->set_boundary_condition("Left", bc_func);
        mesh->set_boundary_condition("Right", bc_func);

        LaplaceDGSolver solver(mesh, 10.0);
        Eigen::VectorXd solution = solver.solve(source);

        auto errors = solver.compute_error(exact_poly, exact_grad);

        // Should reproduce polynomial exactly (up to round-off and quadrature error)
        EXPECT_LT(errors["L2"], 1e-9)
            << "Order " << p << " DG should reproduce degree " << p << " polynomial exactly";
        EXPECT_LT(errors["H1"], 1e-8) << "Order " << p << " DG should reproduce degree " << p
                                      << " polynomial gradient exactly";
    }
}

TEST_F(PRefinementTest, HPRefinementCombination) {
    // Test that combining h and p refinement gives optimal convergence
    // Use smooth solution

    const double pi = M_PI;
    auto exact = [pi](const Eigen::Vector2d& x) -> double {
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_grad = [pi](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        Eigen::Vector2d grad;
        grad[0] = pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
        grad[1] = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
        return grad;
    };

    auto source = [pi](const Eigen::Vector2d& x) -> double {
        return 2.0 * pi * pi * std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    // Test: (h=0.2, p=1), (h=0.1, p=2), (h=0.05, p=3)
    std::vector<std::pair<double, int>> hp_pairs = {{0.2, 1}, {0.1, 2}, {0.05, 3}};
    std::vector<double> l2_errors;

    double penalty = 10.0;

    for (const auto& [h, p] : hp_pairs) {
        auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("triangle", p);
        mesh->initialize_dg_space(dg_space, 1);

        auto bc_zero = make_dirichlet_bc(0.0);
        mesh->set_boundary_condition("Bottom", bc_zero);
        mesh->set_boundary_condition("Top", bc_zero);
        mesh->set_boundary_condition("Left", bc_zero);
        mesh->set_boundary_condition("Right", bc_zero);

        LaplaceDGSolver solver(mesh, penalty);
        Eigen::VectorXd solution = solver.solve(source);

        auto errors = solver.compute_error(exact, exact_grad);
        l2_errors.push_back(errors["L2"]);
    }

    // HP refinement should give very fast convergence
    // Each step should reduce error significantly
    EXPECT_LT(l2_errors[1] / l2_errors[0], 0.1)
        << "HP refinement step 1 should reduce error by > 10x";
    EXPECT_LT(l2_errors[2] / l2_errors[1], 0.1)
        << "HP refinement step 2 should reduce error by > 10x";
}
