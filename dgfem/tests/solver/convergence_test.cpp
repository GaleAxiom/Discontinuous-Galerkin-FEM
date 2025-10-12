/**
 * @file convergence_test.cpp
 * @brief Tests for h-convergence rates of DG solver
 */

#include <gtest/gtest.h>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/utils/mesh_creation.hpp>
#include <dgfem/boundary/conditions.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

using namespace dgfem;

class ConvergenceRateTest : public ::testing::Test {
protected:
    void SetUp() override {
        MeshCreator::initialize_gmsh();
    }
    
    void TearDown() override {
        MeshCreator::finalize_gmsh();
    }
    
    /**
     * @brief Compute convergence rate from error at two mesh sizes
     * @param h1 First mesh size
     * @param err1 Error at h1
     * @param h2 Second mesh size
     * @param err2 Error at h2
     * @return Convergence rate
     */
    double compute_convergence_rate(double h1, double err1, double h2, double err2) {
        return std::log(err1 / err2) / std::log(h1 / h2);
    }
};

TEST_F(ConvergenceRateTest, HConvergenceTrianglesOrder1) {
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
    
    // Test on sequence of refined meshes
    std::vector<double> mesh_sizes = {0.2, 0.1, 0.05};
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;
    
    int order = 1;
    double penalty = 10.0;
    
    for (double h : mesh_sizes) {
        auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("triangle", order);
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
    
    // Compute convergence rates
    // For order p, expect L2 rate ≈ p+1 and H1 rate ≈ p
    double l2_rate_1 = compute_convergence_rate(mesh_sizes[0], l2_errors[0], 
                                                  mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 = compute_convergence_rate(mesh_sizes[1], l2_errors[1], 
                                                  mesh_sizes[2], l2_errors[2]);
    
    double h1_rate_1 = compute_convergence_rate(mesh_sizes[0], h1_errors[0], 
                                                  mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 = compute_convergence_rate(mesh_sizes[1], h1_errors[1], 
                                                  mesh_sizes[2], h1_errors[2]);
    
    // For order 1, expect L2 rate ≈ 2, H1 rate ≈ 1
    EXPECT_NEAR(l2_rate_1, 2.0, 0.3) << "L2 convergence rate should be ~2 for order 1";
    EXPECT_NEAR(l2_rate_2, 2.0, 0.3) << "L2 convergence rate should be consistent";
    
    EXPECT_NEAR(h1_rate_1, 1.0, 0.3) << "H1 convergence rate should be ~1 for order 1";
    EXPECT_NEAR(h1_rate_2, 1.0, 0.3) << "H1 convergence rate should be consistent";
    
    // Print rates for inspection
    std::cout << "Order 1 Triangles - L2 rates: " << l2_rate_1 << ", " << l2_rate_2 << std::endl;
    std::cout << "Order 1 Triangles - H1 rates: " << h1_rate_1 << ", " << h1_rate_2 << std::endl;
}

TEST_F(ConvergenceRateTest, HConvergenceTrianglesOrder2) {
    // Same manufactured solution
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
    
    std::vector<double> mesh_sizes = {0.15, 0.075, 0.0375};
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;
    
    int order = 2;
    double penalty = 10.0;
    
    for (double h : mesh_sizes) {
        auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("triangle", order);
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
        h1_errors.push_back(errors["H1"]);
    }
    
    double l2_rate_1 = compute_convergence_rate(mesh_sizes[0], l2_errors[0], 
                                                  mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 = compute_convergence_rate(mesh_sizes[1], l2_errors[1], 
                                                  mesh_sizes[2], l2_errors[2]);
    
    double h1_rate_1 = compute_convergence_rate(mesh_sizes[0], h1_errors[0], 
                                                  mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 = compute_convergence_rate(mesh_sizes[1], h1_errors[1], 
                                                  mesh_sizes[2], h1_errors[2]);
    
    // For order 2, expect L2 rate ≈ 3, H1 rate ≈ 2
    EXPECT_NEAR(l2_rate_1, 3.0, 0.5) << "L2 convergence rate should be ~3 for order 2";
    EXPECT_NEAR(l2_rate_2, 3.0, 0.5) << "L2 convergence rate should be consistent";
    
    EXPECT_NEAR(h1_rate_1, 2.0, 0.4) << "H1 convergence rate should be ~2 for order 2";
    EXPECT_NEAR(h1_rate_2, 2.0, 0.4) << "H1 convergence rate should be consistent";
    
    std::cout << "Order 2 Triangles - L2 rates: " << l2_rate_1 << ", " << l2_rate_2 << std::endl;
    std::cout << "Order 2 Triangles - H1 rates: " << h1_rate_1 << ", " << h1_rate_2 << std::endl;
}

TEST_F(ConvergenceRateTest, HConvergenceQuadsOrder2) {
    // Test quad elements
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
    
    std::vector<double> mesh_sizes = {0.5, 0.25, 0.125, 0.0625, 0.03125};
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;
    
    int order = 2;
    double penalty = 100000.0; // very high penalty for quads to get good convergence
    
    for (double h : mesh_sizes) {
        auto mesh = MeshCreator::create_rectangular_mesh(h, false, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("quad", order);
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
        h1_errors.push_back(errors["H1"]);
    }
    
    double l2_rate_1 = compute_convergence_rate(mesh_sizes[0], l2_errors[0], 
                                                  mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 = compute_convergence_rate(mesh_sizes[1], l2_errors[1], 
                                                  mesh_sizes[2], l2_errors[2]);
    double l2_rate_3 = compute_convergence_rate(mesh_sizes[2], l2_errors[2], 
                                                  mesh_sizes[3], l2_errors[3]);
    double l2_rate_4 = compute_convergence_rate(mesh_sizes[3], l2_errors[3], 
                                                  mesh_sizes[4], l2_errors[4]);
    
    double h1_rate_1 = compute_convergence_rate(mesh_sizes[0], h1_errors[0], 
                                                  mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 = compute_convergence_rate(mesh_sizes[1], h1_errors[1], 
                                                  mesh_sizes[2], h1_errors[2]);
    double h1_rate_3 = compute_convergence_rate(mesh_sizes[2], h1_errors[2],
                                                  mesh_sizes[3], h1_errors[3]);
    double h1_rate_4 = compute_convergence_rate(mesh_sizes[3], h1_errors[3],
                                                  mesh_sizes[4], h1_errors[4]);
    
    // For order 2 quads, expect L2 rate ≈ 3, H1 rate ≈ 2
    EXPECT_NEAR(l2_rate_1, 3.0, 0.5) << "L2 convergence rate should be ~3 for order 2 quads";
    EXPECT_NEAR(l2_rate_2, 3.0, 0.5) << "L2 convergence rate should be consistent";
    EXPECT_NEAR(l2_rate_3, 3.0, 0.5) << "L2 convergence rate should be consistent";
    EXPECT_NEAR(l2_rate_4, 3.0, 0.5) << "L2 convergence rate should be consistent";

    EXPECT_NEAR(h1_rate_1, 2.0, 0.4) << "H1 convergence rate should be ~2 for order 2 quads";
    EXPECT_NEAR(h1_rate_2, 2.0, 0.4) << "H1 convergence rate should be consistent";
    EXPECT_NEAR(h1_rate_3, 2.0, 0.4) << "H1 convergence rate should be consistent";
    EXPECT_NEAR(h1_rate_4, 2.0, 0.4) << "H1 convergence rate should be consistent";

    std::cout << "Order 2 Quads - L2 rates: " << l2_rate_1 << ", " << l2_rate_2 << ", " << l2_rate_3 << ", " << l2_rate_4 << std::endl;
    std::cout << "Order 2 Quads - H1 rates: " << h1_rate_1 << ", " << h1_rate_2 << ", " << h1_rate_3 << ", " << h1_rate_4 << std::endl;
}

TEST_F(ConvergenceRateTest, PolynomialReproductionOrder2) {
    // Test that order 2 exactly reproduces a quadratic polynomial
    // u = x²y + xy²
    auto exact = [](const Eigen::Vector2d& x) -> double {
        return x[0]*x[0]*x[1] + x[0]*x[1]*x[1];
    };
    
    auto exact_grad = [](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        Eigen::Vector2d grad;
        grad[0] = 2.0*x[0]*x[1] + x[1]*x[1];
        grad[1] = x[0]*x[0] + 2.0*x[0]*x[1];
        return grad;
    };
    
    // -Δu = -(2y + 2x) = -2(x+y)
    auto source = [](const Eigen::Vector2d& x) -> double {
        return -2.0 * (x[0] + x[1]);
    };
    
    // Test on a moderately fine mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.1, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);
    
    // Set Dirichlet BCs with exact solution values
    auto bc_bottom = make_dirichlet_bc([exact](const Eigen::Vector2d& x) { return exact(x); });
    auto bc_top = make_dirichlet_bc([exact](const Eigen::Vector2d& x) { return exact(x); });
    auto bc_left = make_dirichlet_bc([exact](const Eigen::Vector2d& x) { return exact(x); });
    auto bc_right = make_dirichlet_bc([exact](const Eigen::Vector2d& x) { return exact(x); });
    
    mesh->set_boundary_condition("Bottom", bc_bottom);
    mesh->set_boundary_condition("Top", bc_top);
    mesh->set_boundary_condition("Left", bc_left);
    mesh->set_boundary_condition("Right", bc_right);
    
    LaplaceDGSolver solver(mesh, 10.0);
    Eigen::VectorXd solution = solver.solve(source);
    
    auto errors = solver.compute_error(exact, exact_grad);
    
    // Order 2 should reproduce quadratic polynomials up to machine precision
    EXPECT_LT(errors["L2"], 2e-5) << "Order 2 should reproduce quadratic polynomials exactly";
    EXPECT_LT(errors["H1"], 5e-3) << "Gradients should also be reproduced exactly";
}
