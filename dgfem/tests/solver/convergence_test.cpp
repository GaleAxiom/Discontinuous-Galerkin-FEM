/**
 * @file convergence_test.cpp
 * @brief Tests for h-convergence rates of DG solver
 */

#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <vector>

#include <gtest/gtest.h>

using namespace dgfem;

class ConvergenceRateTest : public ::testing::Test {
protected:
    void SetUp() override { MeshCreator::initialize_gmsh(); }

    void TearDown() override { MeshCreator::finalize_gmsh(); }

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
    auto exact = [pi](const Vec2& x) -> double {
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_grad = [pi](const Vec2& x) -> Vec2 {
        Vec2 grad;
        grad[0] = pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
        grad[1] = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
        return grad;
    };

    auto source = [pi](const Vec2& x) -> double {
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
        DView1 solution = solver.solve(source);

        auto errors = solver.compute_error(exact, exact_grad);
        l2_errors.push_back(errors["L2"]);
        h1_errors.push_back(errors["H1"]);
    }

    // Compute convergence rates
    // For order p, expect L2 rate ≈ p+1 and H1 rate ≈ p
    double l2_rate_1 =
        compute_convergence_rate(mesh_sizes[0], l2_errors[0], mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 =
        compute_convergence_rate(mesh_sizes[1], l2_errors[1], mesh_sizes[2], l2_errors[2]);

    double h1_rate_1 =
        compute_convergence_rate(mesh_sizes[0], h1_errors[0], mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 =
        compute_convergence_rate(mesh_sizes[1], h1_errors[1], mesh_sizes[2], h1_errors[2]);

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
    auto exact = [pi](const Vec2& x) -> double {
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_grad = [pi](const Vec2& x) -> Vec2 {
        Vec2 grad;
        grad[0] = pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
        grad[1] = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
        return grad;
    };

    auto source = [pi](const Vec2& x) -> double {
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
        DView1 solution = solver.solve(source);

        auto errors = solver.compute_error(exact, exact_grad);
        l2_errors.push_back(errors["L2"]);
        h1_errors.push_back(errors["H1"]);
    }

    double l2_rate_1 =
        compute_convergence_rate(mesh_sizes[0], l2_errors[0], mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 =
        compute_convergence_rate(mesh_sizes[1], l2_errors[1], mesh_sizes[2], l2_errors[2]);

    double h1_rate_1 =
        compute_convergence_rate(mesh_sizes[0], h1_errors[0], mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 =
        compute_convergence_rate(mesh_sizes[1], h1_errors[1], mesh_sizes[2], h1_errors[2]);

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
    auto exact = [pi](const Vec2& x) -> double {
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_grad = [pi](const Vec2& x) -> Vec2 {
        Vec2 grad;
        grad[0] = pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
        grad[1] = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
        return grad;
    };

    auto source = [pi](const Vec2& x) -> double {
        return 2.0 * pi * pi * std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    std::vector<double> mesh_sizes = {0.5, 0.25, 0.125, 0.0625, 0.03125};
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;

    int order = 2;
    double penalty = 100000.0;  // very high penalty for quads to get good convergence

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
        DView1 solution = solver.solve(source);

        auto errors = solver.compute_error(exact, exact_grad);
        l2_errors.push_back(errors["L2"]);
        h1_errors.push_back(errors["H1"]);
    }

    double l2_rate_1 =
        compute_convergence_rate(mesh_sizes[0], l2_errors[0], mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 =
        compute_convergence_rate(mesh_sizes[1], l2_errors[1], mesh_sizes[2], l2_errors[2]);
    double l2_rate_3 =
        compute_convergence_rate(mesh_sizes[2], l2_errors[2], mesh_sizes[3], l2_errors[3]);
    double l2_rate_4 =
        compute_convergence_rate(mesh_sizes[3], l2_errors[3], mesh_sizes[4], l2_errors[4]);

    double h1_rate_1 =
        compute_convergence_rate(mesh_sizes[0], h1_errors[0], mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 =
        compute_convergence_rate(mesh_sizes[1], h1_errors[1], mesh_sizes[2], h1_errors[2]);
    double h1_rate_3 =
        compute_convergence_rate(mesh_sizes[2], h1_errors[2], mesh_sizes[3], h1_errors[3]);
    double h1_rate_4 =
        compute_convergence_rate(mesh_sizes[3], h1_errors[3], mesh_sizes[4], h1_errors[4]);

    // For order 2 quads, expect L2 rate ≈ 3, H1 rate ≈ 2
    EXPECT_NEAR(l2_rate_1, 3.0, 0.5) << "L2 convergence rate should be ~3 for order 2 quads";
    EXPECT_NEAR(l2_rate_2, 3.0, 0.5) << "L2 convergence rate should be consistent";
    EXPECT_NEAR(l2_rate_3, 3.0, 0.5) << "L2 convergence rate should be consistent";
    EXPECT_NEAR(l2_rate_4, 3.0, 0.5) << "L2 convergence rate should be consistent";

    EXPECT_NEAR(h1_rate_1, 2.0, 0.4) << "H1 convergence rate should be ~2 for order 2 quads";
    EXPECT_NEAR(h1_rate_2, 2.0, 0.4) << "H1 convergence rate should be consistent";
    EXPECT_NEAR(h1_rate_3, 2.0, 0.4) << "H1 convergence rate should be consistent";
    EXPECT_NEAR(h1_rate_4, 2.0, 0.4) << "H1 convergence rate should be consistent";

    std::cout << "Order 2 Quads - L2 rates: " << l2_rate_1 << ", " << l2_rate_2 << ", " << l2_rate_3
              << ", " << l2_rate_4 << std::endl;
    std::cout << "Order 2 Quads - H1 rates: " << h1_rate_1 << ", " << h1_rate_2 << ", " << h1_rate_3
              << ", " << h1_rate_4 << std::endl;
}

TEST_F(ConvergenceRateTest, PolynomialReproductionOrder2) {
    // Test that order 2 exactly reproduces a quadratic polynomial.
    // u = x^2 + xy + y^2 -- every term has degree <= 2, so it's exactly representable by a
    // complete P2 space, unlike x^2*y + x*y^2 (degree 3 in every term), which this test used
    // to test under the same "exact reproduction" claim.
    auto exact = [](const Vec2& x) -> double { return x[0] * x[0] + x[0] * x[1] + x[1] * x[1]; };

    auto exact_grad = [](const Vec2& x) -> Vec2 {
        Vec2 grad;
        grad[0] = 2.0 * x[0] + x[1];
        grad[1] = x[0] + 2.0 * x[1];
        return grad;
    };

    // Δu = 2 + 2 = 4, so -Δu = -4.
    auto source = [](const Vec2& /*x*/) -> double { return -4.0; };

    // Test on a moderately fine mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.1, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);

    // Set Dirichlet BCs with exact solution values
    auto bc_bottom = make_dirichlet_bc([exact](const Vec2& x) { return exact(x); });
    auto bc_top = make_dirichlet_bc([exact](const Vec2& x) { return exact(x); });
    auto bc_left = make_dirichlet_bc([exact](const Vec2& x) { return exact(x); });
    auto bc_right = make_dirichlet_bc([exact](const Vec2& x) { return exact(x); });

    mesh->set_boundary_condition("Bottom", bc_bottom);
    mesh->set_boundary_condition("Top", bc_top);
    mesh->set_boundary_condition("Left", bc_left);
    mesh->set_boundary_condition("Right", bc_right);

    LaplaceDGSolver solver(mesh, 10.0);
    DView1 solution = solver.solve(source);

    auto errors = solver.compute_error(exact, exact_grad);

    // Order 2 should reproduce quadratic polynomials up to machine precision (observed
    // L2 ~ 1e-13, H1 ~ 1e-12 on this mesh; tolerances below keep several orders of margin).
    EXPECT_LT(errors["L2"], 1e-10) << "Order 2 should reproduce quadratic polynomials exactly";
    EXPECT_LT(errors["H1"], 1e-9) << "Gradients should also be reproduced exactly";
}

namespace {
// Shared manufactured solution for the mixed-BC convergence tests below:
// u = sin(pi*x)*sin(pi*y) + x + y. The added "+ x + y" term is harmonic (Delta(x+y) = 0), so
// the source term -Delta(u) = 2*pi^2*sin(pi*x)*sin(pi*y) is unchanged from the pure-Dirichlet
// convergence tests above, but the boundary traces and normal derivatives are now all nonzero
// and vary along each edge -- unlike sin(pi*x)*sin(pi*y) alone, which vanishes identically on
// the whole boundary and so can't exercise a genuinely nonzero, non-constant Neumann/Robin
// datum. This gives an honest correctness check: a sign or scaling bug in the Neumann/Robin
// boundary-integral terms would show up as a wrong asymptotic solution (an O(1) error that
// does not shrink with h), not just a slightly-off convergence rate.
double mixed_bc_exact(const Vec2& x) {
    return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]) + x[0] + x[1];
}

Vec2 mixed_bc_exact_grad(const Vec2& x) {
    Vec2 grad;
    grad[0] = M_PI * std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]) + 1.0;
    grad[1] = M_PI * std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]) + 1.0;
    return grad;
}

double mixed_bc_source(const Vec2& x) {
    return 2.0 * M_PI * M_PI * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
}

// du/dn on the Bottom edge (y = 0, outward normal (0,-1)): -d/dy[u](x, 0).
double mixed_bc_neumann_bottom(const Vec2& x) {
    return -(M_PI * std::sin(M_PI * x[0]) + 1.0);
}

// du/dn on the Top edge (y = 1, outward normal (0,1)): d/dy[u](x, 1).
double mixed_bc_flux_top(double x0) {
    return -M_PI * std::sin(M_PI * x0) + 1.0;
}
}  // namespace

TEST_F(ConvergenceRateTest, HConvergenceMixedDirichletNeumann) {
    // Dirichlet on Left/Right/Top (using the exact trace), Neumann on Bottom (using the exact,
    // position-dependent outward flux). If compute_boundary_rhs_integral's NEUMANN branch had
    // the wrong sign, magnitude, or dropped the position dependence, the solver would converge
    // to the wrong function and the L2/H1 rates below would not match theory.
    std::vector<double> mesh_sizes = {0.2, 0.1, 0.05};
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;

    int order = 1;
    double penalty = 10.0;

    for (double h : mesh_sizes) {
        auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("triangle", order);
        mesh->initialize_dg_space(dg_space, 1);

        mesh->set_boundary_condition("Left", make_dirichlet_bc(mixed_bc_exact));
        mesh->set_boundary_condition("Right", make_dirichlet_bc(mixed_bc_exact));
        mesh->set_boundary_condition("Top", make_dirichlet_bc(mixed_bc_exact));
        mesh->set_boundary_condition("Bottom", make_neumann_bc(mixed_bc_neumann_bottom));

        LaplaceDGSolver solver(mesh, penalty);
        DView1 solution = solver.solve(mixed_bc_source);

        auto errors = solver.compute_error(mixed_bc_exact, mixed_bc_exact_grad);
        l2_errors.push_back(errors["L2"]);
        h1_errors.push_back(errors["H1"]);
    }

    double l2_rate_1 =
        compute_convergence_rate(mesh_sizes[0], l2_errors[0], mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 =
        compute_convergence_rate(mesh_sizes[1], l2_errors[1], mesh_sizes[2], l2_errors[2]);

    double h1_rate_1 =
        compute_convergence_rate(mesh_sizes[0], h1_errors[0], mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 =
        compute_convergence_rate(mesh_sizes[1], h1_errors[1], mesh_sizes[2], h1_errors[2]);

    // For order 1, expect L2 rate ~ 2, H1 rate ~ 1, same as the pure-Dirichlet case.
    EXPECT_NEAR(l2_rate_1, 2.0, 0.3) << "L2 convergence rate should be ~2 for order 1";
    EXPECT_NEAR(l2_rate_2, 2.0, 0.3) << "L2 convergence rate should be consistent";

    EXPECT_NEAR(h1_rate_1, 1.0, 0.3) << "H1 convergence rate should be ~1 for order 1";
    EXPECT_NEAR(h1_rate_2, 1.0, 0.3) << "H1 convergence rate should be consistent";

    std::cout << "Mixed Dirichlet/Neumann - L2 rates: " << l2_rate_1 << ", " << l2_rate_2
              << std::endl;
    std::cout << "Mixed Dirichlet/Neumann - H1 rates: " << h1_rate_1 << ", " << h1_rate_2
              << std::endl;
}

TEST_F(ConvergenceRateTest, HConvergenceMixedDirichletRobin) {
    // Dirichlet on Left/Right/Bottom, Robin (alpha*u + du/dn = g) on Top with alpha = 1.5 and
    // the exact g = alpha*u + du/dn. Both the alpha*u mass-matrix term and the g RHS term are
    // nonzero and vary along the edge here, so this exercises the full Robin implementation
    // (compute_boundary_face_integral's ROBIN branch as well as the RHS), not just the special
    // case alpha = 0 that degenerates to Neumann.
    const double alpha = 1.5;
    auto robin_top = [alpha](const Vec2& x) {
        return alpha * mixed_bc_exact(Vec2{x[0], 1.0}) + mixed_bc_flux_top(x[0]);
    };

    std::vector<double> mesh_sizes = {0.2, 0.1, 0.05};
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;

    int order = 1;
    double penalty = 10.0;

    for (double h : mesh_sizes) {
        auto mesh = MeshCreator::create_rectangular_mesh(h, true, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = std::make_shared<DGSpace>("triangle", order);
        mesh->initialize_dg_space(dg_space, 1);

        mesh->set_boundary_condition("Left", make_dirichlet_bc(mixed_bc_exact));
        mesh->set_boundary_condition("Right", make_dirichlet_bc(mixed_bc_exact));
        mesh->set_boundary_condition("Bottom", make_dirichlet_bc(mixed_bc_exact));
        mesh->set_boundary_condition("Top", make_robin_bc(robin_top, alpha));

        LaplaceDGSolver solver(mesh, penalty);
        DView1 solution = solver.solve(mixed_bc_source);

        auto errors = solver.compute_error(mixed_bc_exact, mixed_bc_exact_grad);
        l2_errors.push_back(errors["L2"]);
        h1_errors.push_back(errors["H1"]);
    }

    double l2_rate_1 =
        compute_convergence_rate(mesh_sizes[0], l2_errors[0], mesh_sizes[1], l2_errors[1]);
    double l2_rate_2 =
        compute_convergence_rate(mesh_sizes[1], l2_errors[1], mesh_sizes[2], l2_errors[2]);

    double h1_rate_1 =
        compute_convergence_rate(mesh_sizes[0], h1_errors[0], mesh_sizes[1], h1_errors[1]);
    double h1_rate_2 =
        compute_convergence_rate(mesh_sizes[1], h1_errors[1], mesh_sizes[2], h1_errors[2]);

    EXPECT_NEAR(l2_rate_1, 2.0, 0.3) << "L2 convergence rate should be ~2 for order 1";
    EXPECT_NEAR(l2_rate_2, 2.0, 0.3) << "L2 convergence rate should be consistent";

    EXPECT_NEAR(h1_rate_1, 1.0, 0.3) << "H1 convergence rate should be ~1 for order 1";
    EXPECT_NEAR(h1_rate_2, 1.0, 0.3) << "H1 convergence rate should be consistent";

    std::cout << "Mixed Dirichlet/Robin - L2 rates: " << l2_rate_1 << ", " << l2_rate_2
              << std::endl;
    std::cout << "Mixed Dirichlet/Robin - H1 rates: " << h1_rate_1 << ", " << h1_rate_2
              << std::endl;
}
