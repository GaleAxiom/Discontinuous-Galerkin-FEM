/**
 * @file basis_typed_tests.hpp
 * @brief Typed test suite for common basis function tests
 *
 * This header provides a template for testing all basis function implementations
 * with common property tests, reducing code duplication.
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>

#include <memory>

#include <gtest/gtest.h>

namespace dgfem {
namespace test {

/**
 * @brief Typed test fixture for basis functions
 * @tparam BasisType The basis function class to test
 *
 * Usage:
 * ```cpp
 * using BasisTypes = ::testing::Types<LegendreBasis, DubinerBasis, MonomialBasisTriangle>;
 * TYPED_TEST_SUITE(BasisPropertyTest, BasisTypes);
 *
 * TYPED_TEST(BasisPropertyTest, OrthogonalityProperty) {
 *     // Test implementation using TypeParam
 * }
 * ```
 */
template <typename BasisType>
class BasisPropertyTest : public ::testing::Test {
protected:
    // Helper method to create basis of given order
    virtual std::unique_ptr<BasisType> createBasis(int order) {
        return std::make_unique<BasisType>(order);
    }

    // Helper to get reference element for this basis type
    virtual std::string getElementType() const = 0;

    // Helper to get maximum supported order
    virtual int getMaxOrder() const { return 5; }

    // Test that basis count is correct for all orders
    void testBasisCount() {
        for (int order = 1; order <= getMaxOrder(); ++order) {
            auto basis = createBasis(order);
            int n_basis = basis->compute_n_basis();
            int expected = expectedBasisCount(order);
            EXPECT_EQ(n_basis, expected) << "Basis count mismatch for order " << order;
        }
    }

    // Test numerical gradient verification
    void testNumericalGradient(int order = 2) {
        auto basis = createBasis(order);
        int n_basis = basis->compute_n_basis();

        // Test at a few points
        std::vector<Eigen::Vector2d> test_points = getTestPoints();

        for (const auto& x : test_points) {
            Eigen::VectorXd values = basis->evaluate(x);
            auto gradients = basis->evaluate_grad(x);

            double h = 1e-7;

            for (int i = 0; i < n_basis; ++i) {
                // Numerical derivative in x
                Eigen::Vector2d x_plus_h = x;
                x_plus_h[0] += h;
                Eigen::VectorXd values_plus = basis->evaluate(x_plus_h);
                double numerical_dx = (values_plus[i] - values[i]) / h;

                EXPECT_NEAR(gradients[i][0], numerical_dx, 1e-5)
                    << "Gradient mismatch in x for basis " << i;

                // Numerical derivative in y
                Eigen::Vector2d y_plus_h = x;
                y_plus_h[1] += h;
                Eigen::VectorXd values_plus_y = basis->evaluate(y_plus_h);
                double numerical_dy = (values_plus_y[i] - values[i]) / h;

                EXPECT_NEAR(gradients[i][1], numerical_dy, 1e-5)
                    << "Gradient mismatch in y for basis " << i;
            }
        }
    }

    // Test polynomial completeness
    void testPolynomialCompleteness(int order = 2) {
        auto basis = createBasis(order);
        int n_basis = basis->compute_n_basis();

        // Every monomial of degree ≤ order should be representable
        // This is a weaker test - just check that we have enough basis functions
        int min_required = expectedMonomialCount(order);
        EXPECT_GE(n_basis, min_required)
            << "Not enough basis functions for polynomial completeness";
    }

    // Test that basis functions are well-defined at boundary
    void testBoundaryValues(int order = 2) {
        auto basis = createBasis(order);

        for (const auto& boundary_point : getBoundaryPoints()) {
            Eigen::VectorXd values = basis->evaluate(boundary_point);

            // All values should be finite
            for (int i = 0; i < values.size(); ++i) {
                EXPECT_TRUE(std::isfinite(values[i]))
                    << "Basis function " << i << " not finite at boundary point ("
                    << boundary_point[0] << ", " << boundary_point[1] << ")";
            }
        }
    }

protected:
    // Virtual methods to be implemented by specific basis types
    virtual int expectedBasisCount(int order) const = 0;
    virtual int expectedMonomialCount(int order) const = 0;
    virtual std::vector<Eigen::Vector2d> getTestPoints() const = 0;
    virtual std::vector<Eigen::Vector2d> getBoundaryPoints() const = 0;
};

/**
 * @brief Specialization for orthogonal basis tests
 */
template <typename OrthogonalBasisType>
class OrthogonalBasisTest : public BasisPropertyTest<OrthogonalBasisType> {
protected:
    // Test orthogonality/orthonormality
    void testOrthogonality(int order = 2, double tolerance = 1e-10) {
        auto basis = this->createBasis(order);
        int n_basis = basis->compute_n_basis();

        // Get quadrature for integration
        auto quad = getQuadratureRule(order);

        // Compute Gram matrix: G_ij = ∫ φ_i φ_j dx
        Eigen::MatrixXd gram(n_basis, n_basis);
        gram.setZero();

        for (size_t q = 0; q < quad.points.size(); ++q) {
            Eigen::VectorXd phi = basis->evaluate(quad.points[q]);
            double w = quad.weights[q];

            for (int i = 0; i < n_basis; ++i) {
                for (int j = 0; j < n_basis; ++j) {
                    gram(i, j) += w * phi[i] * phi[j];
                }
            }
        }

        // Check if orthogonal or orthonormal
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                if (i == j) {
                    // Diagonal should be 1 for orthonormal, positive for orthogonal
                    EXPECT_GT(gram(i, j), 0.0)
                        << "Diagonal entry (" << i << "," << j << ") not positive";
                } else {
                    // Off-diagonal should be zero
                    EXPECT_NEAR(gram(i, j), 0.0, tolerance)
                        << "Off-diagonal entry (" << i << "," << j << ") not zero";
                }
            }
        }
    }

    virtual struct QuadratureRule {
        std::vector<Eigen::Vector2d> points;
        std::vector<double> weights;
    } getQuadratureRule(int order) const = 0;
};

/**
 * @brief Test suite for convergence properties
 */
template <typename SolverType>
class ConvergencePropertyTest : public ::testing::Test {
protected:
    /**
     * @brief Test h-refinement convergence rate
     */
    void testHConvergence(std::function<double(const Eigen::Vector2d&)> exact_solution,
                          std::function<Eigen::Vector2d(const Eigen::Vector2d&)> exact_gradient,
                          std::function<double(const Eigen::Vector2d&)> source_term,
                          int polynomial_order, double expected_l2_rate, double expected_h1_rate,
                          double tolerance = 0.3  // Allow 30% deviation from theoretical rate
    ) {
        std::vector<double> mesh_sizes = {0.2, 0.1, 0.05};
        std::vector<double> l2_errors;
        std::vector<double> h1_errors;

        for (double h : mesh_sizes) {
            auto [l2_err, h1_err] = solveAndComputeError(h, polynomial_order, exact_solution,
                                                         exact_gradient, source_term);
            l2_errors.push_back(l2_err);
            h1_errors.push_back(h1_err);
        }

        // Compute convergence rates
        double l2_rate =
            computeConvergenceRate(mesh_sizes[0], l2_errors[0], mesh_sizes[1], l2_errors[1]);

        double h1_rate =
            computeConvergenceRate(mesh_sizes[0], h1_errors[0], mesh_sizes[1], h1_errors[1]);

        EXPECT_NEAR(l2_rate, expected_l2_rate, tolerance)
            << "L2 convergence rate " << l2_rate << " differs from expected " << expected_l2_rate;

        EXPECT_NEAR(h1_rate, expected_h1_rate, tolerance)
            << "H1 convergence rate " << h1_rate << " differs from expected " << expected_h1_rate;
    }

protected:
    virtual std::pair<double, double>
    solveAndComputeError(double h, int order, std::function<double(const Eigen::Vector2d&)> exact,
                         std::function<Eigen::Vector2d(const Eigen::Vector2d&)> grad,
                         std::function<double(const Eigen::Vector2d&)> source) = 0;

    double computeConvergenceRate(double h1, double err1, double h2, double err2) {
        return std::log(err1 / err2) / std::log(h1 / h2);
    }
};

}  // namespace test
}  // namespace dgfem
