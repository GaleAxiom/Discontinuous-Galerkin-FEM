#include <Eigen/Dense>
#include <cmath>
#include <dgfem/basis/monomial.hpp>
#include <dgfem/reference/elements.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

TEST(MonomialBasisTriangleTest, ThrowsErrorForInvalidOrder) {
    EXPECT_THROW(MonomialBasisTriangle(0), std::invalid_argument);  // Below minimum
    EXPECT_THROW(MonomialBasisTriangle(11),
                 std::invalid_argument);  // Above maximum (MAX_ORDER = 10)
}

TEST(MonomialBasisTriangleTest, ValidOrdersDoNotThrow) {
    for (int order = 1; order <= 10; ++order) {
        EXPECT_NO_THROW(MonomialBasisTriangle basis(order));
    }
}

TEST(MonomialBasisTriangleTest, CorrectBasisCount) {
    // For triangular elements, the number of basis functions is (order + 1)(order + 2)/2
    for (int order = 1; order <= 7; ++order) {
        MonomialBasisTriangle basis(order);
        EXPECT_EQ(basis.get_n_basis(), (order + 1) * (order + 2) / 2);
    }
}

TEST(MonomialBasisTriangleTest, EvaluateAtPoint) {
    MonomialBasisTriangle basis(1);  // Test with linear basis
    Eigen::Vector2d point(2.0, 3.0);

    Eigen::VectorXd values = basis.evaluate(point);

    // For linear basis, we should have 3 functions: 1, y, x
    EXPECT_EQ(values.size(), 3);

    // Check values:
    // f_0 = 1
    // f_1 = y
    // f_2 = x
    EXPECT_NEAR(values(0), 1.0, 1e-12);
    EXPECT_NEAR(values(1), 3.0, 1e-12);
    EXPECT_NEAR(values(2), 2.0, 1e-12);
}

TEST(MonomialBasisTriangleTest, EvaluateGradientAtOrigin) {
    MonomialBasisTriangle basis(2);  // Test with quadratic basis
    Eigen::Vector2d origin(0.0, 0.0);

    Eigen::MatrixXd gradient = basis.evaluate_gradient(origin);

    // Size should be 6x2 for quadratic basis
    EXPECT_EQ(gradient.rows(), 6);
    EXPECT_EQ(gradient.cols(), 2);

    // At origin, assuming basis order 1, y, x, y^2, xy, x^2
    // - Constant term (1) has zero gradient
    EXPECT_NEAR(gradient(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(gradient(0, 1), 0.0, 1e-12);

    // - Linear terms (y, x) have constant gradients
    EXPECT_NEAR(gradient(1, 0), 0.0, 1e-12);  // d/dx(y) = 0
    EXPECT_NEAR(gradient(1, 1), 1.0, 1e-12);  // d/dy(y) = 1
    EXPECT_NEAR(gradient(2, 0), 1.0, 1e-12);  // d/dx(x) = 1
    EXPECT_NEAR(gradient(2, 1), 0.0, 1e-12);  // d/dy(x) = 0

    // - Quadratic terms (y², xy, x²) have zero gradients at origin
    for (int i = 3; i < gradient.rows(); ++i) {
        EXPECT_NEAR(gradient(i, 0), 0.0, 1e-12);
        EXPECT_NEAR(gradient(i, 1), 0.0, 1e-12);
    }
}

TEST(MonomialBasisTriangleTest, EvaluateGradientAtPoint) {
    MonomialBasisTriangle basis(2);  // Test with quadratic basis
    Eigen::Vector2d point(2.0, 3.0);

    Eigen::MatrixXd gradient = basis.evaluate_gradient(point);

    // Test a few known derivatives at (2,3):
    // d/dx(y²) = 0
    // d/dy(y²) = 2y = 6
    // d/dx(xy) = y = 3
    // d/dy(xy) = x = 2
    // d/dx(x²) = 2x = 4
    // d/dy(x²) = 0

    // These indices assume the basis ordering: 1, y, x, y², xy, x²
    EXPECT_NEAR(gradient(3, 0), 0.0, 1e-12);  // d/dx(y²)
    EXPECT_NEAR(gradient(3, 1), 6.0, 1e-12);  // d/dy(y²)
    EXPECT_NEAR(gradient(4, 0), 3.0, 1e-12);  // d/dx(xy)
    EXPECT_NEAR(gradient(4, 1), 2.0, 1e-12);  // d/dy(xy)
    EXPECT_NEAR(gradient(5, 0), 4.0, 1e-12);  // d/dx(x²)
    EXPECT_NEAR(gradient(5, 1), 0.0, 1e-12);  // d/dy(x²)
}

// Test polynomial completeness - monomials should span all polynomials of degree ≤ p
TEST(MonomialBasisTriangleTest, PolynomialCompleteness) {
    MonomialBasisTriangle basis(2);  // Quadratic basis

    // Test that basis can represent any quadratic polynomial exactly
    // Example: p(x,y) = 3 + 2y + 4x - y² + 2xy + 3x²

    // The coefficients should directly match the monomial basis (1, y, x, y², xy, x²)
    Eigen::VectorXd expected_coeffs(6);
    expected_coeffs << 3.0, 2.0, 4.0, -1.0, 2.0, 3.0;

    // Test at several points
    std::vector<Eigen::Vector2d> test_points = {
        Eigen::Vector2d(0.1, 0.2), Eigen::Vector2d(0.5, 0.3), Eigen::Vector2d(0.2, 0.6)};

    for (const auto& pt : test_points) {
        // Exact value
        double x = pt(0), y = pt(1);
        double exact = 3.0 + 2.0 * y + 4.0 * x - y * y + 2.0 * x * y + 3.0 * x * x;

        // Approximation via basis
        Eigen::VectorXd phi = basis.evaluate(pt);
        double approx = expected_coeffs.dot(phi);

        EXPECT_NEAR(approx, exact, 1e-12)
            << "Polynomial representation failed at (" << x << ", " << y << ")";
    }
}

// Test high-order evaluation
TEST(MonomialBasisTriangleTest, HighOrderEvaluation) {
    MonomialBasisTriangle basis(3);  // Cubic basis

    // Should have 10 basis functions: (3+1)(3+2)/2 = 10
    EXPECT_EQ(basis.get_n_basis(), 10);

    Eigen::Vector2d test_point(0.4, 0.3);
    Eigen::VectorXd values = basis.evaluate(test_point);

    EXPECT_EQ(values.size(), 10);
    EXPECT_TRUE(values.allFinite());

    // Test specific cubic monomials
    double x = test_point(0), y = test_point(1);

    // Ordering: 1, y, x, y², xy, x², y³, xy², x²y, x³
    EXPECT_NEAR(values(0), 1.0, 1e-12);
    EXPECT_NEAR(values(1), y, 1e-12);
    EXPECT_NEAR(values(2), x, 1e-12);
    EXPECT_NEAR(values(6), y * y * y, 1e-12);  // y³
    EXPECT_NEAR(values(9), x * x * x, 1e-12);  // x³
}

// Test numerical gradient verification
TEST(MonomialBasisTriangleTest, NumericalGradientVerification) {
    MonomialBasisTriangle basis(2);
    Eigen::Vector2d test_point(0.35, 0.45);
    double h = 1e-7;

    Eigen::MatrixXd grad_analytical = basis.evaluate_gradient(test_point);

    int n_basis = basis.get_n_basis();
    Eigen::MatrixXd grad_numerical(n_basis, 2);

    for (int i = 0; i < n_basis; ++i) {
        // x-derivative
        double val_plus_x = basis.evaluate(test_point + Eigen::Vector2d(h, 0))(i);
        double val_minus_x = basis.evaluate(test_point - Eigen::Vector2d(h, 0))(i);
        grad_numerical(i, 0) = (val_plus_x - val_minus_x) / (2 * h);

        // y-derivative
        double val_plus_y = basis.evaluate(test_point + Eigen::Vector2d(0, h))(i);
        double val_minus_y = basis.evaluate(test_point - Eigen::Vector2d(0, h))(i);
        grad_numerical(i, 1) = (val_plus_y - val_minus_y) / (2 * h);
    }

    for (int i = 0; i < n_basis; ++i) {
        EXPECT_NEAR(grad_analytical(i, 0), grad_numerical(i, 0), 1e-5)
            << "x-gradient mismatch for basis " << i;
        EXPECT_NEAR(grad_analytical(i, 1), grad_numerical(i, 1), 1e-5)
            << "y-gradient mismatch for basis " << i;
    }
}
