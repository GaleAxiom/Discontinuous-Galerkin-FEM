#include <cmath>
#include <dgfem/basis/dubiner.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/quadrature/factory.hpp>
#include <dgfem/reference/elements.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

TEST(DubinerBasisTest, ThrowsErrorForInvalidOrder) {
    EXPECT_THROW(DubinerBasis(0), std::invalid_argument);   // Below minimum
    EXPECT_THROW(DubinerBasis(-1), std::invalid_argument);  // Negative
    EXPECT_THROW(DubinerBasis(-5), std::invalid_argument);  // Negative
    EXPECT_THROW(DubinerBasis(11), std::invalid_argument);  // Above maximum (MAX_ORDER = 10)
}

TEST(DubinerBasisTest, ValidOrdersDoNotThrow) {
    for (int order = 1; order <= 10; ++order) {
        EXPECT_NO_THROW(DubinerBasis basis(order));
    }
}

TEST(DubinerBasisTest, CorrectBasisCount) {
    // For triangular elements, the number of basis functions is (order + 1)(order + 2)/2
    for (int order = 1; order <= 7; ++order) {
        DubinerBasis basis(order);
        EXPECT_EQ(basis.get_n_basis(), (order + 1) * (order + 2) / 2);
    }
}

TEST(DubinerBasisTest, EvaluateAtVertex) {
    DubinerBasis basis(2);  // Test with quadratic basis

    // Test at actual triangle vertices in reference coordinates
    // Reference triangle: (0,0), (1,0), (0,1)

    // Test at origin (0, 0)
    Vec2 vertex1{0.0, 0.0};
    DView1 values1 = basis.evaluate(vertex1);
    EXPECT_EQ(values1.size(), 6);  // (2+1)(2+2)/2 = 6
    EXPECT_TRUE(all_finite(values1));

    // First basis function should be non-zero at all points (it's constant-like)
    EXPECT_NE(values1(0), 0.0);

    // Test at (1, 0)
    Vec2 vertex2{1.0, 0.0};
    DView1 values2 = basis.evaluate(vertex2);
    EXPECT_EQ(values2.size(), 6);
    EXPECT_TRUE(all_finite(values2));

    // Test at (0, 1) - the singular point in collapsed coordinates
    Vec2 vertex3{0.0, 1.0};
    DView1 values3 = basis.evaluate(vertex3);
    EXPECT_EQ(values3.size(), 6);
    EXPECT_TRUE(all_finite(values3));  // Critical: must handle singularity
}

TEST(DubinerBasisTest, EvaluateGradientAtOrigin) {
    DubinerBasis basis(1);  // Test with linear basis
    Vec2 origin{0.0, 0.0};

    DView2 gradient = basis.evaluate_gradient(origin);

    // Size should be 3x2 for linear basis on triangle
    EXPECT_EQ(gradient.extent(0), 3);
    EXPECT_EQ(gradient.extent(1), 2);

    // All values should be finite
    EXPECT_TRUE(all_finite(gradient));

    // For linear basis on triangle, gradients should be constant
    // First basis function (constant) should have zero gradient
    EXPECT_NEAR(gradient(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(gradient(0, 1), 0.0, 1e-12);

    // Other basis functions should have non-zero gradients
    // (exact values depend on normalization, but should be finite and reasonable)
    for (int i = 1; i < 3; ++i) {
        double grad_magnitude =
            std::sqrt(gradient(i, 0) * gradient(i, 0) + gradient(i, 1) * gradient(i, 1));
        EXPECT_GT(grad_magnitude, 0.0);
        EXPECT_LT(grad_magnitude, 100.0);  // Reasonable bounds due to normalization
    }
}

TEST(DubinerBasisTest, SingularityHandling) {
    DubinerBasis basis(2);

    // Test at top vertex (0, 1) where collapsed coordinates are singular
    Vec2 top_vertex{0.0, 1.0};

    // Should not throw and produce finite values
    EXPECT_NO_THROW({
        DView1 values = basis.evaluate(top_vertex);
        EXPECT_TRUE(all_finite(values));

        DView2 gradient = basis.evaluate_gradient(top_vertex);
        EXPECT_TRUE(all_finite(gradient));
    });
}

TEST(DubinerBasisTest, NormalizationFactors) {
    DubinerBasis basis(1);  // Test with linear basis

    // The normalization factors can be tested by evaluating the L2 norm
    // of the basis functions numerically using quadrature.
    // This is tested in the C++ implementation itself.

    // Here we just verify that evaluations and gradients produce
    // reasonable values at a few points
    std::vector<Vec2> test_points = {Vec2{-0.5, -0.5}, Vec2{0.0, 0.0}, Vec2{0.5, 0.0}};

    for (const auto& point : test_points) {
        DView1 values = basis.evaluate(point);
        DView2 gradient = basis.evaluate_gradient(point);

        // Values and gradients should be finite
        EXPECT_TRUE(all_finite(values));
        EXPECT_TRUE(all_finite(gradient));

        // Values should be of reasonable magnitude due to normalization
        for (int i = 0; i < values.size(); ++i) {
            EXPECT_LT(std::abs(values(i)), 10.0);
        }
    }
}

// CRITICAL TEST: Orthonormality verification - the defining property of Dubiner basis
TEST(DubinerBasisTest, OrthonormalityVerification) {
    DubinerBasis basis(2);  // Quadratic basis

    // Use Dunavant quadrature to integrate over reference triangle
    // Order 5 is sufficient to integrate quadratic basis functions squared (degree 4)
    auto quad = QuadratureFactory::dunavant_triangle(5);

    int n_basis = basis.get_n_basis();  // Should be 6 for order 2

    // Compute Gram matrix: G_ij = ∫∫ φᵢ(x,y) φⱼ(x,y) dA
    DView2 gram_matrix = DView2("tmp", n_basis, n_basis);

    for (int q = 0; q < quad->size(); ++q) {
        Vec2 qp = row2(quad->points, q);
        double w = quad->weights(q);

        DView1 phi = basis.evaluate(qp);

        // Add contribution: w * φᵢ * φⱼ
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                gram_matrix(i, j) += w * phi(i) * phi(j);
            }
        }
    }

    // Reference triangle area is 0.5
    for (int i = 0; i < n_basis; ++i) {
        for (int j = 0; j < n_basis; ++j) {
            gram_matrix(i, j) *= 0.5;
        }
    }

    // Verify orthonormality: G should be identity matrix
    // ∫∫ φᵢ φⱼ dA = δᵢⱼ (Kronecker delta)
    for (int i = 0; i < n_basis; ++i) {
        for (int j = 0; j < n_basis; ++j) {
            if (i == j) {
                // Diagonal: should be 1 (normalized)
                EXPECT_NEAR(gram_matrix(i, j), 1.0, 1e-10)
                    << "Basis function " << i << " is not normalized";
            } else {
                // Off-diagonal: should be 0 (orthogonal)
                EXPECT_NEAR(gram_matrix(i, j), 0.0, 1e-10)
                    << "Basis functions " << i << " and " << j << " are not orthogonal";
            }
        }
    }
}

// Test polynomial completeness - can Dubiner basis represent all polynomials up to order p?
TEST(DubinerBasisTest, PolynomialCompleteness) {
    DubinerBasis basis(2);  // Should be able to represent all polynomials up to degree 2

    // Test polynomial: p(x,y) = 1 + 2x + 3y + x² + xy + y²
    auto test_poly = [](const Vec2& p) {
        return 1.0 + 2.0 * p[0] + 3.0 * p[1] + p[0] * p[0] + p[0] * p[1] + p[1] * p[1];
    };

    // Use quadrature to find coefficients: cᵢ = ∫∫ p(x,y) φᵢ(x,y) dA
    auto quad = QuadratureFactory::dunavant_triangle(5);
    int n_basis = basis.get_n_basis();
    DView1 coeffs = DView1("tmp", n_basis);

    for (int q = 0; q < quad->size(); ++q) {
        Vec2 qp = row2(quad->points, q);
        double w = quad->weights(q);
        double poly_val = test_poly(qp);
        DView1 phi = basis.evaluate(qp);

        for (int i = 0; i < n_basis; ++i) {
            coeffs(i) += w * poly_val * phi(i);
        }
    }
    for (int i = 0; i < n_basis; ++i) {
        coeffs(i) *= 0.5;  // Triangle area
    }

    // Now verify: ∑ cᵢφᵢ(x,y) ≈ p(x,y) at test points
    std::vector<Vec2> test_points = {Vec2{0.2, 0.3}, Vec2{0.5, 0.2}, Vec2{0.1, 0.6}};

    for (const auto& pt : test_points) {
        double exact = test_poly(pt);
        DView1 phi = basis.evaluate(pt);
        double approx = dot(coeffs, phi);

        EXPECT_NEAR(approx, exact, 1e-10)
            << "Polynomial not accurately represented at (" << pt[0] << ", " << pt[1] << ")";
    }
}

// Test numerical gradient vs finite differences
TEST(DubinerBasisTest, NumericalGradientVerification) {
    DubinerBasis basis(2);

    Vec2 test_point{0.3, 0.4};
    double h = 1e-7;  // Finite difference step

    // Analytical gradient
    DView2 grad_analytical = basis.evaluate_gradient(test_point);

    // Numerical gradient via finite differences
    int n_basis = basis.get_n_basis();
    DView2 grad_numerical("grad_numerical", n_basis, 2);

    for (int i = 0; i < n_basis; ++i) {
        // x-derivative
        Vec2 pt_plus_x = test_point + Vec2{h, 0};
        Vec2 pt_minus_x = test_point - Vec2{h, 0};
        double val_plus_x = basis.evaluate(pt_plus_x)(i);
        double val_minus_x = basis.evaluate(pt_minus_x)(i);
        grad_numerical(i, 0) = (val_plus_x - val_minus_x) / (2 * h);

        // y-derivative
        Vec2 pt_plus_y = test_point + Vec2{0, h};
        Vec2 pt_minus_y = test_point - Vec2{0, h};
        double val_plus_y = basis.evaluate(pt_plus_y)(i);
        double val_minus_y = basis.evaluate(pt_minus_y)(i);
        grad_numerical(i, 1) = (val_plus_y - val_minus_y) / (2 * h);
    }

    // Compare
    for (int i = 0; i < n_basis; ++i) {
        EXPECT_NEAR(grad_analytical(i, 0), grad_numerical(i, 0), 1e-5)
            << "x-gradient mismatch for basis function " << i;
        EXPECT_NEAR(grad_analytical(i, 1), grad_numerical(i, 1), 1e-5)
            << "y-gradient mismatch for basis function " << i;
    }
}
