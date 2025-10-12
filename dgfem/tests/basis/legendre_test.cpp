#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <dgfem/basis/legendre.hpp>
#include <dgfem/reference/elements.hpp>
#include <dgfem/quadrature/factory.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace dgfem;
using namespace testing;

TEST(LegendreBasisTest, ThrowsErrorForHighOrder) {
    // MAX_ORDER is now 10, so 11 should throw
    EXPECT_THROW(LegendreBasis(11), std::invalid_argument);
}

TEST(LegendreBasisTest, CorrectBasisCount) {
    // Test for various orders up to MAX_ORDER
    for (int order = 1; order <= 10; ++order) {
        LegendreBasis basis(order);
        auto n_basis = basis.compute_n_basis();
        EXPECT_EQ(n_basis, std::pow(order + 1, 2));
    }
}

TEST(LegendreBasisTest, EvaluateAtOrigin) {
    LegendreBasis basis(2); // Test with order 2
    Eigen::Vector2d origin(0.0, 0.0);
    
    Eigen::VectorXd values = basis.evaluate(origin);
    EXPECT_EQ(values.size(), 9); // (2+1)^2 = 9 basis functions
    
    // Expected values at origin for order 2
    // phi_ij(x,y) = P_i(x) * P_j(y)
    // P_0(0) = 1, P_1(0) = 0, P_2(0) = -0.5
    // The basis functions are ordered by j, then i.
    // i=0, j=0: P_0(0)P_0(0) = 1
    // i=1, j=0: P_1(0)P_0(0) = 0
    // i=2, j=0: P_2(0)P_0(0) = -0.5
    // i=0, j=1: P_0(0)P_1(0) = 0
    // i=1, j=1: P_1(0)P_1(0) = 0
    // i=2, j=1: P_2(0)P_1(0) = 0
    // i=0, j=2: P_0(0)P_2(0) = -0.5
    // i=1, j=2: P_1(0)P_2(0) = 0
    // i=2, j=2: P_2(0)P_2(0) = 0.25
    Eigen::VectorXd expected_values(9);
    expected_values << 1.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.25;

    for (int i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(values(i), expected_values(i), 1e-12);
    }
}

TEST(LegendreBasisTest, EvaluateGradientAtOrigin) {
    LegendreBasis basis(2);
    Eigen::Vector2d origin(0.0, 0.0);
    
    Eigen::MatrixXd gradient = basis.evaluate_gradient(origin);
    EXPECT_EQ(gradient.rows(), 9); // (2+1)^2 = 9 basis functions
    EXPECT_EQ(gradient.cols(), 2); // 2D gradient
    
    // At origin, grad(phi_ij) = (P'_i(0)P_j(0), P_i(0)P'_j(0))
    // P_0(0)=1, P_1(0)=0, P_2(0)=-0.5
    // P'_0(0)=0, P'_1(0)=1, P'_2(0)=0
    // The basis functions are ordered by j, then i.
    Eigen::MatrixXd expected_gradient = Eigen::MatrixXd::Zero(9, 2);
    // k=1: phi_10 = P_1(x)P_0(y) -> grad(P'_1(0)P_0(0), P_1(0)P'_0(0)) = (1*1, 0*0) = (1,0)
    expected_gradient.row(1) << 1.0, 0.0;
    // k=3: phi_01 = P_0(x)P_1(y) -> grad(P'_0(0)P_1(0), P_0(0)P'_1(0)) = (0*0, 1*1) = (0,1)
    expected_gradient.row(3) << 0.0, 1.0;
    // k=7: phi_12 = P_1(x)P_2(y) -> grad(P'_1(0)P_2(0), P_1(0)P'_2(0)) = (1*(-0.5), 0*0) = (-0.5,0)
    expected_gradient.row(7) << -0.5, 0.0;
    // k=5: phi_21 = P_2(x)P_1(y) -> grad(P'_2(0)P_1(0), P_2(0)P'_1(0)) = (0*0, -0.5*1) = (0,-0.5)
    expected_gradient.row(5) << 0.0, -0.5;
    
    for (int i = 0; i < gradient.rows(); ++i) {
        EXPECT_NEAR(gradient(i, 0), expected_gradient(i, 0), 1e-12);
        EXPECT_NEAR(gradient(i, 1), expected_gradient(i, 1), 1e-12);
    }
}

TEST(LegendreBasisTest, EvaluateAtCorners) {
    LegendreBasis basis(1); // Test with linear basis
    
    // Test at all four corners of reference quad [-1,1]²
    std::vector<std::pair<Eigen::Vector2d, std::string>> corners = {
        {Eigen::Vector2d(-1.0, -1.0), "bottom-left"},
        {Eigen::Vector2d(1.0, -1.0), "bottom-right"},
        {Eigen::Vector2d(1.0, 1.0), "top-right"},
        {Eigen::Vector2d(-1.0, 1.0), "top-left"}
    };
    
    for (const auto& [corner, name] : corners) {
        Eigen::VectorXd values = basis.evaluate(corner);
        EXPECT_EQ(values.size(), 4) << "At corner: " << name;
        
        // For tensor product: φᵢⱼ(x,y) = Pᵢ(x)Pⱼ(y)
        // Ordering: (i,j) = (0,0), (1,0), (0,1), (1,1)
        double x = corner(0), y = corner(1);
        EXPECT_NEAR(values(0), 1.0, 1e-12) << "P₀(x)P₀(y) at " << name;  // 1*1 = 1
        EXPECT_NEAR(values(1), x, 1e-12) << "P₁(x)P₀(y) at " << name;    // x*1 = x
        EXPECT_NEAR(values(2), y, 1e-12) << "P₀(x)P₁(y) at " << name;    // 1*y = y
        EXPECT_NEAR(values(3), x*y, 1e-12) << "P₁(x)P₁(y) at " << name;  // x*y
    }
}

// Test orthogonality of Legendre basis over reference quad
TEST(LegendreBasisTest, OrthogonalityVerification) {
    LegendreBasis basis(2);  // Quadratic basis
    
    // Use Gauss-Legendre quadrature - order 3 is sufficient for degree 4 polynomials
    auto quad = QuadratureFactory::gauss_legendre_quad(3);
    
    int n_basis = basis.compute_n_basis();  // 9 for order 2
    Eigen::MatrixXd gram_matrix = Eigen::MatrixXd::Zero(n_basis, n_basis);
    
    for (int q = 0; q < quad->size(); ++q) {
        Eigen::Vector2d qp = quad->points.row(q);
        double w = quad->weights(q);
        Eigen::VectorXd phi = basis.evaluate(qp);
        
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                gram_matrix(i, j) += w * phi(i) * phi(j);
            }
        }
    }
    
    // Reference quad [-1,1]² has area 4, but Legendre polynomials 
    // are orthogonal with ∫₋₁¹ Pᵢ Pⱼ dx = 2/(2i+1) δᵢⱼ
    // For tensor product: ∫∫ Pᵢ(x)Pⱼ(y)Pₖ(x)Pₗ(y) = [2/(2i+1)δᵢₖ][2/(2j+1)δⱼₗ]
    
    // Expected normalization for Legendre polynomials
    auto norm_factor = [](int n) { return std::sqrt(2.0 / (2.0 * n + 1.0)); };
    
    for (int i = 0; i < n_basis; ++i) {
        for (int j = 0; j < n_basis; ++j) {
            // Basis is ordered: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
            int i_x = i % 3;  // x polynomial degree
            int i_y = i / 3;  // y polynomial degree
            int j_x = j % 3;
            int j_y = j / 3;
            
            double expected;
            if (i_x == j_x && i_y == j_y) {
                // Diagonal: ||Pᵢ(x)||² ||Pⱼ(y)||² = [2/(2i+1)][2/(2j+1)]
                expected = (2.0 / (2.0*i_x + 1.0)) * (2.0 / (2.0*i_y + 1.0));
            } else {
                // Off-diagonal: should be zero (orthogonal)
                expected = 0.0;
            }
            
            EXPECT_NEAR(gram_matrix(i, j), expected, 1e-10)
                << "Orthogonality failed for basis (" << i_x << "," << i_y 
                << ") and (" << j_x << "," << j_y << ")";
        }
    }
}

// Test numerical gradient verification
TEST(LegendreBasisTest, NumericalGradientVerification) {
    LegendreBasis basis(2);
    Eigen::Vector2d test_point(0.3, -0.5);
    double h = 1e-7;
    
    Eigen::MatrixXd grad_analytical = basis.evaluate_gradient(test_point);
    
    int n_basis = basis.compute_n_basis();
    Eigen::MatrixXd grad_numerical(n_basis, 2);
    
    for (int i = 0; i < n_basis; ++i) {
        // x-derivative
        double val_plus_x = basis.evaluate(test_point + Eigen::Vector2d(h, 0))(i);
        double val_minus_x = basis.evaluate(test_point - Eigen::Vector2d(h, 0))(i);
        grad_numerical(i, 0) = (val_plus_x - val_minus_x) / (2*h);
        
        // y-derivative
        double val_plus_y = basis.evaluate(test_point + Eigen::Vector2d(0, h))(i);
        double val_minus_y = basis.evaluate(test_point - Eigen::Vector2d(0, h))(i);
        grad_numerical(i, 1) = (val_plus_y - val_minus_y) / (2*h);
    }
    
    for (int i = 0; i < n_basis; ++i) {
        EXPECT_NEAR(grad_analytical(i, 0), grad_numerical(i, 0), 1e-5)
            << "x-gradient mismatch for basis " << i;
        EXPECT_NEAR(grad_analytical(i, 1), grad_numerical(i, 1), 1e-5)
            << "y-gradient mismatch for basis " << i;
    }
}
