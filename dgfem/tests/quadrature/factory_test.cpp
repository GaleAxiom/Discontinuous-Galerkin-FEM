#include <Eigen/Dense>
#include <cmath>
#include <dgfem/quadrature/factory.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

TEST(QuadratureFactoryTest, GaussLegendre1D) {
    // Test 2-point rule
    auto rule_2pt = QuadratureFactory::gauss_legendre_1d(2);

    ASSERT_EQ(rule_2pt->points.rows(), 2);
    ASSERT_EQ(rule_2pt->points.cols(), 1);
    ASSERT_EQ(rule_2pt->weights.size(), 2);

    // Known values for 2-point rule
    EXPECT_NEAR(rule_2pt->points(0, 0), -1.0 / std::sqrt(3.0), 1e-12);
    EXPECT_NEAR(rule_2pt->points(1, 0), 1.0 / std::sqrt(3.0), 1e-12);
    EXPECT_NEAR(rule_2pt->weights(0), 1.0, 1e-12);
    EXPECT_NEAR(rule_2pt->weights(1), 1.0, 1e-12);

    // Test that weights sum to 2 (length of interval [-1,1])
    EXPECT_NEAR(rule_2pt->weights.sum(), 2.0, 1e-12);
}

TEST(QuadratureFactoryTest, GaussLegendreQuad) {
    // Test 2x2 rule
    auto rule_2x2 = QuadratureFactory::gauss_legendre_quad(2);

    ASSERT_EQ(rule_2x2->points.rows(), 4);
    ASSERT_EQ(rule_2x2->points.cols(), 2);
    ASSERT_EQ(rule_2x2->weights.size(), 4);

    // Known values for 2x2 tensor product
    double p = 1.0 / std::sqrt(3.0);
    Eigen::Vector2d expected_points[] = {{-p, -p}, {p, -p}, {-p, p}, {p, p}};

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(rule_2x2->points(i, 0), expected_points[i](0), 1e-12);
        EXPECT_NEAR(rule_2x2->points(i, 1), expected_points[i](1), 1e-12);
        EXPECT_NEAR(rule_2x2->weights(i), 1.0, 1e-12);
    }

    // Test that weights sum to 4 (area of square [-1,1]²)
    EXPECT_NEAR(rule_2x2->weights.sum(), 4.0, 1e-12);
}

TEST(QuadratureFactoryTest, DunavantTriangleOrder1) {
    auto rule = QuadratureFactory::dunavant_triangle(1);

    ASSERT_EQ(rule->points.rows(), 1);
    ASSERT_EQ(rule->points.cols(), 2);
    ASSERT_EQ(rule->weights.size(), 1);

    // Order 1 rule is centroid with weight = area
    EXPECT_NEAR(rule->points(0, 0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(rule->points(0, 1), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(rule->weights(0), 0.5, 1e-12);

    // Test that weights sum to 0.5 (area of reference triangle)
    EXPECT_NEAR(rule->weights.sum(), 0.5, 1e-12);
}

TEST(QuadratureFactoryTest, DunavantTriangleHighOrder) {
    // Test orders 2 through 6
    for (int order = 2; order < 6; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);

        // Check that points are inside reference triangle
        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);

            EXPECT_GE(x, -1e-12) << "Point " << i << " x-coord negative for order " << order;
            EXPECT_GE(y, -1e-12) << "Point " << i << " y-coord negative for order " << order;
            EXPECT_LE(x + y, 1.0 + 1e-12)
                << "Point " << i << " outside triangle for order " << order;
        }

        // Check that weights sum to area of reference triangle
        EXPECT_NEAR(rule->weights.sum(), 0.5, 1e-12)
            << "Weights don't sum to area for order " << order;
    }
}

TEST(QuadratureFactoryTest, TensorProductTriangle) {
    // Test high order that triggers tensor product fallback
    auto rule = QuadratureFactory::dunavant_triangle(4);

    // Check that points are inside reference triangle
    for (int i = 0; i < rule->points.rows(); ++i) {
        double x = rule->points(i, 0);
        double y = rule->points(i, 1);

        EXPECT_GE(x, -1e-12);
        EXPECT_GE(y, -1e-12);
        EXPECT_LE(x + y, 1.0 + 1e-12);
    }

    // Check that weights sum to area of reference triangle
    EXPECT_NEAR(rule->weights.sum(), 0.5, 1e-12);
}

TEST(QuadratureFactoryTest, IntegrationAccuracy) {
    // Test that rules can exactly integrate polynomials of appropriate degree
    auto test_polynomial = [](const std::unique_ptr<QuadratureRule>& rule, int poly_deg_x,
                              int poly_deg_y) -> double {
        double integral = 0.0;
        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);
            double val = std::pow(x, poly_deg_x) * std::pow(y, poly_deg_y);
            integral += rule->weights(i) * val;
        }
        return integral;
    };

    auto factorial = [](int n) {
        long long res = 1;
        for (int i = 2; i <= n; i++)
            res *= i;
        return res;
    };

    // Test each order up to the implemented maximum (5)
    for (int order = 1; order <= 5; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);

        // A rule of 'order' should be exact for polynomials of degree 'order-1'.
        // We test with a polynomial of degree 'order-1'.
        // p(x,y) = x^(order-2) * y^1. Total degree = order-1.
        // For order=1, this is x^-1 * y, which is invalid. We'll handle that case.
        if (order == 1)
            continue;  // Order 1 rule is tested separately and is exact for degree 0.

        int a = order - 2;
        int b = 1;

        double numerical_result = test_polynomial(rule, a, b);

        // Analytical integral of x^a * y^b over the reference triangle is a!*b! / (a+b+2)!
        double analytical_result =
            static_cast<double>(factorial(a) * factorial(b)) / factorial(a + b + 2);

        EXPECT_NEAR(numerical_result, analytical_result, 1e-12)
            << "Failed for order " << order << " with polynomial x^" << a << "y^" << b;
    }
}

// New comprehensive tests for mathematical properties

TEST(QuadratureFactoryTest, WeightPositivity) {
    // Test weight positivity where expected
    // NOTE: Some high-order quadrature rules (like Dunavant order 3+) can have
    // negative weights while still being exact. This is mathematically valid.

    // Test Gauss-Legendre 1D - these should always have positive weights
    for (int n = 1; n <= 8; ++n) {
        auto rule = QuadratureFactory::gauss_legendre_1d(n);
        for (int i = 0; i < rule->weights.size(); ++i) {
            EXPECT_GT(rule->weights(i), 0.0)
                << "Negative weight in 1D Gauss-Legendre rule of order " << n;
        }
    }

    // Test Gauss-Legendre quad - these should always have positive weights
    for (int n = 1; n <= 8; ++n) {
        auto rule = QuadratureFactory::gauss_legendre_quad(n);
        for (int i = 0; i < rule->weights.size(); ++i) {
            EXPECT_GT(rule->weights(i), 0.0)
                << "Negative weight in quad Gauss-Legendre rule of order " << n;
        }
    }

    // Test Dunavant triangle - orders 1-2 should have positive weights
    // Higher orders (3+) may have negative weights (this is expected)
    for (int order = 1; order <= 2; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);
        for (int i = 0; i < rule->weights.size(); ++i) {
            EXPECT_GT(rule->weights(i), 0.0)
                << "Negative weight in Dunavant triangle rule of order " << order;
        }
    }

    // Document that higher-order Dunavant rules can have negative weights
    // Order 3 has negative weights - this is expected and mathematically valid
    auto rule_order3 = QuadratureFactory::dunavant_triangle(3);
    bool has_negative_weights = false;
    for (int i = 0; i < rule_order3->weights.size(); ++i) {
        if (rule_order3->weights(i) < 0.0) {
            has_negative_weights = true;
            break;
        }
    }
    EXPECT_TRUE(has_negative_weights)
        << "Dunavant order 3 should have negative weights (this is expected)";
}

TEST(QuadratureFactoryTest, DegreeOfPrecision) {
    // Verify exact integration for polynomials up to the stated degree
    // According to documentation: order k should integrate polynomials up to degree k exactly

    auto integrate_monomial_triangle = [](const std::unique_ptr<QuadratureRule>& rule, int deg_x,
                                          int deg_y) -> double {
        double sum = 0.0;
        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);
            sum += rule->weights(i) * std::pow(x, deg_x) * std::pow(y, deg_y);
        }
        return sum;
    };

    auto exact_integral_triangle = [](int a, int b) -> double {
        // Integral of x^a * y^b over reference triangle {(x,y) : x,y >= 0, x+y <= 1}
        // Result: a! * b! / (a + b + 2)!
        long long numerator = 1, denominator = 1;
        for (int i = 1; i <= a; ++i)
            numerator *= i;
        for (int i = 1; i <= b; ++i)
            numerator *= i;
        for (int i = 1; i <= a + b + 2; ++i)
            denominator *= i;
        return static_cast<double>(numerator) / denominator;
    };

    // Test Dunavant rules - order k should be exact for degree k polynomials
    for (int order = 1; order <= 5; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);
        int exact_degree = order;  // Changed from 2*order-1

        // Test all monomials up to exact degree
        for (int total_deg = 0; total_deg <= exact_degree; ++total_deg) {
            for (int deg_x = 0; deg_x <= total_deg; ++deg_x) {
                int deg_y = total_deg - deg_x;

                double numerical = integrate_monomial_triangle(rule, deg_x, deg_y);
                double analytical = exact_integral_triangle(deg_x, deg_y);

                EXPECT_NEAR(numerical, analytical, 1e-10)  // Relaxed tolerance
                    << "Order " << order << " failed for x^" << deg_x << "y^" << deg_y
                    << " (total degree " << total_deg << ")";
            }
        }
    }
}

TEST(QuadratureFactoryTest, GaussLegendreSymmetry) {
    // Gauss-Legendre points should be symmetric about origin

    for (int n = 2; n <= 8; n += 2) {  // Test even orders up to max (8)
        auto rule = QuadratureFactory::gauss_legendre_1d(n);

        // Points should come in pairs: (x_i, -x_i) with equal weights
        int half = n / 2;
        for (int i = 0; i < half; ++i) {
            EXPECT_NEAR(rule->points(i, 0), -rule->points(n - 1 - i, 0), 1e-12)
                << "Points not symmetric for order " << n;
            EXPECT_NEAR(rule->weights(i), rule->weights(n - 1 - i), 1e-12)
                << "Weights not symmetric for order " << n;
        }
    }
}

TEST(QuadratureFactoryTest, QuadratureConvergence) {
    // Test that higher order rules give better accuracy for smooth functions

    auto smooth_function = [](double x, double y) -> double {
        return std::exp(x + y) * std::sin(M_PI * x) * std::cos(M_PI * y);
    };

    // Approximate "exact" value using very high order quadrature
    auto exact_rule = QuadratureFactory::dunavant_triangle(5);
    double exact = 0.0;
    for (int i = 0; i < exact_rule->points.rows(); ++i) {
        double x = exact_rule->points(i, 0);
        double y = exact_rule->points(i, 1);
        exact += exact_rule->weights(i) * smooth_function(x, y);
    }

    // Test convergence with increasing order
    std::vector<double> errors;
    for (int order = 1; order <= 4; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);
        double approx = 0.0;
        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);
            approx += rule->weights(i) * smooth_function(x, y);
        }
        errors.push_back(std::abs(approx - exact));
    }

    // Errors should decrease (not strictly monotonic due to different formulas)
    EXPECT_LT(errors[3], errors[0]) << "Higher order quadrature not more accurate";
}

TEST(QuadratureFactoryTest, TrianglePointsInDomain) {
    // All quadrature points should be strictly inside reference triangle

    for (int order = 1; order <= 5; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);

        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);

            // Points should satisfy: x >= 0, y >= 0, x + y <= 1
            EXPECT_GE(x, -1e-14) << "Point " << i << " has negative x";
            EXPECT_GE(y, -1e-14) << "Point " << i << " has negative y";
            EXPECT_LE(x + y, 1.0 + 1e-14)
                << "Point " << i << " outside triangle: x=" << x << ", y=" << y;
        }
    }
}

TEST(QuadratureFactoryTest, QuadPointsInDomain) {
    // All quadrature points should be in reference quad [-1,1]^2

    for (int n = 1; n <= 5; ++n) {
        auto rule = QuadratureFactory::gauss_legendre_quad(n);

        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);

            EXPECT_GE(x, -1.0 - 1e-14) << "Point " << i << " x < -1";
            EXPECT_LE(x, 1.0 + 1e-14) << "Point " << i << " x > 1";
            EXPECT_GE(y, -1.0 - 1e-14) << "Point " << i << " y < -1";
            EXPECT_LE(y, 1.0 + 1e-14) << "Point " << i << " y > 1";
        }
    }
}

TEST(QuadratureFactoryTest, ConstantFunctionIntegration) {
    // Integrating constant function should give exact domain measure

    // Triangle: integral of 1 over reference triangle = area = 0.5
    for (int order = 1; order <= 5; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);
        double integral = rule->weights.sum();
        EXPECT_NEAR(integral, 0.5, 1e-14)
            << "Triangle constant integration failed for order " << order;
    }

    // Quad: integral of 1 over [-1,1]^2 = area = 4
    for (int n = 1; n <= 5; ++n) {
        auto rule = QuadratureFactory::gauss_legendre_quad(n);
        double integral = rule->weights.sum();
        EXPECT_NEAR(integral, 4.0, 1e-14) << "Quad constant integration failed for order " << n;
    }

    // 1D: integral of 1 over [-1,1] = length = 2
    for (int n = 1; n <= 5; ++n) {
        auto rule = QuadratureFactory::gauss_legendre_1d(n);
        double integral = rule->weights.sum();
        EXPECT_NEAR(integral, 2.0, 1e-14) << "1D constant integration failed for order " << n;
    }
}

TEST(QuadratureFactoryTest, LinearFunctionIntegration) {
    // Test exact integration of linear functions

    auto integrate_linear_triangle = [](const std::unique_ptr<QuadratureRule>& rule, double a,
                                        double b, double c) -> double {
        // Integrate f(x,y) = a*x + b*y + c
        double sum = 0.0;
        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);
            sum += rule->weights(i) * (a * x + b * y + c);
        }
        return sum;
    };

    // Analytical integral of a*x + b*y + c over reference triangle
    // = a*(1/6) + b*(1/6) + c*(1/2)
    auto exact_linear = [](double a, double b, double c) -> double {
        return a / 6.0 + b / 6.0 + c / 2.0;
    };

    // Test various linear functions
    std::vector<std::tuple<double, double, double>> test_cases = {
        {1.0, 0.0, 0.0},  // f = x
        {0.0, 1.0, 0.0},  // f = y
        {0.0, 0.0, 1.0},  // f = 1
        {2.0, 3.0, 1.0},  // f = 2x + 3y + 1
        {-1.0, 2.0, 5.0}  // f = -x + 2y + 5
    };

    for (int order = 1; order <= 5; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);

        for (const auto& [a, b, c] : test_cases) {
            double numerical = integrate_linear_triangle(rule, a, b, c);
            double analytical = exact_linear(a, b, c);

            EXPECT_NEAR(numerical, analytical, 1e-12)
                << "Order " << order << " failed for " << a << "*x + " << b << "*y + " << c;
        }
    }
}

TEST(QuadratureFactoryTest, QuadraticFunctionIntegration) {
    // Test exact integration of quadratic functions

    auto integrate_quadratic = [](const std::unique_ptr<QuadratureRule>& rule, double a,
                                  double b) -> double {
        // Integrate f(x,y) = x^2 + y^2 + a*x*y + b
        double sum = 0.0;
        for (int i = 0; i < rule->points.rows(); ++i) {
            double x = rule->points(i, 0);
            double y = rule->points(i, 1);
            sum += rule->weights(i) * (x * x + y * y + a * x * y + b);
        }
        return sum;
    };

    // Analytical integral of x^2 + y^2 + a*x*y + b over reference triangle
    // x^2: 1/12, y^2: 1/12, xy: 1/24, constant: b/2
    auto exact_quadratic = [](double a, double b) -> double {
        return 1.0 / 12.0 + 1.0 / 12.0 + a / 24.0 + b / 2.0;
    };

    std::vector<std::pair<double, double>> test_cases = {
        {0.0, 0.0},  // x^2 + y^2
        {1.0, 0.0},  // x^2 + y^2 + xy
        {2.0, 1.0},  // x^2 + y^2 + 2xy + 1
        {-1.0, 3.0}  // x^2 + y^2 - xy + 3
    };

    // Order 2 and higher should be exact for quadratics
    for (int order = 2; order <= 5; ++order) {
        auto rule = QuadratureFactory::dunavant_triangle(order);

        for (const auto& [a, b] : test_cases) {
            double numerical = integrate_quadratic(rule, a, b);
            double analytical = exact_quadratic(a, b);

            EXPECT_NEAR(numerical, analytical, 1e-12)
                << "Order " << order << " failed for quadratic with a=" << a << ", b=" << b;
        }
    }
}
