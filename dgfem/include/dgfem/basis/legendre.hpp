/**
 * @file legendre.hpp
 * @brief Legendre tensor product basis functions for quadrilateral elements with modern C++
 * optimizations
 */

#pragma once

#include <array>

#include "orthogonal.hpp"

namespace dgfem {

/**
 * @brief Tensor product Legendre basis for quadrilaterals with modern C++ features
 */
class LegendreBasis : public OrthogonalBasis {
public:
    explicit LegendreBasis(int order);

    Eigen::VectorXd evaluate(const Eigen::Vector2d& xi) const override;
    Eigen::MatrixXd evaluate_gradient(const Eigen::Vector2d& xi) const override;

    int compute_n_basis() const override;

    // Maximum supported order for compile-time optimizations
    static constexpr int MAX_ORDER = 10;

    /**
     * @brief Compute number of basis functions at compile time
     */
    [[nodiscard]] static constexpr int n_basis_for_order(int order) noexcept {
        return (order + 1) * (order + 1);
    }

private:
    /**
     * @brief Evaluate Legendre polynomial P_n(x) - constexpr for compile-time evaluation
     */
    [[nodiscard]] constexpr double legendre_polynomial(int n, double x) const noexcept;

    /**
     * @brief Evaluate Legendre polynomial derivative P'_n(x)
     */
    [[nodiscard]] double legendre_derivative(int n, double x) const noexcept;

    /**
     * @brief Evaluate multiple Legendre polynomials at once (vectorized)
     */
    void evaluate_legendre_sequence(int max_n, double x, double* results, int size) const noexcept;
};

// Constexpr implementation for compile-time optimization
constexpr double LegendreBasis::legendre_polynomial(int n, double x) const noexcept {
    if (n == 0)
        return 1.0;
    if (n == 1)
        return x;

    // Use recurrence relation: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
    double P_nm1 = 1.0;  // P_0
    double P_n = x;      // P_1

    for (int i = 2; i <= n; ++i) {
        double P_np1 = ((2 * i - 1) * x * P_n - (i - 1) * P_nm1) / i;
        P_nm1 = P_n;
        P_n = P_np1;
    }

    return P_n;
}

}  // namespace dgfem