/**
 * @file legendre.cpp
 * @brief Implementation of Legendre tensor product basis with modern C++ optimizations
 */

#include "dgfem/basis/legendre.hpp"
#include "dgfem/reference/elements.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <numbers>

namespace dgfem {

LegendreBasis::LegendreBasis(int order) 
    : OrthogonalBasis(std::make_shared<ReferenceQuad>(), order) {
    if (order > MAX_ORDER) {
        std::ostringstream oss;
        oss << "Legendre basis for order " << order << " not supported (max: " << MAX_ORDER << ")";
        throw std::invalid_argument(oss.str());
    }
    n_basis_ = compute_n_basis();
}

int LegendreBasis::compute_n_basis() const {
    return n_basis_for_order(order_);
}

double LegendreBasis::legendre_derivative(int n, double x) const noexcept {
    if (n == 0) return 0.0;
    
    // Handle endpoints where x^2 - 1 = 0
    constexpr double eps = 1e-12;
    if (std::abs(std::abs(x) - 1.0) < eps) {
        double sign = (x > 0) ? 1.0 : -1.0;
        double parity = (n % 2 == 0) ? 1.0 : -1.0;
        return 0.5 * n * (n + 1) * parity * sign;
    }
    
    // Use formula: P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    double P_n = legendre_polynomial(n, x);
    double P_nm1 = legendre_polynomial(n - 1, x);
    return n * (x * P_n - P_nm1) / (x * x - 1.0);
}

void LegendreBasis::evaluate_legendre_sequence(int max_n, double x, double* results, int size) const noexcept {
    if (!results || size == 0 || max_n < 0) return;
    
    results[0] = 1.0;
    if (max_n == 0 || size == 1) return;
    
    results[1] = x;
    if (max_n == 1 || size == 2) return;
    
    // Compute all polynomials up to max_n using recurrence
    for (int i = 2; i <= max_n && i < size; ++i) {
        results[i] = ((2*i - 1) * x * results[i-1] - (i - 1) * results[i-2]) / i;
    }
}

Eigen::VectorXd LegendreBasis::evaluate(const Eigen::Vector2d& xi) const {
    Eigen::VectorXd phi(n_basis_);
    
    // Preallocate for vectorized computation
    std::array<double, MAX_ORDER + 1> P_x, P_y;
    evaluate_legendre_sequence(order_, xi[0], P_x.data(), order_ + 1);
    evaluate_legendre_sequence(order_, xi[1], P_y.data(), order_ + 1);
    
    // Tensor product: phi_{i,j} = P_i(x) * P_j(y)
    int idx = 0;
    for (int j = 0; j <= order_; ++j) {
        for (int i = 0; i <= order_; ++i) {
            phi[idx++] = P_x[i] * P_y[j];
        }
    }
    
    return phi;
}

Eigen::MatrixXd LegendreBasis::evaluate_gradient(const Eigen::Vector2d& xi) const {
    Eigen::MatrixXd grad(n_basis_, 2);
    
    // Precompute polynomial values and derivatives
    std::array<double, MAX_ORDER + 1> P_x, P_y, dP_x, dP_y;
    evaluate_legendre_sequence(order_, xi[0], P_x.data(), order_ + 1);
    evaluate_legendre_sequence(order_, xi[1], P_y.data(), order_ + 1);
    
    for (int i = 0; i <= order_; ++i) {
        dP_x[i] = legendre_derivative(i, xi[0]);
        dP_y[i] = legendre_derivative(i, xi[1]);
    }
    
    // Tensor product gradients
    int idx = 0;
    for (int j = 0; j <= order_; ++j) {
        for (int i = 0; i <= order_; ++i) {
            grad(idx, 0) = dP_x[i] * P_y[j];   // d/d_xi
            grad(idx, 1) = P_x[i] * dP_y[j];   // d/d_eta
            ++idx;
        }
    }
    
    return grad;
}

} // namespace dgfem