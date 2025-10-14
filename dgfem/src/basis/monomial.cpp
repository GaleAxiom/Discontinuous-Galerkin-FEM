/**
 * @file monomial.cpp
 * @brief Implementation of monomial basis for triangles
 */

#include "dgfem/basis/monomial.hpp"

#include "dgfem/reference/elements.hpp"

#include <cmath>

namespace dgfem {

MonomialBasisTriangle::MonomialBasisTriangle(int order)
    : OrthogonalBasis(std::make_shared<ReferenceTriangle>(), order) {
    // Set n_basis_ after construction
    n_basis_ = compute_n_basis();
}

int MonomialBasisTriangle::compute_n_basis() const {
    return (order_ + 1) * (order_ + 2) / 2;
}

double MonomialBasisTriangle::safe_power(double x, int n) const {
    if (n == 0)
        return 1.0;
    return std::pow(x, n);
}

Eigen::VectorXd MonomialBasisTriangle::evaluate(const Eigen::Vector2d& xi) const {
    Eigen::VectorXd phi(n_basis_);

    int idx = 0;
    for (int total_degree = 0; total_degree <= order_; ++total_degree) {
        for (int i = 0; i <= total_degree; ++i) {
            int j = total_degree - i;
            phi[idx] = safe_power(xi[0], i) * safe_power(xi[1], j);
            ++idx;
        }
    }

    return phi;
}

Eigen::MatrixXd MonomialBasisTriangle::evaluate_gradient(const Eigen::Vector2d& xi) const {
    Eigen::MatrixXd grad(n_basis_, 2);

    int idx = 0;
    for (int total_degree = 0; total_degree <= order_; ++total_degree) {
        for (int i = 0; i <= total_degree; ++i) {
            int j = total_degree - i;

            // d/dx: i * x^(i-1) * y^j
            if (i == 0) {
                grad(idx, 0) = 0.0;
            } else {
                grad(idx, 0) = i * safe_power(xi[0], i - 1) * safe_power(xi[1], j);
            }

            // d/dy: x^i * j * y^(j-1)
            if (j == 0) {
                grad(idx, 1) = 0.0;
            } else {
                grad(idx, 1) = safe_power(xi[0], i) * j * safe_power(xi[1], j - 1);
            }

            ++idx;
        }
    }

    return grad;
}

}  // namespace dgfem