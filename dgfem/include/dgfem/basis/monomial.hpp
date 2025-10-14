/**
 * @file monomial.hpp
 * @brief Monomial basis functions for triangular elements
 */

#pragma once

#include "orthogonal.hpp"

namespace dgfem {

/**
 * @brief Monomial basis for triangular elements
 *
 * Implements basis functions of the form x^i * y^j where i+j <= order
 */
class MonomialBasisTriangle : public OrthogonalBasis {
public:
    explicit MonomialBasisTriangle(int order);

    Eigen::VectorXd evaluate(const Eigen::Vector2d& xi) const override;
    Eigen::MatrixXd evaluate_gradient(const Eigen::Vector2d& xi) const override;

protected:
    int compute_n_basis() const override;

private:
    /**
     * @brief Power function that handles x^0 = 1 even when x = 0
     */
    double safe_power(double x, int n) const;
};

}  // namespace dgfem