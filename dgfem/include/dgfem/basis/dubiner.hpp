/**
 * @file dubiner.hpp
 * @brief Dubiner orthogonal basis functions for triangular elements
 */

#pragma once

#include "orthogonal.hpp"

namespace dgfem {

/**
 * @brief Dubiner orthogonal basis for triangular elements
 *
 * Implements the Dubiner basis which provides orthogonal polynomials
 * on the reference triangle using Jacobi polynomials.
 */
class DubinerBasis : public OrthogonalBasis {
public:
    explicit DubinerBasis(int order);

    Eigen::VectorXd evaluate(const Eigen::Vector2d& xi) const override;
    Eigen::MatrixXd evaluate_gradient(const Eigen::Vector2d& xi) const override;

protected:
    int compute_n_basis() const override;

    /**
     * @brief Evaluate Jacobi polynomial P_n^(alpha,beta)(x)
     */
    double jacobi_polynomial(int n, double alpha, double beta, double x) const;

    /**
     * @brief Evaluate derivative of Jacobi polynomial
     */
    double jacobi_derivative(int n, double alpha, double beta, double x) const;

    /**
     * @brief Transform from reference triangle to Dubiner coordinates
     */
    std::pair<double, double> transform_coordinates(const Eigen::Vector2d& xi) const;

    /**
     * @brief Compute the Dubiner basis function indices for given total degree
     */
    std::pair<int, int> get_dubiner_indices(int basis_idx) const;
};

}  // namespace dgfem