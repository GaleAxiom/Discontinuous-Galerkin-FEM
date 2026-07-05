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
class DubinerBasis : public OrthogonalBasisCRTP<DubinerBasis> {
public:
    explicit DubinerBasis(int order);

    DView1 evaluate_impl(const Vec2& xi) const;
    DView2 evaluate_gradient_impl(const Vec2& xi) const;

    // Public (unlike the rest of this class's helpers below) since the CRTP base in
    // orthogonal.hpp dispatches to it via static_cast on a Derived*, which requires
    // public access -- matches LegendreBasis's already-public compute_n_basis.
    int compute_n_basis_impl() const;

protected:
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
    std::pair<double, double> transform_coordinates(const Vec2& xi) const;

    /**
     * @brief Compute the Dubiner basis function indices for given total degree
     */
    std::pair<int, int> get_dubiner_indices(int basis_idx) const;
};

}  // namespace dgfem