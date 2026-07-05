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
class MonomialBasisTriangle : public OrthogonalBasisCRTP<MonomialBasisTriangle> {
public:
    explicit MonomialBasisTriangle(int order);

    DView1 evaluate_impl(const Vec2& xi) const;
    DView2 evaluate_gradient_impl(const Vec2& xi) const;

    // Public (see DubinerBasis for the same note) since the CRTP base dispatches to
    // this via static_cast on a Derived*, which requires public access.
    int compute_n_basis_impl() const;

private:
    /**
     * @brief Power function that handles x^0 = 1 even when x = 0
     */
    double safe_power(double x, int n) const;
};

}  // namespace dgfem