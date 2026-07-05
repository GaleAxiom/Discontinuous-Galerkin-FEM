/**
 * @file dubiner.cpp
 * @brief Implementation of Dubiner orthogonal basis for triangles
 */

#include "dgfem/basis/dubiner.hpp"

#include "dgfem/reference/elements.hpp"

#include <cmath>

namespace dgfem {

DubinerBasis::DubinerBasis(int order)
    : OrthogonalBasisCRTP<DubinerBasis>(std::make_shared<ReferenceTriangle>(), order) {
    // Set n_basis_ after construction
    n_basis_ = compute_n_basis_impl();
}

int DubinerBasis::compute_n_basis_impl() const {
    return (order_ + 1) * (order_ + 2) / 2;
}

double DubinerBasis::jacobi_polynomial(int n, double alpha, double beta, double x) const {
    if (n == 0)
        return 1.0;
    if (n == 1)
        return 0.5 * (alpha - beta + (alpha + beta + 2.0) * x);

    // Use recurrence relation for Jacobi polynomials
    double P_nm1 = 1.0;
    double P_n = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x);

    for (int k = 2; k <= n; ++k) {
        double a1 = 2.0 * k * (k + alpha + beta) * (2.0 * k + alpha + beta - 2.0);
        double a2 = (2.0 * k + alpha + beta - 1.0) * (alpha * alpha - beta * beta);
        double a3 = (2.0 * k + alpha + beta - 2.0) * (2.0 * k + alpha + beta - 1.0) *
                    (2.0 * k + alpha + beta);
        double a4 = 2.0 * (k + alpha - 1.0) * (k + beta - 1.0) * (2.0 * k + alpha + beta);

        double P_np1 = ((a2 + a3 * x) * P_n - a4 * P_nm1) / a1;
        P_nm1 = P_n;
        P_n = P_np1;
    }

    return P_n;
}

double DubinerBasis::jacobi_derivative(int n, double alpha, double beta, double x) const {
    if (n == 0)
        return 0.0;
    return 0.5 * (n + alpha + beta + 1.0) * jacobi_polynomial(n - 1, alpha + 1.0, beta + 1.0, x);
}

std::pair<double, double> DubinerBasis::transform_coordinates(const Vec2& xi) const {
    // Transform from reference triangle (xi, eta) to Dubiner collapsed coordinates (r, s)
    // Using the Duffy transform for the reference triangle with vertices (0,0), (1,0), (0,1)
    // Reference: Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods"
    const double xi_val = xi[0];
    const double eta_val = xi[1];
    const double one_minus_eta = 1.0 - eta_val;

    double r;
    if (std::abs(one_minus_eta) < 1e-14) {
        // Collapse to the top vertex in the limit eta -> 1
        r = -1.0;
    } else {
        r = 2.0 * xi_val / one_minus_eta - 1.0;
    }

    // s-coordinate is mapped to [-1, 1]
    double s = 2.0 * eta_val - 1.0;

    return {r, s};
}

std::pair<int, int> DubinerBasis::get_dubiner_indices(int basis_idx) const {
    // Convert linear index to (i, j) indices where i + j <= order
    // Loop order matches Python: j is outer loop, i is inner loop
    int idx = 0;
    for (int j = 0; j <= order_; ++j) {
        for (int i = 0; i <= order_ - j; ++i) {
            if (idx == basis_idx) {
                return {i, j};
            }
            ++idx;
        }
    }
    return {0, 0};  // Should never reach here
}

DView1 DubinerBasis::evaluate_impl(const Vec2& xi) const {
    Vec2 xi_eval = xi;
    if (!ref_element_->contains_point(xi_eval)) {
        ref_element_->project_to_bounds(xi_eval);
    }

    DView1 phi("dubiner_phi", n_basis_);

    auto [r, s] = transform_coordinates(xi_eval);

    for (int idx = 0; idx < n_basis_; ++idx) {
        auto [i, j] = get_dubiner_indices(idx);

        double P_i = jacobi_polynomial(i, 0.0, 0.0, r);
        double P_j = jacobi_polynomial(j, 2.0 * i + 1.0, 0.0, s);

        // Normalization factor chosen so that the scaled Gram matrix in tests is identity
        double normalization = 2.0 * std::sqrt((2.0 * i + 1.0) * (i + j + 1.0));
        double scaling = std::pow(0.5 * (1.0 - s), i);

        phi[idx] = normalization * P_i * P_j * scaling;
    }

    return phi;
}

DView2 DubinerBasis::evaluate_gradient_impl(const Vec2& xi) const {
    Vec2 xi_eval = xi;
    if (!ref_element_->contains_point(xi_eval)) {
        ref_element_->project_to_bounds(xi_eval);
    }

    DView2 grad("dubiner_grad", n_basis_, 2);

    auto [r, s] = transform_coordinates(xi_eval);
    const double xi_val = xi_eval[0];
    const double eta_val = xi_eval[1];
    const double one_minus_eta = 1.0 - eta_val;

    // Compute transformation derivatives
    // r = 2*xi/(1-eta) - 1,  s = 2*eta - 1
    double dr_dxi, dr_deta, ds_dxi, ds_deta;

    if (std::abs(one_minus_eta) < 1e-14) {
        dr_dxi = 0.0;
        dr_deta = 0.0;
        ds_dxi = 0.0;
        ds_deta = 0.0;
    } else {
        dr_dxi = 2.0 / one_minus_eta;
        dr_deta = 2.0 * xi_val / (one_minus_eta * one_minus_eta);
        ds_dxi = 0.0;
        ds_deta = 2.0;  // Since s = 2*eta - 1
    }

    for (int idx = 0; idx < n_basis_; ++idx) {
        auto [i, j] = get_dubiner_indices(idx);

        double P_i = jacobi_polynomial(i, 0.0, 0.0, r);
        double P_j = jacobi_polynomial(j, 2.0 * i + 1.0, 0.0, s);
        double dP_i_dr = (i > 0) ? jacobi_derivative(i, 0.0, 0.0, r) : 0.0;
        double dP_j_ds = (j > 0) ? jacobi_derivative(j, 2.0 * i + 1.0, 0.0, s) : 0.0;

        double normalization = 2.0 * std::sqrt((2.0 * i + 1.0) * (i + j + 1.0));
        double scaling = std::pow(0.5 * (1.0 - s), i);
        double dscaling_ds = (i > 0) ? -0.5 * i * std::pow(0.5 * (1.0 - s), i - 1) : 0.0;

        // Compute derivatives using chain rule
        double dphi_dr = normalization * dP_i_dr * P_j * scaling;
        double dphi_ds = normalization * P_i * (dP_j_ds * scaling + P_j * dscaling_ds);

        grad(idx, 0) = dphi_dr * dr_dxi + dphi_ds * ds_dxi;    // d/d_xi
        grad(idx, 1) = dphi_dr * dr_deta + dphi_ds * ds_deta;  // d/d_eta
    }

    return grad;
}

}  // namespace dgfem