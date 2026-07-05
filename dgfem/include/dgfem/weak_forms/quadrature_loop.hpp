/**
 * @file quadrature_loop.hpp
 * @brief Shared volume-residual quadrature-loop skeleton for nonlinear (Euler/
 * Navier-Stokes-style) weak formulations.
 *
 * Laplace/Advection's bilinear-form matrix assembly follows a genuinely different
 * shape (no solution interpolation, direct basis-pair accumulation) and is not
 * templated here -- only the Euler/NavierStokes volume-residual pattern (interpolate
 * U at each quad point, evaluate a flux, accumulate w*(dphidx*F + dphidy*G)) is
 * shared, since that is the one skeleton that is genuinely identical across formulations.
 */
#pragma once

#include "dgfem/kokkos_math.hpp"

namespace dgfem {

/// Interpolates the 4-variable solution `U_q` at each volume quadrature point from
/// `u_coeffs_elem` (via the existing `gemm` helper), calls `flux_fn(q, U_q) ->
/// std::pair<Vec4, Vec4>` to get the physical-space flux pair `(F_q, G_q)`, and
/// accumulates `sign * w * (dphidx*F_q[v] + dphidy*G_q[v])` into the returned
/// `(n_basis x n_vars)` residual. `sign` is +1 for inviscid (Euler) terms and -1 for
/// viscous (Navier-Stokes) terms, matching each formulation's existing convention.
/// `flux_fn` may close over `dphi_dx`/`u_coeffs_elem` itself if it additionally needs
/// the solution gradient at quad point `q` (as Navier-Stokes's viscous flux does).
template <typename FluxFn>
DView2 accumulate_volume_residual(const DView2& u_coeffs_elem, const DView2& phi,
                                  const DView2& dphi_dx, const DView1& weights, const DView2& J_det,
                                  double sign, FluxFn&& flux_fn) {
    const int n_basis = static_cast<int>(phi.extent(1));
    const int n_vars = static_cast<int>(u_coeffs_elem.extent(1));
    const int n_quad = static_cast<int>(weights.extent(0));
    DView2 R_vol("R_vol", n_basis, n_vars);

    DView2 U_quad("U_quad", n_quad, n_vars);
    gemm('N', 'N', 1.0, phi, u_coeffs_elem, 0.0, U_quad);

    for (int q = 0; q < n_quad; ++q) {
        const Vec4 U_q{U_quad(q, 0), U_quad(q, 1), U_quad(q, 2), U_quad(q, 3)};
        auto [F_q, G_q] = flux_fn(q, U_q);
        const double w = weights[q] * std::abs(J_det(q, 0));

        for (int i = 0; i < n_basis; ++i) {
            const double dphidx = dphi_dx(q * n_basis + i, 0);
            const double dphidy = dphi_dx(q * n_basis + i, 1);
            for (int v = 0; v < n_vars; ++v) {
                R_vol(i, v) += sign * w * (dphidx * F_q[v] + dphidy * G_q[v]);
            }
        }
    }

    return R_vol;
}

}  // namespace dgfem
