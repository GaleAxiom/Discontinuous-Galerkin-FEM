/**
 * @file riemann_solver.hpp
 * @brief Exact solution of the 1D Euler Riemann problem (Toro's iterative solver), plus a
 * helper to compute DG solution error against it.
 *
 * Promoted out of sod_shock_tube_example.cpp (originally a private struct there) so both the
 * example and the Sod shock tube e2e test can share one implementation.
 */

#pragma once

#include "dgfem/core/mesh.hpp"
#include "dgfem/kokkos_math.hpp"
#include "dgfem/weak_forms/euler_weak_formulation.hpp"

#include <cmath>

#include <algorithm>
#include <array>
#include <memory>
#include <utility>

namespace dgfem {

/**
 * @brief Exact solution of the 1D Riemann problem (Toro's iterative solver).
 *
 * Standard closed-form/iterative construction: Newton's method on the pressure function (see
 * e.g. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", ch. 4).
 */
struct ExactRiemannSolution {
    double gamma;
    double rho_L, u_L, p_L, c_L;
    double rho_R, u_R, p_R, c_R;
    double x0;
    double p_star = 0.0;
    double u_star = 0.0;

    ExactRiemannSolution(double gamma_, double rho_L_, double u_L_, double p_L_, double rho_R_,
                         double u_R_, double p_R_, double x0_)
        : gamma(gamma_), rho_L(rho_L_), u_L(u_L_), p_L(p_L_), rho_R(rho_R_), u_R(u_R_), p_R(p_R_),
          x0(x0_) {
        c_L = std::sqrt(gamma * p_L / rho_L);
        c_R = std::sqrt(gamma * p_R / rho_R);
        solve_star_state();
    }

    [[nodiscard]] double f_branch(double p, double rho_K, double p_K, double c_K) const {
        if (p > p_K) {
            double a_k = 2.0 / ((gamma + 1.0) * rho_K);
            double b_k = (gamma - 1.0) / (gamma + 1.0) * p_K;
            return (p - p_K) * std::sqrt(a_k / (p + b_k));
        }
        return (2.0 * c_K / (gamma - 1.0)) *
               (std::pow(p / p_K, (gamma - 1.0) / (2.0 * gamma)) - 1.0);
    }

    [[nodiscard]] double f_branch_deriv(double p, double rho_K, double p_K, double c_K) const {
        if (p > p_K) {
            double a_k = 2.0 / ((gamma + 1.0) * rho_K);
            double b_k = (gamma - 1.0) / (gamma + 1.0) * p_K;
            return std::sqrt(a_k / (b_k + p)) * (1.0 - (p - p_K) / (2.0 * (b_k + p)));
        }
        return (1.0 / (rho_K * c_K)) * std::pow(p / p_K, -(gamma + 1.0) / (2.0 * gamma));
    }

    [[nodiscard]] double total_f(double p) const {
        return f_branch(p, rho_L, p_L, c_L) + f_branch(p, rho_R, p_R, c_R) + (u_R - u_L);
    }

    [[nodiscard]] double total_f_deriv(double p) const {
        return f_branch_deriv(p, rho_L, p_L, c_L) + f_branch_deriv(p, rho_R, p_R, c_R);
    }

    void solve_star_state() {
        double p = 0.5 * (p_L + p_R);
        for (int iter = 0; iter < 50; ++iter) {
            double f = total_f(p);
            double fp = total_f_deriv(p);
            double p_new = p - f / fp;
            if (p_new < 1e-8) {
                p_new = 1e-8;
            }
            if (std::abs(p_new - p) < 1e-12 * std::abs(p_new)) {
                p = p_new;
                break;
            }
            p = p_new;
        }
        p_star = p;
        u_star = 0.5 * (u_L + u_R) +
                0.5 * (f_branch(p_star, rho_R, p_R, c_R) - f_branch(p_star, rho_L, p_L, c_L));
    }

    // Returns (rho, u, p) at physical position x and time t > 0.
    [[nodiscard]] std::array<double, 3> sample(double x, double t) const {
        double xi = (x - x0) / std::max(t, 1e-12);

        if (xi <= u_star) {
            // Left of the contact discontinuity.
            if (p_star <= p_L) {
                // Left rarefaction fan.
                double c_star_L = c_L * std::pow(p_star / p_L, (gamma - 1.0) / (2.0 * gamma));
                double s_head = u_L - c_L;
                double s_tail = u_star - c_star_L;
                if (xi <= s_head) {
                    return {rho_L, u_L, p_L};
                }
                if (xi <= s_tail) {
                    double c =
                        ((gamma - 1.0) / (gamma + 1.0)) * (u_L + 2.0 * c_L / (gamma - 1.0) - xi);
                    double u = (2.0 / (gamma + 1.0)) * (c_L + 0.5 * (gamma - 1.0) * u_L + xi);
                    double rho = rho_L * std::pow(c / c_L, 2.0 / (gamma - 1.0));
                    double p = p_L * std::pow(c / c_L, 2.0 * gamma / (gamma - 1.0));
                    return {rho, u, p};
                }
                double rho_star_L = rho_L * std::pow(p_star / p_L, 1.0 / gamma);
                return {rho_star_L, u_star, p_star};
            }
            // Left shock.
            double rho_star_L = rho_L * ((p_star / p_L) + (gamma - 1.0) / (gamma + 1.0)) /
                                ((gamma - 1.0) / (gamma + 1.0) * (p_star / p_L) + 1.0);
            double s_shock = u_L - c_L * std::sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) +
                                                   (gamma - 1.0) / (2.0 * gamma));
            if (xi <= s_shock) {
                return {rho_L, u_L, p_L};
            }
            return {rho_star_L, u_star, p_star};
        }

        // Right of the contact discontinuity.
        if (p_star >= p_R) {
            // Right shock.
            double rho_star_R = rho_R * ((p_star / p_R) + (gamma - 1.0) / (gamma + 1.0)) /
                                ((gamma - 1.0) / (gamma + 1.0) * (p_star / p_R) + 1.0);
            double s_shock = u_R + c_R * std::sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) +
                                                   (gamma - 1.0) / (2.0 * gamma));
            if (xi >= s_shock) {
                return {rho_R, u_R, p_R};
            }
            return {rho_star_R, u_star, p_star};
        }
        // Right rarefaction fan.
        double c_star_R = c_R * std::pow(p_star / p_R, (gamma - 1.0) / (2.0 * gamma));
        double s_head = u_R + c_R;
        double s_tail = u_star + c_star_R;
        if (xi >= s_head) {
            return {rho_R, u_R, p_R};
        }
        if (xi >= s_tail) {
            double c = (2.0 * c_R - (gamma - 1.0) * (u_R - xi)) / (gamma + 1.0);
            double u = (-2.0 * c_R + (gamma - 1.0) * u_R + 2.0 * xi) / (gamma + 1.0);
            double rho = rho_R * std::pow(c / c_R, 2.0 / (gamma - 1.0));
            double p = p_R * std::pow(c / c_R, 2.0 * gamma / (gamma - 1.0));
            return {rho, u, p};
        }
        double rho_star_R = rho_R * std::pow(p_star / p_R, 1.0 / gamma);
        return {rho_star_R, u_star, p_star};
    }
};

/**
 * @brief L2 relative error and max pointwise error for one primitive variable at time t,
 * comparing a flat (n_elem x (n_basis*4)) Euler conserved-state solution against the exact
 * Riemann solution. var_idx indexes primitive variables (0=rho, 1=u, 2=v, 3=p).
 */
[[nodiscard]] inline std::pair<double, double>
compute_riemann_solution_error(const std::shared_ptr<DGMesh>& mesh, const DView2& numerical_sol,
                               const ExactRiemannSolution& exact, double gamma, double t,
                               int var_idx) {
    auto space = mesh->get_dg_space();
    auto mapping = space->get_mapping();
    const auto& phi = space->get_volume_basis_values();
    const auto& quad_pts = space->get_volume_quad()->points;
    const auto& quad_wts = space->get_volume_quad()->weights;

    int n_basis = space->get_basis()->get_n_basis();
    double error_sq = 0.0;
    double norm_sq = 0.0;
    double max_error = 0.0;

    for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        const DView2& J_det = elem_data.at("J_det_vol");
        DView2 vertices = mesh->get_element_vertices(elem);

        for (int q = 0; q < static_cast<int>(quad_wts.size()); ++q) {
            Vec2 x_phys = mapping->map_to_physical(vertices, row2(quad_pts, q));

            Vec4 U_num{0.0, 0.0, 0.0, 0.0};
            for (int i = 0; i < n_basis; ++i) {
                for (int v = 0; v < 4; ++v) {
                    U_num[v] += numerical_sol(elem, i * 4 + v) * phi(q, i);
                }
            }
            Vec4 W_num = conserved_to_primitive(U_num, gamma);

            auto [rho_ex, u_ex, p_ex] = exact.sample(x_phys[0], t);
            std::array<double, 4> w_exact{rho_ex, u_ex, 0.0, p_ex};

            double weight = quad_wts(q) * std::abs(J_det(q, 0));
            double diff = W_num[var_idx] - w_exact[var_idx];
            error_sq += diff * diff * weight;
            norm_sq += w_exact[var_idx] * w_exact[var_idx] * weight;
            max_error = std::max(max_error, std::abs(diff));
        }
    }

    double l2_rel = (norm_sq < 1e-14) ? std::sqrt(error_sq) : std::sqrt(error_sq / norm_sq);
    return {l2_rel, max_error};
}

}  // namespace dgfem
