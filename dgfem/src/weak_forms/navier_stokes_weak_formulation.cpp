/**
 * @file navier_stokes_weak_formulation.cpp
 * @brief Implementation of laminar Navier-Stokes weak formulation with SIPG-style viscous fluxes
 */

#include "dgfem/weak_forms/navier_stokes_weak_formulation.hpp"

#include "dgfem/core/space.hpp"
#include "dgfem/weak_forms/quadrature_loop.hpp"

#include <cmath>

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace dgfem {

NavierStokesWeakFormulation::NavierStokesWeakFormulation(double gamma, double dynamic_viscosity,
                                                         double prandtl, double penalty_prefactor)
    : EulerWeakFormulation(gamma), mu_(dynamic_viscosity), prandtl_(prandtl),
      sigma0_(penalty_prefactor), gas_constant_(1.0) {
    if (mu_ <= 0.0) {
        throw std::invalid_argument(
            "Dynamic viscosity must be positive for Navier-Stokes weak formulation");
    }
    if (prandtl_ <= 0.0) {
        throw std::invalid_argument(
            "Prandtl number must be positive for Navier-Stokes weak formulation");
    }
    if (sigma0_ <= 0.0) {
        throw std::invalid_argument(
            "Penalty prefactor must be positive for Navier-Stokes weak formulation");
    }
}

NavierStokesWeakFormulation::PrimitiveGradientData
NavierStokesWeakFormulation::compute_primitive_gradients(const Vec4& U,
                                                         const GradU4& grad_U) const {
    PrimitiveGradientData data;
    double rho_safe = std::max(U[0], 1e-12);

    // Clamp density before converting to primitives, not after: conserved_to_primitive divides
    // momentum/energy by whatever density it's given, so u/v/p (and everything derived from
    // them below) must use the same rho_safe as the gradient formulas a few lines down --
    // otherwise data.u/data.v come from one density and their gradients from another.
    Vec4 U_safe{rho_safe, U[1], U[2], U[3]};
    Vec4 W = conserved_to_primitive(U_safe, get_gamma());
    data.rho = W[0];
    data.u = W[1];
    data.v = W[2];
    data.p = W[3];

    data.grad_rho = grad_U[0];
    Vec2 grad_rhou = grad_U[1];
    Vec2 grad_rhov = grad_U[2];
    Vec2 grad_E = grad_U[3];

    double rho_inv = 1.0 / rho_safe;
    data.grad_u = (grad_rhou - data.u * data.grad_rho) * rho_inv;
    data.grad_v = (grad_rhov - data.v * data.grad_rho) * rho_inv;

    double kinetic_sq = data.u * data.u + data.v * data.v;
    Vec2 grad_velocity_norm = 2.0 * data.u * data.grad_u + 2.0 * data.v * data.grad_v;
    Vec2 momentum_term = data.rho * (data.u * data.grad_u + data.v * data.grad_v);
    Vec2 grad_p = (get_gamma() - 1.0) * (grad_E - 0.5 * kinetic_sq * data.grad_rho - momentum_term);

    double rho_sq = data.rho * data.rho;
    data.temperature = data.p / (data.rho * gas_constant_);
    data.grad_T = (data.rho * grad_p - data.p * data.grad_rho) * (1.0 / (rho_sq * gas_constant_));

    data.divergence = data.grad_u[0] + data.grad_v[1];
    double lambda = -2.0 * mu_ / 3.0;

    data.tau_xx = 2.0 * mu_ * data.grad_u[0] + lambda * data.divergence;
    data.tau_yy = 2.0 * mu_ * data.grad_v[1] + lambda * data.divergence;
    data.tau_xy = mu_ * (data.grad_u[1] + data.grad_v[0]);

    double kappa = mu_ * get_gamma() / (prandtl_ * (get_gamma() - 1.0));
    data.q_x = -kappa * data.grad_T[0];
    data.q_y = -kappa * data.grad_T[1];

    return data;
}

std::pair<Vec4, Vec4>
NavierStokesWeakFormulation::compute_viscous_fluxes(const Vec4& U, const GradU4& grad_U) const {
    PrimitiveGradientData data = compute_primitive_gradients(U, grad_U);

    Vec4 Fv = Vec4{0.0, 0.0, 0.0, 0.0};
    Vec4 Gv = Vec4{0.0, 0.0, 0.0, 0.0};

    Fv[1] = data.tau_xx;
    Fv[2] = data.tau_xy;
    Fv[3] = data.u * data.tau_xx + data.v * data.tau_xy + data.q_x;

    Gv[1] = data.tau_xy;
    Gv[2] = data.tau_yy;
    Gv[3] = data.u * data.tau_xy + data.v * data.tau_yy + data.q_y;

    return {Fv, Gv};
}

DView2
NavierStokesWeakFormulation::viscous_volume_residual(const DView2& u_coeffs_elem,
                                                     const std::map<std::string, DView2>& elem_data,
                                                     std::shared_ptr<DGSpace> dg_space) const {
    const int n_basis = dg_space->get_basis()->get_n_basis();
    const DView1& weights = dg_space->get_volume_quad()->weights;
    const DView2& phi = dg_space->get_volume_basis_values();
    const DView2& dphi_dx = elem_data.at("dphi_dx_vol");
    const DView2& detJ = elem_data.at("J_det_vol");

    // Unlike Euler's inviscid flux (a pure function of U_q), the viscous flux also
    // needs grad(U) at each quad point, so the functor closes over dphi_dx/u_coeffs_elem
    // and rebuilds it from `q` before calling compute_viscous_fluxes.
    return accumulate_volume_residual(
        u_coeffs_elem, phi, dphi_dx, weights, detJ, /*sign=*/-1.0,
        [this, &dphi_dx, &u_coeffs_elem, n_basis](int q, const Vec4& U_q) {
            GradU4 grad_U{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
            for (int i = 0; i < n_basis; ++i) {
                Vec2 grad_phi_i{dphi_dx(q * n_basis + i, 0), dphi_dx(q * n_basis + i, 1)};
                for (int v = 0; v < 4; ++v) {
                    grad_U[v][0] += u_coeffs_elem(i, v) * grad_phi_i[0];
                    grad_U[v][1] += u_coeffs_elem(i, v) * grad_phi_i[1];
                }
            }
            return compute_viscous_fluxes(U_q, grad_U);
        });
}

std::tuple<DView2, DView2> NavierStokesWeakFormulation::viscous_interior_face_residual(
    const DView2& u_coeffs_L, const DView2& u_coeffs_R,
    const std::map<std::string, DView2>& face_data_L,
    const std::map<std::string, DView2>& face_data_R, std::shared_ptr<DGSpace> dg_space,
    const IView1& permutation) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = get_n_vars();
    DView2 R_face_L("R_face_L", n_basis, n_vars);
    DView2 R_face_R("R_face_R", n_basis, n_vars);

    const DView2& weights = face_data_L.at("weights");
    const DView2& phi_L = face_data_L.at("phi");
    const DView2& phi_R = face_data_R.at("phi");
    const DView2& grad_phi_L = face_data_L.at("dphi_dx_face");
    const DView2& grad_phi_R = face_data_R.at("dphi_dx_face");
    Vec2 normal = to_vec2(face_data_L.at("normal"));
    double face_length = face_data_L.at("length")(0, 0);

    int n_quad = static_cast<int>(weights.extent(0));
    int order = dg_space->get_order();
    double sigma = compute_penalty_parameter(order, face_length);

    for (int q = 0; q < n_quad; ++q) {
        int qR = permutation.size() > 0 ? permutation[q] : q;

        Vec4 U_L_q{0.0, 0.0, 0.0, 0.0};
        Vec4 U_R_q{0.0, 0.0, 0.0, 0.0};

        for (int i = 0; i < n_basis; ++i) {
            double p_L = phi_L(q, i);
            double p_R = phi_R(qR, i);
            for (int v = 0; v < n_vars; ++v) {
                U_L_q[v] += p_L * u_coeffs_L(i, v);
                U_R_q[v] += p_R * u_coeffs_R(i, v);
            }
        }

        GradU4 grad_UL{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
        GradU4 grad_UR{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};

        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_i_L = row2(grad_phi_L, q * n_basis + i);
            Vec2 grad_phi_i_R = row2(grad_phi_R, qR * n_basis + i);
            for (int v = 0; v < n_vars; ++v) {
                grad_UL[v][0] += u_coeffs_L(i, v) * grad_phi_i_L[0];
                grad_UL[v][1] += u_coeffs_L(i, v) * grad_phi_i_L[1];
                grad_UR[v][0] += u_coeffs_R(i, v) * grad_phi_i_R[0];
                grad_UR[v][1] += u_coeffs_R(i, v) * grad_phi_i_R[1];
            }
        }

        auto [Fv_L, Gv_L] = compute_viscous_fluxes(U_L_q, grad_UL);
        auto [Fv_R, Gv_R] = compute_viscous_fluxes(U_R_q, grad_UR);

        Vec4 flux_avg = 0.5 * (Fv_L * normal[0] + Gv_L * normal[1]);
        flux_avg += 0.5 * (Fv_R * (-normal[0]) + Gv_R * (-normal[1]));

        Vec4 penalty = sigma * (U_R_q - U_L_q);
        Vec4 Fn = flux_avg - penalty;

        double w_q = weights(q, 0) * face_length * 0.5;

        for (int i = 0; i < n_basis; ++i) {
            double scale_L = w_q * phi_L(q, i);
            double scale_R = w_q * phi_R(qR, i);
            for (int v = 0; v < n_vars; ++v) {
                R_face_L(i, v) -= scale_L * Fn[v];
                R_face_R(i, v) += scale_R * Fn[v];
            }
        }
    }

    return {R_face_L, R_face_R};
}

DView2 NavierStokesWeakFormulation::viscous_boundary_face_residual(
    const DView2& u_coeffs, const std::map<std::string, DView2>& face_data,
    std::shared_ptr<BoundaryConditionEuler> bc, std::shared_ptr<DGSpace> dg_space) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = get_n_vars();
    DView2 R_face("R_face", n_basis, n_vars);

    const DView2& weights = face_data.at("weights");
    const DView2& phi = face_data.at("phi");
    const DView2& grad_phi = face_data.at("dphi_dx_face");
    Vec2 normal = to_vec2(face_data.at("normal"));
    double face_length = face_data.at("length")(0, 0);

    int order = dg_space->get_order();
    double sigma = compute_penalty_parameter(order, face_length);
    int n_quad = static_cast<int>(weights.extent(0));

    for (int q = 0; q < n_quad; ++q) {
        Vec4 U_L_q{0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < n_basis; ++i) {
            double p = phi(q, i);
            for (int v = 0; v < n_vars; ++v) {
                U_L_q[v] += p * u_coeffs(i, v);
            }
        }

        GradU4 grad_UL{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_i = row2(grad_phi, q * n_basis + i);
            for (int v = 0; v < n_vars; ++v) {
                grad_UL[v][0] += u_coeffs(i, v) * grad_phi_i[0];
                grad_UL[v][1] += u_coeffs(i, v) * grad_phi_i[1];
            }
        }

        auto [Fv_L, Gv_L] = compute_viscous_fluxes(U_L_q, grad_UL);
        Vec4 U_bc = U_L_q;
        if (bc) {
            Vec2 x_q = row2(face_data.at("quad_points"), q);

            Vec4 bc_data = bc->evaluate(x_q);
            switch (bc->get_type()) {
            case BCTypeEuler::NO_SLIP_WALL:
                U_bc = primitive_to_conserved(bc_data, get_gamma());
                break;
            case BCTypeEuler::FAR_FIELD:
            case BCTypeEuler::SLIP_WALL:
            case BCTypeEuler::PERIODIC:
            default:
                U_bc = bc_data;
                break;
            }
        }

        Vec4 penalty = sigma * (U_bc - U_L_q);
        Vec4 Fn = (Fv_L * normal[0] + Gv_L * normal[1]) - penalty;
        double w_q = weights(q, 0) * face_length * 0.5;

        for (int i = 0; i < n_basis; ++i) {
            double scale = w_q * phi(q, i);
            for (int v = 0; v < n_vars; ++v) {
                R_face(i, v) -= scale * Fn[v];
            }
        }
    }

    return R_face;
}

double NavierStokesWeakFormulation::compute_penalty_parameter(int p, double h) const {
    double h_safe = std::max(h, 1e-12);
    return sigma0_ * mu_ * (p + 1) * (p + 1) / h_safe;
}

}  // namespace dgfem
