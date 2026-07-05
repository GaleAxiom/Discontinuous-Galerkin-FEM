/**
 * @file euler_weak_formulation.cpp
 * @brief Implementation of Euler weak formulation
 */

#include "dgfem/weak_forms/euler_weak_formulation.hpp"

#include "dgfem/core/space.hpp"
#include "dgfem/solver/assembler.hpp"
#include "dgfem/weak_forms/quadrature_loop.hpp"

#include <cmath>

#include <iostream>
#include <stdexcept>

namespace dgfem {

EulerWeakFormulation::EulerWeakFormulation(double gamma)
    : gamma_(gamma), gamma_minus_one_(gamma - 1.0) {}

// Helper struct to avoid redundant conversions
struct FluxAndPrimitive {
    Vec4 F, G, W;
};

// Optimized: compute primitive variables AND fluxes in one pass
static FluxAndPrimitive get_fluxes_and_primitive(const Vec4& U, double gamma) {
    Vec4 W = conserved_to_primitive(U, gamma);
    double rho = W[0];
    double u = W[1];
    double v = W[2];
    double p = W[3];

    double rho_u = U[1];
    double rho_v = U[2];
    double E = U[3];

    Vec4 F{rho_u, rho * u * u + p, rho * u * v, u * (E + p)};
    Vec4 G{rho_v, rho * u * v, rho * v * v + p, v * (E + p)};

    return {F, G, W};
}

std::tuple<Vec4, Vec4> EulerWeakFormulation::get_fluxes(const Vec4& U) const {
    auto result = get_fluxes_and_primitive(U, gamma_);
    return {result.F, result.G};
}

Vec4 EulerWeakFormulation::rusanov_flux(const Vec4& U_L, const Vec4& U_R,
                                        const Vec2& normal) const {
    // OPTIMIZATION: Single conversion per state (was 2x before)
    auto [F_L, G_L, W_L] = get_fluxes_and_primitive(U_L, gamma_);
    auto [F_R, G_R, W_R] = get_fluxes_and_primitive(U_R, gamma_);

    Vec4 Fn_L = F_L * normal[0] + G_L * normal[1];
    Vec4 Fn_R = F_R * normal[0] + G_R * normal[1];

    double c_L = std::sqrt(gamma_ * W_L[3] / W_L[0]);  // speed of sound
    double c_R = std::sqrt(gamma_ * W_R[3] / W_R[0]);

    double un_L = W_L[1] * normal[0] + W_L[2] * normal[1];  // normal velocity
    double un_R = W_R[1] * normal[0] + W_R[2] * normal[1];

    double s_max = std::max(std::abs(un_L) + c_L, std::abs(un_R) + c_R);

    return 0.5 * (Fn_L + Fn_R) - 0.5 * s_max * (U_R - U_L);
}

DView2 EulerWeakFormulation::volume_residual(const DView2& u_coeffs_elem,
                                             const std::map<std::string, DView2>& elem_data,
                                             std::shared_ptr<DGSpace> dg_space) const {
    const DView1& weights = dg_space->get_volume_quad()->weights;
    const DView2& J_det = elem_data.at("J_det_vol");
    const DView2& phi = dg_space->get_volume_basis_values();
    const DView2& dphi_dx = elem_data.at("dphi_dx_vol");

    return accumulate_volume_residual(
        u_coeffs_elem, phi, dphi_dx, weights, J_det, /*sign=*/1.0,
        [this](int /*q*/, const Vec4& U_q) { return get_fluxes(U_q); });
}

std::tuple<DView2, DView2>
EulerWeakFormulation::interior_face_residual(const DView2& u_coeffs_L, const DView2& u_coeffs_R,
                                             const std::map<std::string, DView2>& face_data_L,
                                             const std::map<std::string, DView2>& face_data_R,
                                             std::shared_ptr<DGSpace> dg_space,
                                             const IView1& permutation) const {
    // --- Setup is the same ---
    const int n_basis = dg_space->get_basis()->get_n_basis();
    const int n_vars = static_cast<int>(u_coeffs_L.extent(1));  // This will be 4
    DView2 R_face_L("R_face_L", n_basis, n_vars);
    DView2 R_face_R("R_face_R", n_basis, n_vars);

    const DView2& weights = face_data_L.at("weights");
    const DView2& phi_L = face_data_L.at("phi");
    const DView2& phi_R = face_data_R.at("phi");
    const Vec2 normal = to_vec2(face_data_L.at("normal"));
    const double hF = face_data_L.at("length")(0, 0);
    const int n_quad = static_cast<int>(weights.extent(0));

    for (int q = 0; q < n_quad; ++q) {
        Vec4 U_L_q{0.0, 0.0, 0.0, 0.0};
        Vec4 U_R_q{0.0, 0.0, 0.0, 0.0};
        const int idx_R_q = (permutation.size() > 0) ? permutation[q] : q;

        // --- OPTIMIZATION 1: Manual loop for U Computation ---
        // Access coefficients directly to avoid .row() and .transpose() overhead.
        for (int i = 0; i < n_basis; ++i) {
            const double p_L = phi_L(q, i);
            const double p_R = phi_R(idx_R_q, i);
            // This inner loop over 'v' will be unrolled and vectorized by the compiler.
            for (int v = 0; v < n_vars; ++v) {
                U_L_q[v] += p_L * u_coeffs_L(i, v);
                U_R_q[v] += p_R * u_coeffs_R(i, v);
            }
        }

        // --- Flux calculation is unchanged ---
        const Vec4 H_q = rusanov_flux(U_L_q, U_R_q, normal);
        const double w_q_phys = weights(q, 0) * hF * 0.5;

        // --- OPTIMIZATION 2: Manual loop for Residual Update ---
        // Again, direct coefficient access is key.
        for (int i = 0; i < n_basis; ++i) {
            const double p_L = phi_L(q, i);
            const double p_R = phi_R(idx_R_q, i);
            const double scale_L = w_q_phys * p_L;
            const double scale_R = w_q_phys * p_R;
            // The compiler will vectorize this loop over 'v' perfectly.
            for (int v = 0; v < n_vars; ++v) {
                R_face_L(i, v) -= scale_L * H_q[v];
                R_face_R(i, v) += scale_R * H_q[v];
            }
        }
    }

    return std::make_tuple(R_face_L, R_face_R);
}

DView2 EulerWeakFormulation::boundary_face_residual(const DView2& u_coeffs,
                                                    const std::map<std::string, DView2>& face_data,
                                                    std::shared_ptr<BoundaryConditionEuler> bc,
                                                    std::shared_ptr<DGSpace> dg_space) const {
    const int n_basis = dg_space->get_basis()->get_n_basis();
    const int n_vars = static_cast<int>(u_coeffs.extent(1));
    DView2 R_face_bc("R_face_bc", n_basis, n_vars);

    const DView2& weights = face_data.at("weights");
    const DView2& phi = face_data.at("phi");
    const Vec2 normal = to_vec2(face_data.at("normal"));
    const double hF = face_data.at("length")(0, 0);
    const DView2& quad_points = face_data.at("quad_points");

    const int n_quad = static_cast<int>(weights.extent(0));

    for (int q = 0; q < n_quad; ++q) {
        // OPTIMIZATION: Use manual loop to avoid .row().transpose() overhead
        Vec4 U_L_q = Vec4{0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < n_basis; ++i) {
            const double phi_val = phi(q, i);
            for (int v = 0; v < n_vars; ++v) {
                U_L_q[v] += phi_val * u_coeffs(i, v);
            }
        }

        Vec4 W_L_q = conserved_to_primitive(U_L_q, gamma_);

        // Determine ghost state based on BC type
        Vec4 U_R_q{0.0, 0.0, 0.0, 0.0};
        Vec2 x_q = row2(quad_points, q);

        if (bc->get_type() == BCTypeEuler::FAR_FIELD) {
            U_R_q = bc->evaluate(x_q);
        } else if (bc->get_type() == BCTypeEuler::SLIP_WALL) {
            // Reflect normal velocity
            double un_L = W_L_q[1] * normal[0] + W_L_q[2] * normal[1];
            Vec2 u_norm_L = un_L * normal;
            Vec2 u_L{W_L_q[1], W_L_q[2]};
            Vec2 u_tan_L = u_L - u_norm_L;
            Vec2 u_R = u_tan_L - u_norm_L;

            Vec4 W_R_q{W_L_q[0], u_R[0], u_R[1], W_L_q[3]};
            U_R_q = primitive_to_conserved(W_R_q, gamma_);
        } else if (bc->get_type() == BCTypeEuler::NO_SLIP_WALL) {
            Vec4 W_bc = bc->evaluate(x_q);
            U_R_q = primitive_to_conserved(W_bc, gamma_);
        } else if (bc->get_type() == BCTypeEuler::PERIODIC) {
            throw std::runtime_error("PERIODIC BC encountered in boundary_face_residual - periodic "
                                     "boundaries should be treated as interior faces");
        } else {
            throw std::runtime_error("Unsupported BC type for Euler equations");
        }

        // Compute numerical flux
        const Vec4 H_q = rusanov_flux(U_L_q, U_R_q, normal);

        const double w_q_phys = weights(q, 0) * hF * 0.5;

        // OPTIMIZATION: Manual loop to avoid .row() overhead
        for (int i = 0; i < n_basis; ++i) {
            const double scale = w_q_phys * phi(q, i);
            for (int v = 0; v < n_vars; ++v) {
                R_face_bc(i, v) -= scale * H_q[v];
            }
        }
    }

    return R_face_bc;
}

void EulerWeakFormulation::assemble(DGAssembler& assembler,
                                    std::function<double(const Vec2&)> source_func,
                                    std::function<double(const Vec2&)> bc_func) const {
    // Euler equations use residual-based assembly, not matrix assembly
    // This method should not be called directly
    throw std::runtime_error("Euler equations use assemble_euler_residual() instead of assemble()");
}

// Utility functions for Euler equations
Vec4 primitive_to_conserved(const Vec4& primitive, double gamma) {
    const double rho = primitive[0];
    const double u = primitive[1];
    const double v = primitive[2];
    const double p = primitive[3];

    const double gamma_m1 = gamma - 1.0;
    const double E = p / gamma_m1 + 0.5 * rho * (u * u + v * v);

    Vec4 conserved;
    conserved[0] = rho;
    conserved[1] = rho * u;
    conserved[2] = rho * v;
    conserved[3] = E;

    return conserved;
}

Vec4 conserved_to_primitive(const Vec4& conserved, double gamma) {
    const double rho = conserved[0];
    const double rho_inv = 1.0 / rho;  // Single division

    const double u = conserved[1] * rho_inv;  // Multiply instead of divide
    const double v = conserved[2] * rho_inv;  // Multiply instead of divide

    const double gamma_m1 = gamma - 1.0;
    const double p = gamma_m1 * (conserved[3] - 0.5 * rho * (u * u + v * v));

    Vec4 primitive;
    primitive[0] = rho;
    primitive[1] = u;
    primitive[2] = v;
    primitive[3] = p;

    return primitive;
}

DView2
EulerWeakFormulation::viscous_volume_residual(const DView2& u_coeffs_elem,
                                              const std::map<std::string, DView2>& /*elem_data*/,
                                              std::shared_ptr<DGSpace> dg_space) const {
    return DView2("R", dg_space->get_basis()->get_n_basis(), get_n_vars());
}

std::tuple<DView2, DView2> EulerWeakFormulation::viscous_interior_face_residual(
    const DView2& u_coeffs_L, const DView2& u_coeffs_R,
    const std::map<std::string, DView2>& /*face_data_L*/,
    const std::map<std::string, DView2>& /*face_data_R*/, std::shared_ptr<DGSpace> dg_space,
    const IView1& /*permutation*/) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = static_cast<int>(u_coeffs_L.extent(1));
    return {DView2("R_L", n_basis, n_vars), DView2("R_R", n_basis, n_vars)};
}

DView2 EulerWeakFormulation::viscous_boundary_face_residual(
    const DView2& /*u_coeffs*/, const std::map<std::string, DView2>& /*face_data*/,
    std::shared_ptr<BoundaryConditionEuler> /*bc*/, std::shared_ptr<DGSpace> dg_space) const {
    return DView2("R", dg_space->get_basis()->get_n_basis(), get_n_vars());
}

}  // namespace dgfem
