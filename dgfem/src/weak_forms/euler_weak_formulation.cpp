/**
 * @file euler_weak_formulation.cpp
 * @brief Implementation of Euler weak formulation
 */

#include "dgfem/weak_forms/euler_weak_formulation.hpp"

#include "dgfem/core/space.hpp"
#include "dgfem/solver/assembler.hpp"

#include <cmath>

#include <iostream>
#include <stdexcept>

namespace dgfem {

EulerWeakFormulation::EulerWeakFormulation(double gamma)
    : gamma_(gamma), gamma_minus_one_(gamma - 1.0) {}

// Helper struct to avoid redundant conversions
struct FluxAndPrimitive {
    Eigen::Vector4d F, G, W;
};

// Optimized: compute primitive variables AND fluxes in one pass
static FluxAndPrimitive get_fluxes_and_primitive(const Eigen::Vector4d& U, double gamma) {
    Eigen::Vector4d W = conserved_to_primitive(U, gamma);
    double rho = W[0];
    double u = W[1];
    double v = W[2];
    double p = W[3];

    double rho_u = U[1];
    double rho_v = U[2];
    double E = U[3];

    Eigen::Vector4d F, G;
    F << rho_u, rho * u * u + p, rho * u * v, u * (E + p);
    G << rho_v, rho * u * v, rho * v * v + p, v * (E + p);

    return {F, G, W};
}

std::tuple<Eigen::Vector4d, Eigen::Vector4d>
EulerWeakFormulation::get_fluxes(const Eigen::Vector4d& U) const {
    auto result = get_fluxes_and_primitive(U, gamma_);
    return {result.F, result.G};
}

Eigen::Vector4d EulerWeakFormulation::rusanov_flux(const Eigen::Vector4d& U_L,
                                                   const Eigen::Vector4d& U_R,
                                                   const Eigen::Vector2d& normal) const {
    // OPTIMIZATION: Single conversion per state (was 2x before)
    auto [F_L, G_L, W_L] = get_fluxes_and_primitive(U_L, gamma_);
    auto [F_R, G_R, W_R] = get_fluxes_and_primitive(U_R, gamma_);

    Eigen::Vector4d Fn_L = F_L * normal[0] + G_L * normal[1];
    Eigen::Vector4d Fn_R = F_R * normal[0] + G_R * normal[1];

    double c_L = std::sqrt(gamma_ * W_L[3] / W_L[0]);  // speed of sound
    double c_R = std::sqrt(gamma_ * W_R[3] / W_R[0]);

    double un_L = W_L[1] * normal[0] + W_L[2] * normal[1];  // normal velocity
    double un_R = W_R[1] * normal[0] + W_R[2] * normal[1];

    double s_max = std::max(std::abs(un_L) + c_L, std::abs(un_R) + c_R);

    return 0.5 * (Fn_L + Fn_R) - 0.5 * s_max * (U_R - U_L);
}

Eigen::MatrixXd
EulerWeakFormulation::volume_residual(const Eigen::MatrixXd& u_coeffs_elem,
                                      const std::map<std::string, Eigen::MatrixXd>& elem_data,
                                      std::shared_ptr<DGSpace> dg_space) const {
    const int n_basis = dg_space->get_basis()->get_n_basis();
    const int n_vars = u_coeffs_elem.cols();  // Should be 4
    Eigen::MatrixXd R_vol = Eigen::MatrixXd::Zero(n_basis, n_vars);

    // Get references to data to avoid map lookups in the loop
    const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
    const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");
    const Eigen::MatrixXd& phi = dg_space->get_volume_basis_values();
    const Eigen::MatrixXd& dphi_dx = elem_data.at("dphi_dx_vol");

    const int n_quad = static_cast<int>(weights.size());

    // OPTIMIZATION: Pre-compute physical weights (avoid repeated computation)
    const Eigen::ArrayXd w_phys = weights.array() * J_det.array().abs();

    // Pre-calculate all solution values at all quadrature points
    // This is one large, efficient matrix-matrix multiplication.
    // U_quad_points has shape (n_quad, n_vars), where each row is a U_q
    const Eigen::MatrixXd U_quad_points = phi * u_coeffs_elem;

    for (int q = 0; q < n_quad; ++q) {
        // U_q is now a simple row lookup
        const Eigen::Vector4d U_q = U_quad_points.row(q).transpose();

        // Get fluxes
        auto [F_q, G_q] = get_fluxes(U_q);

        // Get all gradient components for basis functions at this quad point 'q'
        // This is a view (no data copied) into the dphi_dx matrix.
        // It has shape (n_basis, 2)
        const auto grad_phi_q = dphi_dx.block(q * n_basis, 0, n_basis, 2);

        const double w = w_phys[q];

        // Vectorized outer-product update
        R_vol += w * (grad_phi_q.col(0) * F_q.transpose() + grad_phi_q.col(1) * G_q.transpose());
    }

    return R_vol;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> EulerWeakFormulation::interior_face_residual(
    const Eigen::MatrixXd& u_coeffs_L, const Eigen::MatrixXd& u_coeffs_R,
    const std::map<std::string, Eigen::MatrixXd>& face_data_L,
    const std::map<std::string, Eigen::MatrixXd>& face_data_R, std::shared_ptr<DGSpace> dg_space,
    const Eigen::VectorXi& permutation) const {
    // --- Setup is the same ---
    const int n_basis = dg_space->get_basis()->get_n_basis();
    const int n_vars = u_coeffs_L.cols();  // This will be 4
    Eigen::MatrixXd R_face_L = Eigen::MatrixXd::Zero(n_basis, n_vars);
    Eigen::MatrixXd R_face_R = Eigen::MatrixXd::Zero(n_basis, n_vars);

    const Eigen::VectorXd& weights = face_data_L.at("weights");
    const Eigen::MatrixXd& phi_L = face_data_L.at("phi");
    const Eigen::MatrixXd& phi_R = face_data_R.at("phi");
    const Eigen::Vector2d& normal = face_data_L.at("normal").col(0);
    const double hF = face_data_L.at("length")(0, 0);
    const int n_quad = weights.size();

    // U_L_q and U_R_q are small and will live on the stack.
    Eigen::Vector4d U_L_q, U_R_q;

    for (int q = 0; q < n_quad; ++q) {
        U_L_q.setZero();
        U_R_q.setZero();
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
        const Eigen::Vector4d H_q = rusanov_flux(U_L_q, U_R_q, normal);
        const double w_q_phys = weights[q] * hF * 0.5;

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

Eigen::MatrixXd EulerWeakFormulation::boundary_face_residual(
    const Eigen::MatrixXd& u_coeffs, const std::map<std::string, Eigen::MatrixXd>& face_data,
    std::shared_ptr<BoundaryConditionEuler> bc, std::shared_ptr<DGSpace> dg_space) const {
    const int n_basis = dg_space->get_basis()->get_n_basis();
    const int n_vars = u_coeffs.cols();
    Eigen::MatrixXd R_face_bc = Eigen::MatrixXd::Zero(n_basis, n_vars);

    const Eigen::VectorXd& weights = face_data.at("weights").col(0);
    const Eigen::MatrixXd& phi = face_data.at("phi");
    const Eigen::Vector2d& normal = face_data.at("normal").col(0);
    const double hF = face_data.at("length")(0, 0);
    const Eigen::MatrixXd& quad_points_mat = face_data.at("quad_points");

    // OPTIMIZATION: Zero-copy reshape using Eigen::Map
    const int n_quad = static_cast<int>(weights.size());
    const double* qp_data = quad_points_mat.data();
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>> quad_points(
        qp_data, n_quad, 2);

    for (int q = 0; q < n_quad; ++q) {
        // OPTIMIZATION: Use manual loop to avoid .row().transpose() overhead
        Eigen::Vector4d U_L_q = Eigen::Vector4d::Zero();
        for (int i = 0; i < n_basis; ++i) {
            const double phi_val = phi(q, i);
            for (int v = 0; v < n_vars; ++v) {
                U_L_q[v] += phi_val * u_coeffs(i, v);
            }
        }

        Eigen::Vector4d W_L_q = conserved_to_primitive(U_L_q, gamma_);

        // Determine ghost state based on BC type
        Eigen::Vector4d U_R_q;
        Eigen::Vector2d x_q = quad_points.row(q);

        if (bc->get_type() == BCTypeEuler::FAR_FIELD) {
            U_R_q = bc->evaluate(x_q);
        } else if (bc->get_type() == BCTypeEuler::SLIP_WALL) {
            // Reflect normal velocity
            double un_L = W_L_q[1] * normal[0] + W_L_q[2] * normal[1];
            Eigen::Vector2d u_norm_L = un_L * normal;
            Eigen::Vector2d u_L(W_L_q[1], W_L_q[2]);
            Eigen::Vector2d u_tan_L = u_L - u_norm_L;
            Eigen::Vector2d u_R = u_tan_L - u_norm_L;

            Eigen::Vector4d W_R_q;
            W_R_q << W_L_q[0], u_R[0], u_R[1], W_L_q[3];
            U_R_q = primitive_to_conserved(W_R_q, gamma_);
        } else if (bc->get_type() == BCTypeEuler::NO_SLIP_WALL) {
            Eigen::Vector4d W_bc = bc->evaluate(x_q);
            U_R_q = primitive_to_conserved(W_bc, gamma_);
        } else if (bc->get_type() == BCTypeEuler::INLET) {
            Eigen::Vector4d W_bc = bc->evaluate(x_q);
            U_R_q = primitive_to_conserved(W_bc, gamma_);
        } else if (bc->get_type() == BCTypeEuler::OUTLET) {
            Eigen::Vector4d W_bc = bc->evaluate(x_q);
            Eigen::Vector4d W_R_q = W_L_q;
            W_R_q[0] = W_bc[0];
            W_R_q[1] = W_bc[1];
            W_R_q[2] = W_bc[2];
            W_R_q[3] = W_bc[3];
            U_R_q = primitive_to_conserved(W_R_q, gamma_);
        } else if (bc->get_type() == BCTypeEuler::PERIODIC) {
            throw std::runtime_error("PERIODIC BC encountered in boundary_face_residual - periodic "
                                     "boundaries should be treated as interior faces");
        } else {
            throw std::runtime_error("Unsupported BC type for Euler equations");
        }

        // Compute numerical flux
        const Eigen::Vector4d H_q = rusanov_flux(U_L_q, U_R_q, normal);

        const double w_q_phys = weights[q] * hF * 0.5;

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
                                    std::function<double(const Eigen::Vector2d&)> source_func,
                                    std::function<double(const Eigen::Vector2d&)> bc_func) const {
    // Euler equations use residual-based assembly, not matrix assembly
    // This method should not be called directly
    throw std::runtime_error("Euler equations use assemble_euler_residual() instead of assemble()");
}

// Utility functions for Euler equations
Eigen::Vector4d primitive_to_conserved(const Eigen::Vector4d& primitive, double gamma) {
    const double rho = primitive[0];
    const double u = primitive[1];
    const double v = primitive[2];
    const double p = primitive[3];

    const double gamma_m1 = gamma - 1.0;
    const double E = p / gamma_m1 + 0.5 * rho * (u * u + v * v);

    Eigen::Vector4d conserved;
    conserved[0] = rho;
    conserved[1] = rho * u;
    conserved[2] = rho * v;
    conserved[3] = E;

    return conserved;
}

Eigen::Vector4d conserved_to_primitive(const Eigen::Vector4d& conserved, double gamma) {
    const double rho = conserved[0];
    const double rho_inv = 1.0 / rho;  // Single division

    const double u = conserved[1] * rho_inv;  // Multiply instead of divide
    const double v = conserved[2] * rho_inv;  // Multiply instead of divide

    const double gamma_m1 = gamma - 1.0;
    const double p = gamma_m1 * (conserved[3] - 0.5 * rho * (u * u + v * v));

    Eigen::Vector4d primitive;
    primitive[0] = rho;
    primitive[1] = u;
    primitive[2] = v;
    primitive[3] = p;

    return primitive;
}

Eigen::MatrixXd EulerWeakFormulation::viscous_volume_residual(
    const Eigen::MatrixXd& u_coeffs_elem,
    const std::map<std::string, Eigen::MatrixXd>& /*elem_data*/,
    std::shared_ptr<DGSpace> dg_space) const {
    return Eigen::MatrixXd::Zero(dg_space->get_basis()->get_n_basis(), get_n_vars());
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> EulerWeakFormulation::viscous_interior_face_residual(
    const Eigen::MatrixXd& u_coeffs_L, const Eigen::MatrixXd& u_coeffs_R,
    const std::map<std::string, Eigen::MatrixXd>& /*face_data_L*/,
    const std::map<std::string, Eigen::MatrixXd>& /*face_data_R*/,
    std::shared_ptr<DGSpace> dg_space, const Eigen::VectorXi& /*permutation*/) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = u_coeffs_L.cols();
    return {Eigen::MatrixXd::Zero(n_basis, n_vars), Eigen::MatrixXd::Zero(n_basis, n_vars)};
}

Eigen::MatrixXd EulerWeakFormulation::viscous_boundary_face_residual(
    const Eigen::MatrixXd& /*u_coeffs*/,
    const std::map<std::string, Eigen::MatrixXd>& /*face_data*/,
    std::shared_ptr<BoundaryConditionEuler> /*bc*/, std::shared_ptr<DGSpace> dg_space) const {
    return Eigen::MatrixXd::Zero(dg_space->get_basis()->get_n_basis(), get_n_vars());
}

}  // namespace dgfem
