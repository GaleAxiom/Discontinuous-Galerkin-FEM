/**
 * @file navier_stokes_weak_formulation.cpp
 * @brief Implementation of laminar Navier-Stokes weak formulation with SIPG-style viscous fluxes
 */

#include "dgfem/weak_forms/navier_stokes_weak_formulation.hpp"

#include "dgfem/core/space.hpp"

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
NavierStokesWeakFormulation::compute_primitive_gradients(
    const Eigen::Vector4d& U, const Eigen::Matrix<double, 4, 2>& grad_U) const {
    PrimitiveGradientData data;
    double rho = U[0];
    double rho_safe = std::max(rho, 1e-12);

    Eigen::Vector4d W = conserved_to_primitive(U, get_gamma());
    data.rho = W[0];
    data.u = W[1];
    data.v = W[2];
    data.p = W[3];

    data.grad_rho = grad_U.row(0).transpose();
    Eigen::Vector2d grad_rhou = grad_U.row(1).transpose();
    Eigen::Vector2d grad_rhov = grad_U.row(2).transpose();
    Eigen::Vector2d grad_E = grad_U.row(3).transpose();

    double rho_inv = 1.0 / rho_safe;
    data.grad_u = (grad_rhou - data.u * data.grad_rho) * rho_inv;
    data.grad_v = (grad_rhov - data.v * data.grad_rho) * rho_inv;

    double kinetic_sq = data.u * data.u + data.v * data.v;
    Eigen::Vector2d grad_velocity_norm = 2.0 * data.u * data.grad_u + 2.0 * data.v * data.grad_v;
    Eigen::Vector2d momentum_term = data.rho * (data.u * data.grad_u + data.v * data.grad_v);
    Eigen::Vector2d grad_p =
        (get_gamma() - 1.0) * (grad_E - 0.5 * kinetic_sq * data.grad_rho - momentum_term);

    double rho_sq = data.rho * data.rho;
    data.temperature = data.p / (data.rho * gas_constant_);
    data.grad_T = (data.rho * grad_p - data.p * data.grad_rho) / (rho_sq * gas_constant_);

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

std::pair<Eigen::Vector4d, Eigen::Vector4d> NavierStokesWeakFormulation::compute_viscous_fluxes(
    const Eigen::Vector4d& U, const Eigen::Matrix<double, 4, 2>& grad_U) const {
    PrimitiveGradientData data = compute_primitive_gradients(U, grad_U);

    Eigen::Vector4d Fv = Eigen::Vector4d::Zero();
    Eigen::Vector4d Gv = Eigen::Vector4d::Zero();

    Fv[1] = data.tau_xx;
    Fv[2] = data.tau_xy;
    Fv[3] = data.u * data.tau_xx + data.v * data.tau_xy + data.q_x;

    Gv[1] = data.tau_xy;
    Gv[2] = data.tau_yy;
    Gv[3] = data.u * data.tau_xy + data.v * data.tau_yy + data.q_y;

    return {Fv, Gv};
}

Eigen::MatrixXd NavierStokesWeakFormulation::viscous_volume_residual(
    const Eigen::MatrixXd& u_coeffs_elem, const std::map<std::string, Eigen::MatrixXd>& elem_data,
    std::shared_ptr<DGSpace> dg_space) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd R_visc = Eigen::MatrixXd::Zero(n_basis, get_n_vars());

    const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
    const Eigen::MatrixXd& phi = dg_space->get_volume_basis_values();
    const Eigen::MatrixXd& dphi_dx = elem_data.at("dphi_dx_vol");
    const Eigen::VectorXd& detJ = elem_data.at("J_det_vol");

    int n_quad = weights.size();

    for (int q = 0; q < n_quad; ++q) {
        Eigen::Vector4d U_q = Eigen::Vector4d::Zero();
        for (int i = 0; i < n_basis; ++i) {
            U_q += phi(q, i) * u_coeffs_elem.row(i).transpose();
        }

        Eigen::Matrix<double, 4, 2> grad_U = Eigen::Matrix<double, 4, 2>::Zero();
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_i;
            grad_phi_i << dphi_dx(q * n_basis + i, 0), dphi_dx(q * n_basis + i, 1);
            for (int v = 0; v < get_n_vars(); ++v) {
                grad_U(v, 0) += u_coeffs_elem(i, v) * grad_phi_i[0];
                grad_U(v, 1) += u_coeffs_elem(i, v) * grad_phi_i[1];
            }
        }

        auto [Fv, Gv] = compute_viscous_fluxes(U_q, grad_U);
        double w_q = weights[q] * std::abs(detJ[q]);

        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_i;
            grad_phi_i << dphi_dx(q * n_basis + i, 0), dphi_dx(q * n_basis + i, 1);
            R_visc.row(i) -= w_q * (Fv * grad_phi_i[0] + Gv * grad_phi_i[1]).transpose();
        }
    }

    return R_visc;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
NavierStokesWeakFormulation::viscous_interior_face_residual(
    const Eigen::MatrixXd& u_coeffs_L, const Eigen::MatrixXd& u_coeffs_R,
    const std::map<std::string, Eigen::MatrixXd>& face_data_L,
    const std::map<std::string, Eigen::MatrixXd>& face_data_R, std::shared_ptr<DGSpace> dg_space,
    const Eigen::VectorXi& permutation) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd R_face_L = Eigen::MatrixXd::Zero(n_basis, get_n_vars());
    Eigen::MatrixXd R_face_R = Eigen::MatrixXd::Zero(n_basis, get_n_vars());

    const Eigen::VectorXd weights = face_data_L.at("weights").col(0);
    const Eigen::MatrixXd& phi_L = face_data_L.at("phi");
    const Eigen::MatrixXd& phi_R = face_data_R.at("phi");
    const Eigen::MatrixXd& grad_phi_L = face_data_L.at("dphi_dx_face");
    const Eigen::MatrixXd& grad_phi_R = face_data_R.at("dphi_dx_face");
    Eigen::Vector2d normal = face_data_L.at("normal").col(0);
    double face_length = face_data_L.at("length")(0, 0);

    int n_quad = weights.size();
    int order = dg_space->get_order();
    double sigma = compute_penalty_parameter(order, face_length);

    for (int q = 0; q < n_quad; ++q) {
        int qR = permutation.size() > 0 ? permutation[q] : q;

        Eigen::Vector4d U_L_q = Eigen::Vector4d::Zero();
        Eigen::Vector4d U_R_q = Eigen::Vector4d::Zero();

        for (int i = 0; i < n_basis; ++i) {
            U_L_q += phi_L(q, i) * u_coeffs_L.row(i).transpose();
            U_R_q += phi_R(qR, i) * u_coeffs_R.row(i).transpose();
        }

        Eigen::Matrix<double, 4, 2> grad_UL = Eigen::Matrix<double, 4, 2>::Zero();
        Eigen::Matrix<double, 4, 2> grad_UR = Eigen::Matrix<double, 4, 2>::Zero();

        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_i_L = grad_phi_L.block(q * n_basis + i, 0, 1, 2).transpose();
            Eigen::Vector2d grad_phi_i_R = grad_phi_R.block(qR * n_basis + i, 0, 1, 2).transpose();
            for (int v = 0; v < get_n_vars(); ++v) {
                grad_UL(v, 0) += u_coeffs_L(i, v) * grad_phi_i_L[0];
                grad_UL(v, 1) += u_coeffs_L(i, v) * grad_phi_i_L[1];
                grad_UR(v, 0) += u_coeffs_R(i, v) * grad_phi_i_R[0];
                grad_UR(v, 1) += u_coeffs_R(i, v) * grad_phi_i_R[1];
            }
        }

        auto [Fv_L, Gv_L] = compute_viscous_fluxes(U_L_q, grad_UL);
        auto [Fv_R, Gv_R] = compute_viscous_fluxes(U_R_q, grad_UR);

        Eigen::Vector4d flux_avg = 0.5 * (Fv_L * normal[0] + Gv_L * normal[1]);
        flux_avg += 0.5 * (Fv_R * (-normal[0]) + Gv_R * (-normal[1]));

        Eigen::Vector4d penalty = sigma * (U_R_q - U_L_q);
        Eigen::Vector4d Fn = flux_avg - penalty;

        double w_q = weights[q] * face_length * 0.5;

        for (int i = 0; i < n_basis; ++i) {
            R_face_L.row(i) -= w_q * phi_L(q, i) * Fn.transpose();
            R_face_R.row(i) += w_q * phi_R(qR, i) * Fn.transpose();
        }
    }

    return {R_face_L, R_face_R};
}

Eigen::MatrixXd NavierStokesWeakFormulation::viscous_boundary_face_residual(
    const Eigen::MatrixXd& u_coeffs, const std::map<std::string, Eigen::MatrixXd>& face_data,
    std::shared_ptr<BoundaryConditionEuler> bc, std::shared_ptr<DGSpace> dg_space) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd R_face = Eigen::MatrixXd::Zero(n_basis, get_n_vars());

    const Eigen::VectorXd weights = face_data.at("weights").col(0);
    const Eigen::MatrixXd& phi = face_data.at("phi");
    const Eigen::MatrixXd& grad_phi = face_data.at("dphi_dx_face");
    Eigen::Vector2d normal = face_data.at("normal").col(0);
    double face_length = face_data.at("length")(0, 0);

    int order = dg_space->get_order();
    double sigma = compute_penalty_parameter(order, face_length);
    int n_quad = weights.size();

    for (int q = 0; q < n_quad; ++q) {
        Eigen::Vector4d U_L_q = Eigen::Vector4d::Zero();
        for (int i = 0; i < n_basis; ++i) {
            U_L_q += phi(q, i) * u_coeffs.row(i).transpose();
        }

        Eigen::Matrix<double, 4, 2> grad_UL = Eigen::Matrix<double, 4, 2>::Zero();
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_i = grad_phi.block(q * n_basis + i, 0, 1, 2).transpose();
            for (int v = 0; v < get_n_vars(); ++v) {
                grad_UL(v, 0) += u_coeffs(i, v) * grad_phi_i[0];
                grad_UL(v, 1) += u_coeffs(i, v) * grad_phi_i[1];
            }
        }

        auto [Fv_L, Gv_L] = compute_viscous_fluxes(U_L_q, grad_UL);
        Eigen::Vector4d U_bc = U_L_q;
        if (bc) {
            Eigen::Vector2d x_q;
            const Eigen::MatrixXd& face_quad_points = face_data.at("quad_points");
            int n_quad = face_quad_points.rows() / 2;
            x_q[0] = face_quad_points(q, 0);
            x_q[1] = face_quad_points(q + n_quad, 0);

            Eigen::Vector4d bc_data = bc->evaluate(x_q);
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

        Eigen::Vector4d penalty = sigma * (U_bc - U_L_q);
        Eigen::Vector4d Fn = (Fv_L * normal[0] + Gv_L * normal[1]) - penalty;
        double w_q = weights[q] * face_length * 0.5;

        for (int i = 0; i < n_basis; ++i) {
            R_face.row(i) -= w_q * phi(q, i) * Fn.transpose();
        }
    }

    return R_face;
}

double NavierStokesWeakFormulation::compute_penalty_parameter(int p, double h) const {
    double h_safe = std::max(h, 1e-12);
    return sigma0_ * mu_ * (p + 1) * (p + 1) / h_safe;
}

}  // namespace dgfem
