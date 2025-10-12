/**
 * @file euler_weak_formulation.cpp
 * @brief Implementation of Euler weak formulation
 */

#include "dgfem/weak_forms/euler_weak_formulation.hpp"
#include "dgfem/solver/assembler.hpp"
#include "dgfem/core/space.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace dgfem {

EulerWeakFormulation::EulerWeakFormulation(double gamma)
    : gamma_(gamma) {}

std::tuple<Eigen::Vector4d, Eigen::Vector4d> EulerWeakFormulation::get_fluxes(
    const Eigen::Vector4d& U) const {
    
    Eigen::Vector4d W = conserved_to_primitive(U, gamma_);
    double rho = W[0];
    double u = W[1];
    double v = W[2];
    double p = W[3];
    
    double rho_u = U[1];
    double rho_v = U[2];
    double E = U[3];
    
    Eigen::Vector4d F, G;
    F << rho_u,
         rho * u * u + p,
         rho * u * v,
         u * (E + p);
    
    G << rho_v,
         rho * u * v,
         rho * v * v + p,
         v * (E + p);
    
    return std::make_tuple(F, G);
}

Eigen::Vector4d EulerWeakFormulation::rusanov_flux(
    const Eigen::Vector4d& U_L,
    const Eigen::Vector4d& U_R,
    const Eigen::Vector2d& normal) const {
    
    auto [F_L, G_L] = get_fluxes(U_L);
    auto [F_R, G_R] = get_fluxes(U_R);
    
    Eigen::Vector4d Fn_L = F_L * normal[0] + G_L * normal[1];
    Eigen::Vector4d Fn_R = F_R * normal[0] + G_R * normal[1];
    
    Eigen::Vector4d W_L = conserved_to_primitive(U_L, gamma_);
    Eigen::Vector4d W_R = conserved_to_primitive(U_R, gamma_);
    
    double c_L = std::sqrt(gamma_ * W_L[3] / W_L[0]);  // speed of sound
    double c_R = std::sqrt(gamma_ * W_R[3] / W_R[0]);
    
    double un_L = W_L[1] * normal[0] + W_L[2] * normal[1];  // normal velocity
    double un_R = W_R[1] * normal[0] + W_R[2] * normal[1];
    
    double s_max = std::max(std::abs(un_L) + c_L, std::abs(un_R) + c_R);
    
    return 0.5 * (Fn_L + Fn_R) - 0.5 * s_max * (U_R - U_L);
}

Eigen::MatrixXd EulerWeakFormulation::volume_residual(
    const Eigen::MatrixXd& u_coeffs_elem,
    const std::map<std::string, Eigen::MatrixXd>& elem_data,
    std::shared_ptr<DGSpace> dg_space) const {
    
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = u_coeffs_elem.cols();
    Eigen::MatrixXd R_vol = Eigen::MatrixXd::Zero(n_basis, n_vars);
    
    const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
    const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");
    const Eigen::MatrixXd& phi = dg_space->get_volume_basis_values();
    const Eigen::MatrixXd& dphi_dx = elem_data.at("dphi_dx_vol");
    
    int n_quad = weights.size();
    
    for (int q = 0; q < n_quad; ++q) {
        // Compute solution at quadrature point
        Eigen::Vector4d U_q = Eigen::Vector4d::Zero();
        for (int i = 0; i < n_basis; ++i) {
            U_q += phi(q, i) * u_coeffs_elem.row(i).transpose();
        }
        
        // Get fluxes
        auto [F_q, G_q] = get_fluxes(U_q);
        
        double w_q_phys = weights[q] * std::abs(J_det[q]);
        
        // R_vol[i,v] += w * (F_q[v]*dphi_dx[q,i,0] + G_q[v]*dphi_dx[q,i,1])
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_i;
            grad_phi_i << dphi_dx(q * n_basis + i, 0), dphi_dx(q * n_basis + i, 1);
            
            R_vol.row(i) += w_q_phys * (F_q * grad_phi_i[0] + G_q * grad_phi_i[1]).transpose();
        }
    }
    
    return R_vol;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> EulerWeakFormulation::interior_face_residual(
    const Eigen::MatrixXd& u_coeffs_L,
    const Eigen::MatrixXd& u_coeffs_R,
    const std::map<std::string, Eigen::MatrixXd>& face_data_L,
    const std::map<std::string, Eigen::MatrixXd>& face_data_R,
    std::shared_ptr<DGSpace> dg_space,
    const Eigen::VectorXi& permutation) const {
    
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = u_coeffs_L.cols();
    Eigen::MatrixXd R_face_L = Eigen::MatrixXd::Zero(n_basis, n_vars);
    Eigen::MatrixXd R_face_R = Eigen::MatrixXd::Zero(n_basis, n_vars);
    
    const Eigen::VectorXd weights = face_data_L.at("weights").col(0);
    const Eigen::MatrixXd& phi_L = face_data_L.at("phi");
    const Eigen::MatrixXd& phi_R = face_data_R.at("phi");
    const Eigen::Vector2d normal = face_data_L.at("normal").col(0);
    double hF = face_data_L.at("length")(0, 0);
    
    int n_quad = weights.size();
    
    for (int q = 0; q < n_quad; ++q) {
        // Compute solution at quadrature point from both sides
        Eigen::Vector4d U_L_q = Eigen::Vector4d::Zero();
        Eigen::Vector4d U_R_q = Eigen::Vector4d::Zero();
        
        for (int i = 0; i < n_basis; ++i) {
            U_L_q += phi_L(q, i) * u_coeffs_L.row(i).transpose();
            
            int idx_R = (permutation.size() > 0) ? permutation[q] : q;
            U_R_q += phi_R(idx_R, i) * u_coeffs_R.row(i).transpose();
        }
        
        // Compute numerical flux
        Eigen::Vector4d H_q = rusanov_flux(U_L_q, U_R_q, normal);
        
        double w_q_phys = weights[q] * hF * 0.5;
        
        // Update residuals
        for (int i = 0; i < n_basis; ++i) {
            R_face_L.row(i) -= w_q_phys * phi_L(q, i) * H_q.transpose();
            
            int idx_R = (permutation.size() > 0) ? permutation[q] : q;
            R_face_R.row(i) += w_q_phys * phi_R(idx_R, i) * H_q.transpose();
        }
    }
    
    return std::make_tuple(R_face_L, R_face_R);
}

Eigen::MatrixXd EulerWeakFormulation::boundary_face_residual(
    const Eigen::MatrixXd& u_coeffs,
    const std::map<std::string, Eigen::MatrixXd>& face_data,
    std::shared_ptr<BoundaryConditionEuler> bc,
    std::shared_ptr<DGSpace> dg_space) const {
    
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = u_coeffs.cols();
    Eigen::MatrixXd R_face_bc = Eigen::MatrixXd::Zero(n_basis, n_vars);
    
    const Eigen::VectorXd weights = face_data.at("weights").col(0);
    const Eigen::MatrixXd& phi = face_data.at("phi");
    const Eigen::Vector2d normal = face_data.at("normal").col(0);
    double hF = face_data.at("length")(0, 0);
    const Eigen::MatrixXd quad_points_mat = face_data.at("quad_points");
    
    // Reshape quad_points from column vector to matrix
    int n_quad = weights.size();
    Eigen::MatrixXd quad_points(n_quad, 2);
    for (int q = 0; q < n_quad; ++q) {
        quad_points(q, 0) = quad_points_mat(q * 2, 0);
        quad_points(q, 1) = quad_points_mat(q * 2 + 1, 0);
    }
    
    for (int q = 0; q < n_quad; ++q) {
        // Compute interior solution at quadrature point
        Eigen::Vector4d U_L_q = Eigen::Vector4d::Zero();
        for (int i = 0; i < n_basis; ++i) {
            U_L_q += phi(q, i) * u_coeffs.row(i).transpose();
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
        } else if (bc->get_type() == BCTypeEuler::PERIODIC) {
            throw std::runtime_error("PERIODIC BC encountered in boundary_face_residual - periodic boundaries should be treated as interior faces");
        } else {
            throw std::runtime_error("Unsupported BC type for Euler equations");
        }
        
        // Compute numerical flux
        Eigen::Vector4d H_q = rusanov_flux(U_L_q, U_R_q, normal);
        
        double w_q_phys = weights[q] * hF * 0.5;
        
        // Update residual
        for (int i = 0; i < n_basis; ++i) {
            R_face_bc.row(i) -= w_q_phys * phi(q, i) * H_q.transpose();
        }
    }
    
    return R_face_bc;
}

void EulerWeakFormulation::assemble(
    DGAssembler& assembler,
    std::function<double(const Eigen::Vector2d&)> source_func,
    std::function<double(const Eigen::Vector2d&)> bc_func) const {
    
    // Euler equations use residual-based assembly, not matrix assembly
    // This method should not be called directly
    throw std::runtime_error("Euler equations use assemble_euler_residual() instead of assemble()");
}

// Utility functions for Euler equations
Eigen::Vector4d primitive_to_conserved(const Eigen::Vector4d& primitive, double gamma) {
    double rho = primitive[0];
    double u = primitive[1];
    double v = primitive[2];
    double p = primitive[3];
    
    double E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    
    Eigen::Vector4d conserved;
    conserved[0] = rho;
    conserved[1] = rho * u;
    conserved[2] = rho * v;
    conserved[3] = E;
    
    return conserved;
}

Eigen::Vector4d conserved_to_primitive(const Eigen::Vector4d& conserved, double gamma) {
    double rho = conserved[0];
    double rho_u = conserved[1];
    double rho_v = conserved[2];
    double E = conserved[3];
    
    double u = rho_u / rho;
    double v = rho_v / rho;
    double p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v));
    
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
    const Eigen::MatrixXd& u_coeffs_L,
    const Eigen::MatrixXd& u_coeffs_R,
    const std::map<std::string, Eigen::MatrixXd>& /*face_data_L*/,
    const std::map<std::string, Eigen::MatrixXd>& /*face_data_R*/,
    std::shared_ptr<DGSpace> dg_space,
    const Eigen::VectorXi& /*permutation*/) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_vars = u_coeffs_L.cols();
    return {
        Eigen::MatrixXd::Zero(n_basis, n_vars),
        Eigen::MatrixXd::Zero(n_basis, n_vars)
    };
}

Eigen::MatrixXd EulerWeakFormulation::viscous_boundary_face_residual(
    const Eigen::MatrixXd& /*u_coeffs*/,
    const std::map<std::string, Eigen::MatrixXd>& /*face_data*/,
    std::shared_ptr<BoundaryConditionEuler> /*bc*/,
    std::shared_ptr<DGSpace> dg_space) const {
    return Eigen::MatrixXd::Zero(dg_space->get_basis()->get_n_basis(), get_n_vars());
}

} // namespace dgfem
