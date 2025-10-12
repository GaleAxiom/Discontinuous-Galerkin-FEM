/**
 * @file advection_weak_formulation.cpp
 * @brief Implementation of Advection weak formulation
 */

#include "dgfem/weak_forms/advection_weak_formulation.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/mapping.hpp"
#include <iostream>
#include <cmath>

namespace dgfem {

AdvectionWeakFormulation::AdvectionWeakFormulation(const Eigen::Vector2d& velocity)
    : beta_(velocity) {}

Eigen::MatrixXd AdvectionWeakFormulation::compute_volume_integral(
    const std::map<std::string, Eigen::MatrixXd>& elem_data,
    std::shared_ptr<DGSpace> dg_space) const {
    
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd L_vol = Eigen::MatrixXd::Zero(n_basis, n_basis);
    
    const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
    const Eigen::VectorXd& J_det_vals = elem_data.at("J_det_vol");
    const Eigen::MatrixXd& dphi_dx_vals = elem_data.at("dphi_dx_vol");
    const Eigen::MatrixXd& phi_vals = dg_space->get_volume_basis_values();
    
    int n_quad = weights.size();
    
    for (int q = 0; q < n_quad; ++q) {
        double w_q_phys = weights[q] * std::abs(J_det_vals[q]);
        
        // Compute beta · grad(phi) for all basis functions at quadrature point q
        Eigen::VectorXd beta_dot_grad_phi(n_basis);
        for (int j = 0; j < n_basis; ++j) {
            Eigen::Vector2d grad_phi_j = dphi_dx_vals.block(q * n_basis + j, 0, 1, 2).transpose();
            beta_dot_grad_phi[j] = beta_.dot(grad_phi_j);
        }
        
        Eigen::VectorXd phi_q = phi_vals.row(q).transpose();
        L_vol += w_q_phys * (beta_dot_grad_phi * phi_q.transpose());
    }
    
    return L_vol;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
AdvectionWeakFormulation::compute_interior_face_integral(
    int elem_L, int face_L, int elem_R, int face_R,
    std::shared_ptr<DGMesh> mesh,
    const Eigen::VectorXi& permutation) const {
    
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    
    Eigen::MatrixXd L_LL = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd L_LR = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd L_RL = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd L_RR = Eigen::MatrixXd::Zero(n_basis, n_basis);
    
    // Get face data
    const auto& face_L_data = mesh->get_face_data(elem_L, face_L);
    Eigen::Vector2d nL = face_L_data.at("normal").head<2>();
    double hF = face_L_data.at("length")[0];
    
    double beta_n_L = beta_.dot(nL);
    
    // Get face basis values
    const std::vector<Eigen::MatrixXd>& phi_face = dg_space->get_face_basis_values();
    const Eigen::VectorXd& w1d = dg_space->get_face_quad()->weights;
    
    const Eigen::MatrixXd& phi_L = phi_face[face_L];
    Eigen::MatrixXd phi_R = phi_face[face_R];
    
    int n_face_quad = w1d.size();
    
    // Apply permutation if provided
    if (permutation.size() > 0) {
        Eigen::MatrixXd permuted_phi_R(phi_R.rows(), phi_R.cols());
        for (int i = 0; i < n_face_quad; ++i) {
            permuted_phi_R.row(i) = phi_R.row(permutation(i));
        }
        phi_R = permuted_phi_R;
    }
    
    // Upwind flux implementation
    for (int q = 0; q < n_face_quad; ++q) {
        double w_q = w1d[q] * hF * 0.5;
        Eigen::VectorXd phi_L_q = phi_L.row(q).transpose();
        Eigen::VectorXd phi_R_q = phi_R.row(q).transpose();
        
        // Contribution to element L's equation: -∫ v_L u* (β⋅n_L) dS
        if (beta_n_L >= 0.0) {  // Outflow from L, u* = u_L
            L_LL -= (w_q * beta_n_L) * (phi_L_q * phi_L_q.transpose());
        } else {  // Inflow to L, u* = u_R
            L_LR -= (w_q * beta_n_L) * (phi_L_q * phi_R_q.transpose());
        }
        
        // Contribution to element R's equation: -∫ v_R u* (β⋅n_R) dS, where n_R = -n_L
        double beta_n_R = -beta_n_L;
        if (beta_n_L >= 0.0) {  // Inflow to R, u* = u_L
            L_RL -= (w_q * beta_n_R) * (phi_R_q * phi_L_q.transpose());
        } else {  // Outflow from R, u* = u_R
            L_RR -= (w_q * beta_n_R) * (phi_R_q * phi_R_q.transpose());
        }
    }
    
    return std::make_tuple(L_LL, L_LR, L_RL, L_RR);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> AdvectionWeakFormulation::compute_boundary_face_integral(
    int elem_id, int face_id,
    std::shared_ptr<DGMesh> mesh,
    std::function<double(const Eigen::Vector2d&)> bc_func) const {
    
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd L_bc = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::VectorXd F_bc = Eigen::VectorXd::Zero(n_basis);
    
    if (!bc_func) {
        return std::make_tuple(L_bc, F_bc);
    }
    
    // Get face data
    const auto& face_data = mesh->get_face_data(elem_id, face_id);
    Eigen::Vector2d n = face_data.at("normal").head<2>();
    double h_F = face_data.at("length")[0];
    
    const std::vector<Eigen::MatrixXd>& phi_face = dg_space->get_face_basis_values();
    const Eigen::VectorXd& weights = dg_space->get_face_quad()->weights;
    
    const Eigen::MatrixXd& phi = phi_face[face_id];
    int n_face_quad = weights.size();
    
    double beta_n = beta_.dot(n);
    
    // Get element vertices for coordinate mapping
    Eigen::MatrixXd vertices(mesh->get_elements().cols(), 2);
    for (int i = 0; i < mesh->get_elements().cols(); ++i) {
        vertices.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem_id, i));
    }
    auto mapping = dg_space->get_mapping();
    const Eigen::MatrixXd& face_quad_points = dg_space->get_face_quad()->points;
    
    for (int q = 0; q < n_face_quad; ++q) {
        double w_q = weights[q] * h_F * 0.5;
        
        // Get physical coordinates of quadrature point
        Eigen::Vector2d xi_face = dg_space->map_face_quad_point(face_id, face_quad_points(q, 0));
        Eigen::Vector2d x_quad = mapping->map_to_physical(vertices, xi_face);
        
        Eigen::VectorXd phi_q = phi.row(q).transpose();
        
        if (beta_n >= 0.0) {  // Outflow, u* = u_L (no BC needed)
            L_bc -= (w_q * beta_n) * (phi_q * phi_q.transpose());
        } else {  // Inflow, u* = u_g (prescribed BC)
            double g_q = bc_func(x_quad);
            F_bc -= (w_q * beta_n * g_q) * phi_q;
        }
    }
    
    return std::make_tuple(L_bc, F_bc);
}

} // namespace dgfem
