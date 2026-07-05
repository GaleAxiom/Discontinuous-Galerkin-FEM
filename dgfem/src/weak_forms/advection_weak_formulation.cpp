/**
 * @file advection_weak_formulation.cpp
 * @brief Implementation of Advection weak formulation
 */

#include "dgfem/weak_forms/advection_weak_formulation.hpp"

#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/mapping.hpp"

#include <cmath>

#include <iostream>

namespace dgfem {

AdvectionWeakFormulation::AdvectionWeakFormulation(const Vec2& velocity) : beta_(velocity) {}

DView2
AdvectionWeakFormulation::compute_volume_integral(const std::map<std::string, DView2>& elem_data,
                                                  std::shared_ptr<DGSpace> dg_space) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView2 L_vol("L_vol", n_basis, n_basis);

    const DView1& weights = dg_space->get_volume_quad()->weights;
    const DView2& J_det_vals = elem_data.at("J_det_vol");
    const DView2& dphi_dx_vals = elem_data.at("dphi_dx_vol");
    const DView2& phi_vals = dg_space->get_volume_basis_values();

    int n_quad = weights.extent(0);

    for (int q = 0; q < n_quad; ++q) {
        double w_q_phys = weights[q] * std::abs(J_det_vals(q, 0));

        // Compute beta . grad(phi) for all basis functions at quadrature point q
        DView1 beta_dot_grad_phi("beta_dot_grad_phi", n_basis);
        for (int j = 0; j < n_basis; ++j) {
            Vec2 grad_phi_j = row2(dphi_dx_vals, q * n_basis + j);
            beta_dot_grad_phi[j] = dot(beta_, grad_phi_j);
        }

        DView1 phi_q = row_of(phi_vals, q);
        outer_add(L_vol, w_q_phys, beta_dot_grad_phi, phi_q);
    }

    return L_vol;
}

std::tuple<DView2, DView2, DView2, DView2>
AdvectionWeakFormulation::compute_interior_face_integral(int elem_L, int face_L, int elem_R,
                                                         int face_R, std::shared_ptr<DGMesh> mesh,
                                                         const IView1& permutation) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();

    DView2 L_LL("L_LL", n_basis, n_basis);
    DView2 L_LR("L_LR", n_basis, n_basis);
    DView2 L_RL("L_RL", n_basis, n_basis);
    DView2 L_RR("L_RR", n_basis, n_basis);

    // Get face data
    const auto& face_L_data = mesh->get_face_data(elem_L, face_L);
    Vec2 nL = to_vec2(face_L_data.at("normal"));
    double hF = scalar_of(face_L_data.at("length"));

    double beta_n_L = dot(beta_, nL);

    // Get face basis values
    const std::vector<DView2>& phi_face = dg_space->get_face_basis_values();
    const DView1& w1d = dg_space->get_face_quad()->weights;

    const DView2& phi_L = phi_face[face_L];
    const DView2& phi_R_raw = phi_face[face_R];

    int n_face_quad = w1d.extent(0);

    // Apply permutation if provided
    DView2 phi_R = phi_R_raw;
    if (permutation.extent(0) > 0) {
        DView2 permuted_phi_R("permuted_phi_R", n_face_quad, n_basis);
        for (int i = 0; i < n_face_quad; ++i) {
            for (int b = 0; b < n_basis; ++b) {
                permuted_phi_R(i, b) = phi_R_raw(permutation(i), b);
            }
        }
        phi_R = permuted_phi_R;
    }

    // Upwind flux implementation
    for (int q = 0; q < n_face_quad; ++q) {
        double w_q = w1d[q] * hF * 0.5;
        DView1 phi_L_q = row_of(phi_L, q);
        DView1 phi_R_q = row_of(phi_R, q);

        // Contribution to element L's equation: -\int v_L u* (beta.n_L) dS
        if (beta_n_L >= 0.0) {  // Outflow from L, u* = u_L
            outer_add(L_LL, -(w_q * beta_n_L), phi_L_q, phi_L_q);
        } else {  // Inflow to L, u* = u_R
            outer_add(L_LR, -(w_q * beta_n_L), phi_L_q, phi_R_q);
        }

        // Contribution to element R's equation: -\int v_R u* (beta.n_R) dS, where n_R = -n_L
        double beta_n_R = -beta_n_L;
        if (beta_n_L >= 0.0) {  // Inflow to R, u* = u_L
            outer_add(L_RL, -(w_q * beta_n_R), phi_R_q, phi_L_q);
        } else {  // Outflow from R, u* = u_R
            outer_add(L_RR, -(w_q * beta_n_R), phi_R_q, phi_R_q);
        }
    }

    return std::make_tuple(L_LL, L_LR, L_RL, L_RR);
}

std::tuple<DView2, DView1> AdvectionWeakFormulation::compute_boundary_face_integral(
    int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
    std::function<double(const Vec2&)> bc_func) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView2 L_bc("L_bc", n_basis, n_basis);
    DView1 F_bc("F_bc", n_basis);

    if (!bc_func) {
        return std::make_tuple(L_bc, F_bc);
    }

    // Get face data
    const auto& face_data = mesh->get_face_data(elem_id, face_id);
    Vec2 n = to_vec2(face_data.at("normal"));
    double h_F = scalar_of(face_data.at("length"));

    const std::vector<DView2>& phi_face = dg_space->get_face_basis_values();
    const DView1& weights = dg_space->get_face_quad()->weights;

    const DView2& phi = phi_face[face_id];
    int n_face_quad = weights.extent(0);

    double beta_n = dot(beta_, n);

    DView2 vertices = mesh->get_element_vertices(elem_id);
    auto mapping = dg_space->get_mapping();
    const DView2& face_quad_points = dg_space->get_face_quad()->points;

    for (int q = 0; q < n_face_quad; ++q) {
        double w_q = weights[q] * h_F * 0.5;

        // Get physical coordinates of quadrature point
        Vec2 xi_face = dg_space->map_face_quad_point(face_id, face_quad_points(q, 0));
        Vec2 x_quad = mapping->map_to_physical(vertices, xi_face);

        DView1 phi_q = row_of(phi, q);

        if (beta_n >= 0.0) {  // Outflow, u* = u_L (no BC needed)
            outer_add(L_bc, -(w_q * beta_n), phi_q, phi_q);
        } else {  // Inflow, u* = u_g (prescribed BC)
            double g_q = bc_func(x_quad);
            double coeff = -(w_q * beta_n * g_q);
            for (int i = 0; i < n_basis; ++i) {
                F_bc[i] += coeff * phi_q[i];
            }
        }
    }

    return std::make_tuple(L_bc, F_bc);
}

}  // namespace dgfem
