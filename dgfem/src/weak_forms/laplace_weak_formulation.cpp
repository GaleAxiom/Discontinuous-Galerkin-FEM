/**
 * @file laplace_weak_formulation.cpp
 * @brief Implementation of Laplace weak formulation
 */

#include "dgfem/weak_forms/laplace_weak_formulation.hpp"

#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/mapping.hpp"
#include "dgfem/solver/assembler.hpp"

#include <cmath>

#include <iostream>
#include <set>

namespace dgfem {

LaplaceWeakFormulation::LaplaceWeakFormulation(double penalty_parameter)
    : sigma_0_(penalty_parameter) {}

double LaplaceWeakFormulation::compute_penalty_parameter(int p, double h) const {
    if (h < 1e-12)
        return sigma_0_ * (p + 1) * (p + 1);
    return sigma_0_ * (p + 1) * (p + 1) / h;
}

DView2
LaplaceWeakFormulation::compute_volume_integral(const std::map<std::string, DView2>& elem_data,
                                                std::shared_ptr<DGSpace> dg_space) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView2 K_vol("K_vol", n_basis, n_basis);

    const DView2& J_det_vals = elem_data.at("J_det_vol");
    const DView2& dphi_dx_vals = elem_data.at("dphi_dx_vol");
    const DView1& weights = dg_space->get_volume_quad()->weights;

    int n_quad = weights.extent(0);

    for (int q = 0; q < n_quad; ++q) {
        double w_q_phys = weights[q] * std::abs(J_det_vals(q, 0));

        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_i = row2(dphi_dx_vals, q * n_basis + i);
            for (int j = 0; j < n_basis; ++j) {
                Vec2 grad_phi_j = row2(dphi_dx_vals, q * n_basis + j);
                K_vol(i, j) += w_q_phys * dot(grad_phi_i, grad_phi_j);
            }
        }
    }

    return K_vol;
}

std::tuple<DView2, DView2, DView2, DView2>
LaplaceWeakFormulation::compute_interior_face_integral(int elem_L, int face_L, int elem_R,
                                                       int face_R, std::shared_ptr<DGMesh> mesh,
                                                       const IView1& permutation) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();

    DView2 K_LL("K_LL", n_basis, n_basis);
    DView2 K_LR("K_LR", n_basis, n_basis);
    DView2 K_RL("K_RL", n_basis, n_basis);
    DView2 K_RR("K_RR", n_basis, n_basis);

    // Get face data
    const auto& face_L_data = mesh->get_face_data(elem_L, face_L);

    Vec2 normal = to_vec2(face_L_data.at("normal"));
    double face_length = scalar_of(face_L_data.at("length"));

    // Get face basis values
    const std::vector<DView2>& phi_face = dg_space->get_face_basis_values();
    const DView1& face_weights = dg_space->get_face_quad()->weights;

    int n_face_quad = face_weights.extent(0);

    // Penalty parameter (simplified)
    double h_face = face_length;
    int p = dg_space->get_order();
    double sigma = compute_penalty_parameter(p, h_face);

    auto elem_data_L = mesh->get_element_data(elem_L);
    auto elem_data_R = mesh->get_element_data(elem_R);

    DView2 grad_phi_L = interpolate_gradient_to_face(elem_data_L, face_L, dg_space);
    DView2 grad_phi_R = interpolate_gradient_to_face(elem_data_R, face_R, dg_space);

    const DView2& phi_L = phi_face[face_L];
    const DView2& phi_R_raw = phi_face[face_R];

    // Apply the face permutation (matching quadrature-point ordering between the two sides).
    DView2 phi_R("phi_R_permuted", n_face_quad, n_basis);
    DView2 grad_phi_R_permuted("grad_phi_R_permuted", n_face_quad * n_basis, 2);
    for (int i = 0; i < n_face_quad; ++i) {
        int src = permutation(i);
        for (int b = 0; b < n_basis; ++b) {
            phi_R(i, b) = phi_R_raw(src, b);
            set_row2(grad_phi_R_permuted, i * n_basis + b, row2(grad_phi_R, src * n_basis + b));
        }
    }
    grad_phi_R = grad_phi_R_permuted;

    // SIPG formulation
    for (int q = 0; q < n_face_quad; ++q) {
        double w_q = face_weights[q] * face_length * 0.5;

        DView1 vL = row_of(phi_L, q);
        DView1 vR = row_of(phi_R, q);
        const DView1& uL = vL;
        const DView1& uR = vR;

        DView1 C_vec("C_vec", n_basis);
        DView1 D_vec("D_vec", n_basis);
        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_L_i = row2(grad_phi_L, q * n_basis + i);
            Vec2 grad_phi_R_i = row2(grad_phi_R, q * n_basis + i);
            C_vec(i) = dot(grad_phi_L_i, normal);
            D_vec(i) = dot(grad_phi_R_i, normal);
        }
        const DView1& A_vec = C_vec;
        const DView1& B_vec = D_vec;

        outer_add(K_LL, -0.5 * w_q, vL, A_vec);
        outer_add(K_LL, -0.5 * w_q, C_vec, uL);
        outer_add(K_LL, sigma * w_q, vL, uL);

        outer_add(K_LR, -0.5 * w_q, vL, B_vec);
        outer_add(K_LR, 0.5 * w_q, C_vec, uR);
        outer_add(K_LR, -sigma * w_q, vL, uR);

        outer_add(K_RL, 0.5 * w_q, vR, A_vec);
        outer_add(K_RL, -0.5 * w_q, D_vec, uL);
        outer_add(K_RL, -sigma * w_q, vR, uL);

        outer_add(K_RR, 0.5 * w_q, vR, B_vec);
        outer_add(K_RR, 0.5 * w_q, D_vec, uR);
        outer_add(K_RR, sigma * w_q, vR, uR);
    }

    return std::make_tuple(K_LL, K_LR, K_RL, K_RR);
}

DView2 LaplaceWeakFormulation::compute_boundary_face_integral(
    int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
    std::shared_ptr<BoundaryCondition> bc) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView2 K_boundary("K_boundary", n_basis, n_basis);

    // Get face data
    const auto& face_data = mesh->get_face_data(elem_id, face_id);

    Vec2 normal = to_vec2(face_data.at("normal"));
    double face_length = scalar_of(face_data.at("length"));

    // Get face basis values
    const std::vector<DView2>& phi_face = dg_space->get_face_basis_values();
    const DView1& face_weights = dg_space->get_face_quad()->weights;

    int n_face_quad = face_weights.extent(0);

    // Penalty parameter
    int p = dg_space->get_order();
    double sigma = compute_penalty_parameter(p, face_length);

    auto elem_data = mesh->get_element_data(elem_id);
    DView2 grad_phi = interpolate_gradient_to_face(elem_data, face_id, dg_space);

    const DView2& phi = phi_face[face_id];

    // SIPG formulation for boundary face
    if (bc && bc->get_type() == BCType::DIRICHLET) {
        for (int q = 0; q < n_face_quad; ++q) {
            double w_q = face_weights[q] * face_length * 0.5;

            DView1 v = row_of(phi, q);
            const DView1& u = v;

            DView1 C_vec("C_vec", n_basis);
            for (int i = 0; i < n_basis; ++i) {
                Vec2 grad_phi_i = row2(grad_phi, q * n_basis + i);
                C_vec(i) = dot(grad_phi_i, normal);
            }

            outer_add(K_boundary, -w_q, v, C_vec);
            outer_add(K_boundary, -w_q, C_vec, u);
            outer_add(K_boundary, sigma * w_q, v, u);
        }
    } else {
        std::cout << "No valid boundary condition provided for element " << elem_id << " face "
                  << face_id << ". Skipping boundary integral." << std::endl;
    }

    return K_boundary;
}

DView1
LaplaceWeakFormulation::compute_source_integral(int elem_id,
                                                std::function<double(const Vec2&)> source_func,
                                                std::shared_ptr<DGMesh> mesh) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView1 F_source("F_source", n_basis);

    if (!source_func)
        return F_source;

    // Get element data
    const auto& elem_data = mesh->get_element_data(elem_id);
    const DView2& J_det_vals = elem_data.at("J_det_vol");
    const DView1& weights = dg_space->get_volume_quad()->weights;
    const DView2& vol_basis = dg_space->get_volume_basis_values();

    DView2 vertices = mesh->get_element_vertices(elem_id);

    auto mapping = dg_space->get_mapping();
    int n_quad = weights.extent(0);

    for (int q = 0; q < n_quad; ++q) {
        Vec2 xi = row2(dg_space->get_volume_quad()->points, q);
        Vec2 x_phys = mapping->map_to_physical(vertices, xi);

        double source_val = source_func(x_phys);
        double w_q_phys = weights[q] * J_det_vals(q, 0);

        for (int i = 0; i < n_basis; ++i) {
            F_source[i] += w_q_phys * source_val * vol_basis(q, i);
        }
    }

    return F_source;
}

DView2
LaplaceWeakFormulation::interpolate_gradient_to_face(const std::map<std::string, DView2>& elem_data,
                                                     int face_id,
                                                     std::shared_ptr<DGSpace> dg_space) const {
    int n_face_quad = dg_space->get_face_quad()->weights.extent(0);
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView2 dphi_dx_face("dphi_dx_face", n_face_quad * n_basis, 2);

    const DView2& elem_vertices = elem_data.at("vertices");
    const DView2& face_quad_points = dg_space->get_face_quad()->points;

    auto mapping = dg_space->get_mapping();

    for (int q = 0; q < n_face_quad; ++q) {
        Vec2 xi_face = dg_space->map_face_quad_point(face_id, face_quad_points(q, 0));

        DView2 dphi_dxi = dg_space->get_basis()->evaluate_gradient(xi_face);
        auto mapping_data = mapping->compute_mapping(elem_vertices, xi_face);

        for (int i = 0; i < n_basis; ++i) {
            Vec2 dphi_dx = mapping_data.dxi_dx.apply(row2(dphi_dxi, i));
            set_row2(dphi_dx_face, q * n_basis + i, dphi_dx);
        }
    }

    return dphi_dx_face;
}

DView1
LaplaceWeakFormulation::compute_boundary_rhs_integral(int elem_id, int face_id,
                                                      std::shared_ptr<DGMesh> mesh,
                                                      std::shared_ptr<BoundaryCondition> bc) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView1 F_boundary("F_boundary", n_basis);

    if (!bc)
        return F_boundary;

    if (bc->get_type() == BCType::DIRICHLET) {
        const auto& face_data = mesh->get_face_data(elem_id, face_id);
        double face_length = scalar_of(face_data.at("length"));
        Vec2 normal = to_vec2(face_data.at("normal"));

        const std::vector<DView2>& phi_face = dg_space->get_face_basis_values();
        const DView1& face_weights = dg_space->get_face_quad()->weights;

        int n_face_quad = face_weights.extent(0);
        int p = dg_space->get_order();
        double sigma = compute_penalty_parameter(p, face_length);

        auto elem_data = mesh->get_element_data(elem_id);
        DView2 grad_phi = interpolate_gradient_to_face(elem_data, face_id, dg_space);
        const DView2& phi = phi_face[face_id];

        DView2 vertices = mesh->get_element_vertices(elem_id);
        auto mapping = dg_space->get_mapping();
        const DView2& face_quad_points = dg_space->get_face_quad()->points;

        for (int q = 0; q < n_face_quad; ++q) {
            Vec2 xi_face = dg_space->map_face_quad_point(face_id, face_quad_points(q, 0));
            Vec2 x_quad = mapping->map_to_physical(vertices, xi_face);

            double bc_value = bc->evaluate(x_quad);
            double w_q = face_weights[q] * face_length * 0.5;

            for (int i = 0; i < n_basis; ++i) {
                double phi_i = phi(q, i);
                Vec2 grad_phi_i = row2(grad_phi, q * n_basis + i);
                F_boundary[i] +=
                    w_q * (sigma * bc_value * phi_i - bc_value * dot(grad_phi_i, normal));
            }
        }
    }

    return F_boundary;
}

void LaplaceWeakFormulation::assemble(DGAssembler& assembler,
                                      std::function<double(const Vec2&)> source_func,
                                      std::function<double(const Vec2&)> bc_func) const {
    auto mesh = assembler.get_mesh();
    auto dg_space = assembler.get_dg_space();

    assembler.clear_assembly_data();

    std::cout << "\nAssembling Laplace system..." << std::endl;

    // Volume integrals
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        const auto& elem_data = mesh->get_element_data(elem_id);
        DView2 K_vol = compute_volume_integral(elem_data, dg_space);
        assembler.add_to_matrix(elem_id, elem_id, K_vol);

        // Source term
        if (source_func) {
            DView1 F_src = compute_source_integral(elem_id, source_func, mesh);
            assembler.add_to_rhs(elem_id, F_src);
        }
    }

    std::cout << "Volume integrals assembled" << std::endl;

    // Face integrals
    std::set<std::pair<int, int>> processed_faces;

    for (int elem_L = 0; elem_L < mesh->get_n_elements(); ++elem_L) {
        auto neighbors = mesh->get_element_neighbors(elem_L);

        for (int face_L = 0; face_L < static_cast<int>(neighbors.size()); ++face_L) {
            int elem_R = neighbors[face_L].first;
            int face_R = neighbors[face_L].second;

            if (elem_R >= 0) {
                // Interior face
                std::pair<int, int> face_sig = (elem_L < elem_R) ? std::make_pair(elem_L, elem_R)
                                                                 : std::make_pair(elem_R, elem_L);
                if (processed_faces.find(face_sig) == processed_faces.end()) {
                    processed_faces.insert(face_sig);

                    DView2 v_L = mesh->get_element_vertices(elem_L);
                    DView2 v_R = mesh->get_element_vertices(elem_R);

                    auto perm = dg_space->compute_face_permutation(face_L, v_L, face_R, v_R);

                    auto [K_LL, K_LR, K_RL, K_RR] =
                        compute_interior_face_integral(elem_L, face_L, elem_R, face_R, mesh, perm);

                    assembler.add_to_matrix(elem_L, elem_L, K_LL);
                    assembler.add_to_matrix(elem_L, elem_R, K_LR);
                    assembler.add_to_matrix(elem_R, elem_L, K_RL);
                    assembler.add_to_matrix(elem_R, elem_R, K_RR);
                }
            } else {
                // Boundary face (elem_R == -1)
                auto bc = mesh->get_boundary_condition(elem_L, face_L);
                if (!bc) {
                    std::cerr << "Warning: No boundary condition set for element " << elem_L
                              << " face " << face_L << std::endl;
                    continue;
                }

                DView2 K_bnd = compute_boundary_face_integral(elem_L, face_L, mesh, bc);
                assembler.add_to_matrix(elem_L, elem_L, K_bnd);

                DView1 F_bnd = compute_boundary_rhs_integral(elem_L, face_L, mesh, bc);
                assembler.add_to_rhs(elem_L, F_bnd);
            }
        }
    }

    std::cout << "Face integrals assembled" << std::endl;

    assembler.finalize_assembly();
}

}  // namespace dgfem
