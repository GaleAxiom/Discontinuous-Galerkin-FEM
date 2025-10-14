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

Eigen::MatrixXd LaplaceWeakFormulation::compute_volume_integral(
    const std::map<std::string, Eigen::MatrixXd>& elem_data,
    std::shared_ptr<DGSpace> dg_space) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd K_vol = Eigen::MatrixXd::Zero(n_basis, n_basis);

    const Eigen::VectorXd& J_det_vals = elem_data.at("J_det_vol");
    const Eigen::MatrixXd& dphi_dx_vals = elem_data.at("dphi_dx_vol");
    const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;

    int n_quad = weights.size();

    for (int q = 0; q < n_quad; ++q) {
        double w_q_phys = weights[q] * std::abs(J_det_vals[q]);

        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                Eigen::Vector2d grad_phi_i =
                    dphi_dx_vals.block(q * n_basis + i, 0, 1, 2).transpose();
                Eigen::Vector2d grad_phi_j =
                    dphi_dx_vals.block(q * n_basis + j, 0, 1, 2).transpose();
                K_vol(i, j) += w_q_phys * grad_phi_i.dot(grad_phi_j);
            }
        }
    }

    return K_vol;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
LaplaceWeakFormulation::compute_interior_face_integral(int elem_L, int face_L, int elem_R,
                                                       int face_R, std::shared_ptr<DGMesh> mesh,
                                                       const Eigen::VectorXi& permutation) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();

    Eigen::MatrixXd K_LL = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd K_LR = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd K_RL = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd K_RR = Eigen::MatrixXd::Zero(n_basis, n_basis);

    // Get face data
    const auto& face_L_data = mesh->get_face_data(elem_L, face_L);

    // Extract face normal and length
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double face_length = 0.0;

    // Since face_data returns map<string, VectorXd>, we need to extract properly
    if (face_L_data.find("normal") != face_L_data.end()) {
        const Eigen::VectorXd& normal_vec = face_L_data.at("normal");
        normal = normal_vec.head<2>();
    }
    if (face_L_data.find("length") != face_L_data.end()) {
        face_length = face_L_data.at("length")[0];
    }

    // Get face basis values
    const std::vector<Eigen::MatrixXd>& phi_face = dg_space->get_face_basis_values();
    const Eigen::VectorXd& face_weights = dg_space->get_face_quad()->weights;

    int n_face_quad = face_weights.size();

    // Penalty parameter (simplified)
    double h_face = face_length;
    int p = dg_space->get_order();
    double sigma = compute_penalty_parameter(p, h_face);

    auto elem_data_L = mesh->get_element_data(elem_L);
    auto elem_data_R = mesh->get_element_data(elem_R);

    Eigen::MatrixXd grad_phi_L = interpolate_gradient_to_face(elem_data_L, face_L, dg_space);
    Eigen::MatrixXd grad_phi_R = interpolate_gradient_to_face(elem_data_R, face_R, dg_space);

    const Eigen::MatrixXd& phi_L = phi_face[face_L];
    Eigen::MatrixXd phi_R = phi_face[face_R];

    Eigen::MatrixXd permuted_phi_R(phi_R.rows(), phi_R.cols());
    Eigen::MatrixXd permuted_grad_phi_R(grad_phi_R.rows(), grad_phi_R.cols());
    for (int i = 0; i < n_face_quad; ++i) {
        permuted_phi_R.row(i) = phi_R.row(permutation(i));
        permuted_grad_phi_R.block(i * n_basis, 0, n_basis, 2) =
            grad_phi_R.block(permutation(i) * n_basis, 0, n_basis, 2);
    }
    phi_R = permuted_phi_R;
    grad_phi_R = permuted_grad_phi_R;

    // SIPG formulation
    for (int q = 0; q < n_face_quad; ++q) {
        double w_q = face_weights[q] * face_length * 0.5;

        Eigen::VectorXd vL = phi_L.row(q).transpose();
        Eigen::VectorXd vR = phi_R.row(q).transpose();
        Eigen::VectorXd uL = vL;
        Eigen::VectorXd uR = vR;

        Eigen::VectorXd C_vec(n_basis);
        Eigen::VectorXd D_vec(n_basis);
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_L_i = grad_phi_L.block(q * n_basis + i, 0, 1, 2).transpose();
            Eigen::Vector2d grad_phi_R_i = grad_phi_R.block(q * n_basis + i, 0, 1, 2).transpose();
            C_vec(i) = grad_phi_L_i.dot(normal);
            D_vec(i) = grad_phi_R_i.dot(normal);
        }
        Eigen::VectorXd A_vec = C_vec;
        Eigen::VectorXd B_vec = D_vec;

        K_LL += w_q * (-0.5 * (vL * A_vec.transpose()) - 0.5 * (C_vec * uL.transpose()) +
                       sigma * (vL * uL.transpose()));
        K_LR += w_q * (-0.5 * (vL * B_vec.transpose()) + 0.5 * (C_vec * uR.transpose()) -
                       sigma * (vL * uR.transpose()));
        K_RL += w_q * (+0.5 * (vR * A_vec.transpose()) - 0.5 * (D_vec * uL.transpose()) -
                       sigma * (vR * uL.transpose()));
        K_RR += w_q * (+0.5 * (vR * B_vec.transpose()) + 0.5 * (D_vec * uR.transpose()) +
                       sigma * (vR * uR.transpose()));
    }

    return std::make_tuple(K_LL, K_LR, K_RL, K_RR);
}

Eigen::MatrixXd LaplaceWeakFormulation::compute_boundary_face_integral(
    int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
    std::shared_ptr<BoundaryCondition> bc) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd K_boundary = Eigen::MatrixXd::Zero(n_basis, n_basis);

    // Get face data
    const auto& face_data = mesh->get_face_data(elem_id, face_id);

    // Extract face normal and length
    Eigen::Vector2d normal = face_data.at("normal").head<2>();
    double face_length = face_data.at("length")[0];

    // Get face basis values
    const std::vector<Eigen::MatrixXd>& phi_face = dg_space->get_face_basis_values();
    const Eigen::VectorXd& face_weights = dg_space->get_face_quad()->weights;

    int n_face_quad = face_weights.size();

    // Penalty parameter
    int p = dg_space->get_order();
    double sigma = compute_penalty_parameter(p, face_length);

    auto elem_data = mesh->get_element_data(elem_id);
    Eigen::MatrixXd grad_phi = interpolate_gradient_to_face(elem_data, face_id, dg_space);

    const Eigen::MatrixXd& phi = phi_face[face_id];

    // SIPG formulation for boundary face
    if (bc && bc->get_type() == BCType::DIRICHLET) {
        for (int q = 0; q < n_face_quad; ++q) {
            double w_q = face_weights[q] * face_length * 0.5;

            Eigen::VectorXd v = phi.row(q).transpose();
            Eigen::VectorXd u = v;

            Eigen::VectorXd C_vec(n_basis);
            for (int i = 0; i < n_basis; ++i) {
                Eigen::Vector2d grad_phi_i = grad_phi.block(q * n_basis + i, 0, 1, 2).transpose();
                C_vec(i) = grad_phi_i.dot(normal);
            }

            K_boundary += w_q * (-(v * C_vec.transpose()) - (C_vec * u.transpose()) +
                                 sigma * (v * u.transpose()));
        }
    } else {
        std::cout << "No valid boundary condition provided for element " << elem_id << " face "
                  << face_id << ". Skipping boundary integral." << std::endl;
    }

    return K_boundary;
}

Eigen::VectorXd LaplaceWeakFormulation::compute_source_integral(
    int elem_id, std::function<double(const Eigen::Vector2d&)> source_func,
    std::shared_ptr<DGMesh> mesh) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::VectorXd F_source = Eigen::VectorXd::Zero(n_basis);

    if (!source_func)
        return F_source;

    // Get element data
    const auto& elem_data = mesh->get_element_data(elem_id);
    const Eigen::VectorXd& J_det_vals = elem_data.at("J_det_vol");
    const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
    const Eigen::MatrixXd& vol_basis = dg_space->get_volume_basis_values();

    // Get element vertices for coordinate mapping
    Eigen::MatrixXd vertices(mesh->get_elements().cols(), 2);
    for (int i = 0; i < mesh->get_elements().cols(); ++i) {
        vertices.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem_id, i));
    }

    auto mapping = dg_space->get_mapping();
    int n_quad = weights.size();

    for (int q = 0; q < n_quad; ++q) {
        Eigen::Vector2d xi = dg_space->get_volume_quad()->points.row(q);
        Eigen::Vector2d x_phys = mapping->map_to_physical(vertices, xi);

        double source_val = source_func(x_phys);
        double w_q_phys = weights[q] * J_det_vals[q];

        for (int i = 0; i < n_basis; ++i) {
            F_source[i] += w_q_phys * source_val * vol_basis(q, i);
        }
    }

    return F_source;
}

Eigen::MatrixXd LaplaceWeakFormulation::interpolate_gradient_to_face(
    const std::map<std::string, Eigen::MatrixXd>& elem_data, int face_id,
    std::shared_ptr<DGSpace> dg_space) const {
    int n_face_quad = dg_space->get_face_quad()->weights.size();
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd dphi_dx_face(n_face_quad * n_basis, 2);

    const Eigen::MatrixXd& elem_vertices = elem_data.at("vertices");
    const Eigen::MatrixXd& face_quad_points = dg_space->get_face_quad()->points;

    auto mapping = dg_space->get_mapping();

    for (int q = 0; q < n_face_quad; ++q) {
        Eigen::Vector2d xi_face = dg_space->map_face_quad_point(face_id, face_quad_points(q, 0));

        Eigen::MatrixXd dphi_dxi = dg_space->get_basis()->evaluate_gradient(xi_face);
        auto mapping_data = mapping->compute_mapping(elem_vertices, xi_face);
        const Eigen::Matrix2d& dxi_dx = mapping_data.dxi_dx;

        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d dphi_dx = dxi_dx * dphi_dxi.row(i).transpose();
            dphi_dx_face.block(q * n_basis + i, 0, 1, 2) = dphi_dx.transpose();
        }
    }

    return dphi_dx_face;
}

Eigen::VectorXd
LaplaceWeakFormulation::compute_boundary_rhs_integral(int elem_id, int face_id,
                                                      std::shared_ptr<DGMesh> mesh,
                                                      std::shared_ptr<BoundaryCondition> bc) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::VectorXd F_boundary = Eigen::VectorXd::Zero(n_basis);

    if (!bc)
        return F_boundary;

    if (bc->get_type() == BCType::DIRICHLET) {
        const auto& face_data = mesh->get_face_data(elem_id, face_id);
        double face_length = face_data.at("length")[0];
        Eigen::Vector2d normal = face_data.at("normal").head<2>();

        const std::vector<Eigen::MatrixXd>& phi_face = dg_space->get_face_basis_values();
        const Eigen::VectorXd& face_weights = dg_space->get_face_quad()->weights;

        int n_face_quad = face_weights.size();
        int p = dg_space->get_order();
        double sigma = compute_penalty_parameter(p, face_length);

        auto elem_data = mesh->get_element_data(elem_id);
        Eigen::MatrixXd grad_phi = interpolate_gradient_to_face(elem_data, face_id, dg_space);
        const Eigen::MatrixXd& phi = phi_face[face_id];

        // Get element vertices for coordinate mapping
        Eigen::MatrixXd vertices(mesh->get_elements().cols(), 2);
        for (int i = 0; i < mesh->get_elements().cols(); ++i) {
            vertices.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem_id, i));
        }
        auto mapping = dg_space->get_mapping();
        const Eigen::MatrixXd& face_quad_points = dg_space->get_face_quad()->points;

        for (int q = 0; q < n_face_quad; ++q) {
            Eigen::Vector2d xi_face =
                dg_space->map_face_quad_point(face_id, face_quad_points(q, 0));
            Eigen::Vector2d x_quad = mapping->map_to_physical(vertices, xi_face);

            double bc_value = bc->evaluate(x_quad);
            double w_q = face_weights[q] * face_length * 0.5;

            for (int i = 0; i < n_basis; ++i) {
                double phi_i = phi(q, i);
                Eigen::Vector2d grad_phi_i = grad_phi.block(q * n_basis + i, 0, 1, 2).transpose();
                F_boundary[i] +=
                    w_q * (sigma * bc_value * phi_i - bc_value * grad_phi_i.dot(normal));
            }
        }
    }

    return F_boundary;
}

void LaplaceWeakFormulation::assemble(DGAssembler& assembler,
                                      std::function<double(const Eigen::Vector2d&)> source_func,
                                      std::function<double(const Eigen::Vector2d&)> bc_func) const {
    auto mesh = assembler.get_mesh();
    auto dg_space = assembler.get_dg_space();

    assembler.clear_assembly_data();

    std::cout << "\nAssembling Laplace system..." << std::endl;

    // Volume integrals
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        const auto& elem_data = mesh->get_element_data(elem_id);
        Eigen::MatrixXd K_vol = compute_volume_integral(elem_data, dg_space);
        assembler.add_to_matrix(elem_id, elem_id, K_vol);

        // Source term
        if (source_func) {
            Eigen::VectorXd F_src = compute_source_integral(elem_id, source_func, mesh);
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

                    const auto& elem_vertices_L = mesh->get_elements().row(elem_L);
                    const auto& elem_vertices_R = mesh->get_elements().row(elem_R);
                    Eigen::MatrixXd v_L(elem_vertices_L.size(), mesh->get_vertices().cols());
                    Eigen::MatrixXd v_R(elem_vertices_R.size(), mesh->get_vertices().cols());

                    for (int i = 0; i < elem_vertices_L.size(); ++i) {
                        v_L.row(i) = mesh->get_vertices().row(elem_vertices_L(i));
                    }
                    for (int i = 0; i < elem_vertices_R.size(); ++i) {
                        v_R.row(i) = mesh->get_vertices().row(elem_vertices_R(i));
                    }

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

                Eigen::MatrixXd K_bnd = compute_boundary_face_integral(elem_L, face_L, mesh, bc);
                assembler.add_to_matrix(elem_L, elem_L, K_bnd);

                Eigen::VectorXd F_bnd = compute_boundary_rhs_integral(elem_L, face_L, mesh, bc);
                assembler.add_to_rhs(elem_L, F_bnd);
            }
        }
    }

    std::cout << "Face integrals assembled" << std::endl;

    assembler.finalize_assembly();
}

}  // namespace dgfem
