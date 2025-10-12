/**
 * @file weak_formulation_base.cpp
 * @brief Implementation of base weak formulation classes
 */

#include "dgfem/weak_forms/weak_formulation_base.hpp"
#include "dgfem/solver/assembler.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/mapping.hpp"
#include <iostream>
#include <cmath>
#include <set>

namespace dgfem {

// ============================================================================
// TimeIndependentWeakFormulation implementation
// ============================================================================

void TimeIndependentWeakFormulation::assemble(
    DGAssembler& assembler,
    std::function<double(const Eigen::Vector2d&)> source_func,
    std::function<double(const Eigen::Vector2d&)> bc_func) const {
    
    auto mesh = assembler.get_mesh();
    auto dg_space = assembler.get_dg_space();
    
    assembler.clear_assembly_data();
    
    std::cout << "\nAssembling time-independent system..." << std::endl;
    
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
                std::pair<int, int> face_sig = (elem_L < elem_R) ? 
                    std::make_pair(elem_L, elem_R) : std::make_pair(elem_R, elem_L);
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

                    auto [K_LL, K_LR, K_RL, K_RR] = compute_interior_face_integral(
                        elem_L, face_L, elem_R, face_R, mesh, perm);
                                    
                    assembler.add_to_matrix(elem_L, elem_L, K_LL);
                    assembler.add_to_matrix(elem_L, elem_R, K_LR);
                    assembler.add_to_matrix(elem_R, elem_L, K_RL);
                    assembler.add_to_matrix(elem_R, elem_R, K_RR);
                }
            } else {
                // Boundary face (elem_R == -1)
                // Note: Derived classes must handle boundary conditions appropriately
                // This is a stub that derived classes should override if needed
            }
        }
    }
    
    std::cout << "Face integrals assembled" << std::endl;
    
    assembler.finalize_assembly();
}

Eigen::VectorXd TimeIndependentWeakFormulation::compute_source_integral(
    int elem_id,
    std::function<double(const Eigen::Vector2d&)> source_func,
    std::shared_ptr<DGMesh> mesh) const {
    
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::VectorXd F = Eigen::VectorXd::Zero(n_basis);
    
    const auto& elem_data = mesh->get_element_data(elem_id);
    const Eigen::MatrixXd& phi = elem_data.at("phi");
    const Eigen::MatrixXd& xy = elem_data.at("xy");
    const Eigen::VectorXd& weights = elem_data.at("weights");
    double jac_det = elem_data.at("jac_det")(0);
    
    for (int q = 0; q < weights.size(); ++q) {
        Eigen::Vector2d pt = xy.row(q);
        double f_val = source_func(pt);
        for (int i = 0; i < n_basis; ++i) {
            F(i) += weights(q) * jac_det * f_val * phi(q, i);
        }
    }
    
    return F;
}

// ============================================================================
// TimeDependentWeakFormulation implementation
// ============================================================================

void TimeDependentWeakFormulation::assemble(
    DGAssembler& assembler,
    std::function<double(const Eigen::Vector2d&)> source_func,
    std::function<double(const Eigen::Vector2d&)> bc_func) const {
    
    auto mesh = assembler.get_mesh();
    auto dg_space = assembler.get_dg_space();
    
    assembler.clear_assembly_data();
    
    // Volume term (stiffness integral)
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        const auto& elem_data = mesh->get_element_data(elem_id);
        Eigen::MatrixXd L_vol = compute_volume_integral(elem_data, dg_space);
        assembler.add_to_matrix(elem_id, elem_id, L_vol);
    }
    
    // Interior face terms with upwind flux
    std::set<std::pair<std::pair<int,int>, std::pair<int,int>>> processed_faces;
    
    int n_faces_per_elem = (mesh->get_element_type() == "triangle") ? 3 : 4;
    int n_vertices_per_elem = mesh->get_elements().cols();
    
    for (int elem_L = 0; elem_L < mesh->get_n_elements(); ++elem_L) {
        const auto& neighbors = mesh->get_element_neighbors(elem_L);
        
        for (int face_L = 0; face_L < n_faces_per_elem; ++face_L) {
            int elem_R = neighbors[face_L].first;
            int face_R = neighbors[face_L].second;
            
            if (elem_R >= 0) {
                std::pair<int,int> pair_L = std::make_pair(elem_L, face_L);
                std::pair<int,int> pair_R = std::make_pair(elem_R, face_R);
                auto face_key = std::minmax(pair_L, pair_R);
                
                if (processed_faces.find(face_key) == processed_faces.end()) {
                    processed_faces.insert(face_key);
                    
                    // Compute face permutation
                    Eigen::MatrixXd v_L(n_vertices_per_elem, 2);
                    Eigen::MatrixXd v_R(n_vertices_per_elem, 2);
                    for (int i = 0; i < n_vertices_per_elem; ++i) {
                        v_L.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem_L, i));
                        v_R.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem_R, i));
                    }
                    
                    Eigen::VectorXi perm = dg_space->compute_face_permutation(face_L, v_L, face_R, v_R);
                    
                    auto [L_LL, L_LR, L_RL, L_RR] = compute_interior_face_integral(
                        elem_L, face_L, elem_R, face_R, mesh, perm);
                    
                    assembler.add_to_matrix(elem_L, elem_L, L_LL);
                    assembler.add_to_matrix(elem_L, elem_R, L_LR);
                    assembler.add_to_matrix(elem_R, elem_L, L_RL);
                    assembler.add_to_matrix(elem_R, elem_R, L_RR);
                }
            }
        }
    }
    
    // Boundary face terms
    for (const auto& [elem_id, face_id] : mesh->get_boundary_faces()) {
        auto [L_bc, F_bc] = compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);
        assembler.add_to_matrix(elem_id, elem_id, L_bc);
        assembler.add_to_rhs(elem_id, F_bc);
    }
    
    assembler.finalize_assembly();
}

Eigen::MatrixXd TimeDependentWeakFormulation::compute_mass_integral(
    const std::map<std::string, Eigen::MatrixXd>& elem_data,
    std::shared_ptr<DGSpace> dg_space) const {
    
    int n_basis = dg_space->get_basis()->get_n_basis();
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_basis, n_basis);
    
    const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
    const Eigen::MatrixXd& phi = dg_space->get_volume_basis_values();
    const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");
    
    int n_quad = weights.size();
    
    for (int q = 0; q < n_quad; ++q) {
        double w_q_phys = weights[q] * std::abs(J_det[q]);
        Eigen::VectorXd phi_q = phi.row(q).transpose();
        M += w_q_phys * (phi_q * phi_q.transpose());
    }
    
    return M;
}

} // namespace dgfem
