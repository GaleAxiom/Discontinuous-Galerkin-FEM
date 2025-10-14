/**
 * @file assembler.cpp
 * @brief Implementation of DG assembler
 */

#include "dgfem/solver/assembler.hpp"

#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace dgfem {

std::string DGAssembler::get_weak_form_type() const {
    if (!weak_form_) {
        return "None";
    }
    return weak_form_->get_type();
}

void DGAssembler::assemble(std::function<double(const Eigen::Vector2d&)> source_func,
                           std::function<double(const Eigen::Vector2d&)> bc_func) {
    if (!weak_form_) {
        throw std::runtime_error("Weak formulation not set");
    }

    // Delegate assembly to the weak formulation
    weak_form_->assemble(*this, source_func, bc_func);
}

Eigen::SparseMatrix<double> DGAssembler::assemble_mass_matrix() {
    // Use polymorphism to work with any time-dependent weak formulation
    // This replaces the old approach of checking for specific formulation types
    if (!weak_form_) {
        throw std::runtime_error("Weak formulation not set");
    }

    if (!weak_form_->is_time_dependent()) {
        throw std::runtime_error("Mass matrix assembly requires a time-dependent weak formulation");
    }

    // Cast to TimeDependentWeakFormulation to access compute_mass_integral
    auto time_dep_weak_form = std::dynamic_pointer_cast<TimeDependentWeakFormulation>(weak_form_);
    if (!time_dep_weak_form) {
        throw std::runtime_error("Failed to cast to TimeDependentWeakFormulation");
    }

    int n_basis = dg_space_->get_basis()->get_n_basis();
    std::vector<Eigen::Triplet<double>> mass_triplets;

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        auto dof_indices = get_dof_indices(elem_id, 0);
        const auto& elem_data = mesh_->get_element_data(elem_id);

        Eigen::MatrixXd M_local = time_dep_weak_form->compute_mass_integral(elem_data, dg_space_);

        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                if (std::abs(M_local(i, j)) > 1e-14) {
                    mass_triplets.emplace_back(dof_indices[i], dof_indices[j], M_local(i, j));
                }
            }
        }
    }

    Eigen::SparseMatrix<double> mass_matrix(n_dofs_, n_dofs_);
    mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
    return mass_matrix;
}

void DGAssembler::distribute_solution(const Eigen::VectorXd& solution) {
    if (solution.size() != n_dofs_) {
        throw std::invalid_argument("Solution vector size mismatch");
    }

    auto mesh_solution = mesh_->get_solution();
    int n_basis = dg_space_->get_basis()->get_n_basis();

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        auto dof_indices = get_dof_indices(elem_id, 0);
        Eigen::VectorXd elem_coeffs(n_basis);

        for (int i = 0; i < n_basis; ++i) {
            elem_coeffs[i] = solution[dof_indices[i]];
        }

        mesh_solution->set_element_coeffs(elem_id, 0, elem_coeffs);
    }
}

std::vector<int> DGAssembler::get_dof_indices(int elem_id, int var_id) const {
    int n_basis = dg_space_->get_basis()->get_n_basis();
    std::vector<int> indices(n_basis);

    for (int i = 0; i < n_basis; ++i) {
        indices[i] = elem_id * n_basis + i;
    }

    return indices;
}

void DGAssembler::add_to_matrix(int elem_i, int elem_j, const Eigen::MatrixXd& K_local) {
    auto dofs_i = get_dof_indices(elem_i, 0);
    auto dofs_j = get_dof_indices(elem_j, 0);

    for (int i = 0; i < static_cast<int>(dofs_i.size()); ++i) {
        for (int j = 0; j < static_cast<int>(dofs_j.size()); ++j) {
            if (std::abs(K_local(i, j)) > 1e-14) {
                triplets_.emplace_back(dofs_i[i], dofs_j[j], K_local(i, j));
            }
        }
    }
}

void DGAssembler::add_to_rhs(int elem_id, const Eigen::VectorXd& F_local) {
    auto dof_indices = get_dof_indices(elem_id, 0);

    for (int i = 0; i < static_cast<int>(dof_indices.size()); ++i) {
        rhs_[dof_indices[i]] += F_local[i];
    }
}

void DGAssembler::clear_assembly_data() {
    triplets_.clear();
    rhs_.setZero();
}

void DGAssembler::finalize_assembly() {
    system_matrix_.resize(n_dofs_, n_dofs_);
    system_matrix_.setFromTriplets(triplets_.begin(), triplets_.end());
    system_matrix_.makeCompressed();
}

std::vector<Eigen::MatrixXd>
DGAssembler::assemble_euler_residual(const std::vector<Eigen::MatrixXd>& u_coeffs) {
    if (!euler_weak_form_) {
        throw std::runtime_error("Euler weak formulation not set");
    }

    static int call_count = 0;
    static double total_vol_time = 0.0;
    static double total_face_time = 0.0;

    int n_elem = mesh_->get_n_elements();
    int n_basis = dg_space_->get_basis()->get_n_basis();
    int n_vars = 4;  // [rho, rho*u, rho*v, E]

    std::vector<Eigen::MatrixXd> residuals(n_elem, Eigen::MatrixXd::Zero(n_basis, n_vars));

    // Volume contributions
    auto vol_start = std::chrono::high_resolution_clock::now();
    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        residuals[elem_id] =
            euler_weak_form_->volume_residual(u_coeffs[elem_id], elem_data, dg_space_);
    }

    if (euler_weak_form_->has_viscous_terms()) {
        for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
            const auto& elem_data = mesh_->get_element_data(elem_id);
            residuals[elem_id] +=
                euler_weak_form_->viscous_volume_residual(u_coeffs[elem_id], elem_data, dg_space_);
        }
    }
    auto vol_end = std::chrono::high_resolution_clock::now();
    total_vol_time += std::chrono::duration<double>(vol_end - vol_start).count();

    // Face contributions - using precomputed face data
    auto face_start = std::chrono::high_resolution_clock::now();

    // Process interior faces
    const auto& interior_faces = mesh_->get_interior_faces();
    for (const auto& face : interior_faces) {
        const auto& face_data_L = mesh_->get_element_face_data(face.elem_L, face.face_L);
        const auto& face_data_R = mesh_->get_element_face_data(face.elem_R, face.face_R);

        auto [R_face_L, R_face_R] = euler_weak_form_->interior_face_residual(
            u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, dg_space_,
            face.permutation);

        residuals[face.elem_L] += R_face_L;
        residuals[face.elem_R] += R_face_R;

        if (euler_weak_form_->has_viscous_terms()) {
            auto [R_visc_L, R_visc_R] = euler_weak_form_->viscous_interior_face_residual(
                u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, dg_space_,
                face.permutation);
            residuals[face.elem_L] += R_visc_L;
            residuals[face.elem_R] += R_visc_R;
        }
    }

    // Process boundary faces
    const auto& boundary_faces = mesh_->get_boundary_face_data();
    for (const auto& face : boundary_faces) {
        if (face.bc_euler) {
            const auto& face_data = mesh_->get_element_face_data(face.elem_L, face.face_L);
            Eigen::MatrixXd R_face_bc = euler_weak_form_->boundary_face_residual(
                u_coeffs[face.elem_L], face_data, face.bc_euler, dg_space_);

            residuals[face.elem_L] += R_face_bc;

            if (euler_weak_form_->has_viscous_terms()) {
                Eigen::MatrixXd R_visc_bc = euler_weak_form_->viscous_boundary_face_residual(
                    u_coeffs[face.elem_L], face_data, face.bc_euler, dg_space_);
                residuals[face.elem_L] += R_visc_bc;
            }
        }
    }

    auto face_end = std::chrono::high_resolution_clock::now();
    total_face_time += std::chrono::duration<double>(face_end - face_start).count();

    call_count++;
    // Print timing every 300 calls (100 time steps * 3 RK stages)
    if (call_count % 300 == 0) {
        std::cout << "[TIMER] Residual assembly (" << call_count
                  << " calls): " << "Volume=" << std::fixed << std::setprecision(4)
                  << total_vol_time << "s, " << "Face=" << total_face_time << "s, "
                  << "Total=" << (total_vol_time + total_face_time) << "s" << std::endl;
    }

    return residuals;
}

}  // namespace dgfem