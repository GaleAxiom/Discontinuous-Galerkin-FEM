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
#include <map>

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

Teuchos::RCP<TpetraCrsMatrix> DGAssembler::assemble_mass_matrix() {
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
    // Block-diagonal (mass integrals never couple different elements): n_basis nonzeros/row.
    auto mass_matrix = Teuchos::rcp(new TpetraCrsMatrix(map_, n_basis));

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        auto dof_indices = get_dof_indices(elem_id, 0);
        const auto& elem_data = mesh_->get_element_data(elem_id);

        Eigen::MatrixXd M_local = time_dep_weak_form->compute_mass_integral(elem_data, dg_space_);

        for (int i = 0; i < n_basis; ++i) {
            std::vector<TpetraGlobalOrdinal> cols;
            std::vector<TpetraScalar> vals;
            for (int j = 0; j < n_basis; ++j) {
                if (std::abs(M_local(i, j)) > 1e-14) {
                    cols.push_back(dof_indices[j]);
                    vals.push_back(M_local(i, j));
                }
            }
            mass_matrix->insertGlobalValues(dof_indices[i], cols, vals);
        }
    }
    mass_matrix->fillComplete();
    return mass_matrix;
}

void DGAssembler::distribute_solution(const TpetraMultiVector& solution) {
    if (static_cast<int>(solution.getGlobalLength()) != n_dofs_) {
        throw std::invalid_argument("Solution vector size mismatch");
    }

    auto mesh_solution = mesh_->get_solution();
    int n_basis = dg_space_->get_basis()->get_n_basis();
    auto view = solution.getLocalViewHost(Tpetra::Access::ReadOnly);

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        auto dof_indices = get_dof_indices(elem_id, 0);
        Eigen::VectorXd elem_coeffs(n_basis);

        for (int i = 0; i < n_basis; ++i) {
            elem_coeffs[i] = view(dof_indices[i], 0);
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
    auto view = rhs_->getLocalViewHost(Tpetra::Access::ReadWrite);

    for (int i = 0; i < static_cast<int>(dof_indices.size()); ++i) {
        view(dof_indices[i], 0) += F_local[i];
    }
}

void DGAssembler::clear_assembly_data() {
    triplets_.clear();
    rhs_->putScalar(0.0);
}

void DGAssembler::finalize_assembly() {
    // Group triplets by row so each row's nonzero columns can be inserted together (Tpetra
    // has no direct (row, col, value) triplet-list constructor like Eigen::setFromTriplets).
    std::map<TpetraGlobalOrdinal, std::map<TpetraGlobalOrdinal, double>> rows;
    for (const auto& t : triplets_) {
        rows[t.row()][t.col()] += t.value();
    }

    std::vector<size_t> num_entries_per_row(n_dofs_, 0);
    for (const auto& [row, cols] : rows) {
        num_entries_per_row[row] = cols.size();
    }
    system_matrix_ = Teuchos::rcp(
        new TpetraCrsMatrix(map_, Teuchos::ArrayView<const size_t>(num_entries_per_row)));
    for (const auto& [row, cols] : rows) {
        std::vector<TpetraGlobalOrdinal> col_ids;
        std::vector<TpetraScalar> values;
        col_ids.reserve(cols.size());
        values.reserve(cols.size());
        for (const auto& [col, value] : cols) {
            col_ids.push_back(col);
            values.push_back(value);
        }
        system_matrix_->insertGlobalValues(row, col_ids, values);
    }
    system_matrix_->fillComplete();
}

void DGAssembler::assemble_euler_residual(const std::vector<Eigen::MatrixXd>& u_coeffs,
                                          std::vector<Eigen::MatrixXd>& residuals_out) {
    // if (!euler_weak_form_) {
    //     throw std::runtime_error("Euler weak formulation not set");
    // }

    static int call_count = 0;
    static double total_resize_time = 0.0;
    static double total_volume_time = 0.0;
    static double total_visc_volume_time = 0.0;
    static double total_interior_face_time = 0.0;
    static double total_interior_visc_time = 0.0;
    static double total_boundary_face_time = 0.0;
    static double total_boundary_visc_time = 0.0;

    int n_elem = mesh_->get_n_elements();
    int n_basis = dg_space_->get_basis()->get_n_basis();
    int n_vars = 4;  // [rho, rho*u, rho*v, E]

    {
        auto resize_start = std::chrono::high_resolution_clock::now();
        residuals_out.resize(n_elem);
        for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
            auto& elem_residual = residuals_out[elem_id];
            if (elem_residual.rows() != n_basis || elem_residual.cols() != n_vars) {
                elem_residual.resize(n_basis, n_vars);
            }
            elem_residual.setZero();
        }
        auto resize_end = std::chrono::high_resolution_clock::now();
        total_resize_time += std::chrono::duration<double>(resize_end - resize_start).count();
    }

    // Volume contributions
    auto vol_start = std::chrono::high_resolution_clock::now();
    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        residuals_out[elem_id] =
            euler_weak_form_->volume_residual(u_coeffs[elem_id], elem_data, dg_space_);
    }
    auto vol_end = std::chrono::high_resolution_clock::now();
    total_volume_time += std::chrono::duration<double>(vol_end - vol_start).count();

    const bool has_viscous_terms = euler_weak_form_->has_viscous_terms();

    if (has_viscous_terms) {
        auto visc_vol_start = std::chrono::high_resolution_clock::now();
        for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
            const auto& elem_data = mesh_->get_element_data(elem_id);
            residuals_out[elem_id] +=
                euler_weak_form_->viscous_volume_residual(u_coeffs[elem_id], elem_data, dg_space_);
        }
        auto visc_vol_end = std::chrono::high_resolution_clock::now();
        total_visc_volume_time +=
            std::chrono::duration<double>(visc_vol_end - visc_vol_start).count();
    }

    // Face contributions - using precomputed face data
    // Process interior faces
    auto interior_face_start = std::chrono::high_resolution_clock::now();
    const auto& interior_faces = mesh_->get_interior_faces();
    for (const auto& face : interior_faces) {
        const auto& face_data_L = mesh_->get_element_face_data(face.elem_L, face.face_L);
        const auto& face_data_R = mesh_->get_element_face_data(face.elem_R, face.face_R);

        auto [R_face_L, R_face_R] = euler_weak_form_->interior_face_residual(
            u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, dg_space_,
            face.permutation);

        residuals_out[face.elem_L] += R_face_L;
        residuals_out[face.elem_R] += R_face_R;
    }
    auto interior_face_end = std::chrono::high_resolution_clock::now();
    total_interior_face_time +=
        std::chrono::duration<double>(interior_face_end - interior_face_start).count();

    if (has_viscous_terms) {
        auto interior_visc_start = std::chrono::high_resolution_clock::now();
        for (const auto& face : interior_faces) {
            const auto& face_data_L = mesh_->get_element_face_data(face.elem_L, face.face_L);
            const auto& face_data_R = mesh_->get_element_face_data(face.elem_R, face.face_R);

            auto [R_visc_L, R_visc_R] = euler_weak_form_->viscous_interior_face_residual(
                u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, dg_space_,
                face.permutation);
            residuals_out[face.elem_L] += R_visc_L;
            residuals_out[face.elem_R] += R_visc_R;
        }
        auto interior_visc_end = std::chrono::high_resolution_clock::now();
        total_interior_visc_time +=
            std::chrono::duration<double>(interior_visc_end - interior_visc_start).count();
    }

    // Process boundary faces
    auto boundary_face_start = std::chrono::high_resolution_clock::now();
    const auto& boundary_faces = mesh_->get_boundary_face_data();
    for (const auto& face : boundary_faces) {
        if (face.bc_euler) {
            const auto& face_data = mesh_->get_element_face_data(face.elem_L, face.face_L);
            Eigen::MatrixXd R_face_bc = euler_weak_form_->boundary_face_residual(
                u_coeffs[face.elem_L], face_data, face.bc_euler, dg_space_);

            residuals_out[face.elem_L] += R_face_bc;
        }
    }
    auto boundary_face_end = std::chrono::high_resolution_clock::now();
    total_boundary_face_time +=
        std::chrono::duration<double>(boundary_face_end - boundary_face_start).count();

    if (has_viscous_terms) {
        auto boundary_visc_start = std::chrono::high_resolution_clock::now();
        for (const auto& face : boundary_faces) {
            if (face.bc_euler) {
                const auto& face_data = mesh_->get_element_face_data(face.elem_L, face.face_L);
                Eigen::MatrixXd R_visc_bc = euler_weak_form_->viscous_boundary_face_residual(
                    u_coeffs[face.elem_L], face_data, face.bc_euler, dg_space_);
                residuals_out[face.elem_L] += R_visc_bc;
            }
        }
        auto boundary_visc_end = std::chrono::high_resolution_clock::now();
        total_boundary_visc_time +=
            std::chrono::duration<double>(boundary_visc_end - boundary_visc_start).count();
    }

    call_count++;
    // Print timing every 300 calls (100 time steps * 3 RK stages)
    if (call_count % 300 == 0) {
        double total_face_time = total_interior_face_time + total_interior_visc_time +
                                 total_boundary_face_time + total_boundary_visc_time;
        double total_volume = total_volume_time + total_visc_volume_time;
        double grand_total = total_resize_time + total_volume + total_face_time;

        std::cout << "[TIMER] Residual assembly (" << call_count << " calls): "
                  << "Resize=" << std::fixed << std::setprecision(4) << total_resize_time << "s, "
                  << "Vol=" << total_volume_time << "s, "
                  << "ViscVol=" << total_visc_volume_time << "s, "
                  << "IntFace=" << total_interior_face_time << "s, "
                  << "IntVisc=" << total_interior_visc_time << "s, "
                  << "BndFace=" << total_boundary_face_time << "s, "
                  << "BndVisc=" << total_boundary_visc_time << "s, "
                  << "Total=" << grand_total << "s" << std::endl;
    }
}

}  // namespace dgfem