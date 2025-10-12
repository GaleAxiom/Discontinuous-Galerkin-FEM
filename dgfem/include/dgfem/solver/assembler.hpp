/**
 * @file assembler.hpp
 * @brief DG system assembler
 */

#pragma once

#include "weak_form.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include <Eigen/Sparse>
#include <functional>
#include <memory>
#include <vector>
#include <variant>

namespace dgfem {

/**
 * @brief DG system assembler for various weak formulations
 */
class DGAssembler {
public:
    /**
     * @brief Unified constructor for any weak formulation
     */
    template<typename WeakFormT>
    DGAssembler(std::shared_ptr<DGMesh> mesh, std::shared_ptr<WeakFormT> weak_form);
    
    // Delete copy, default move
    DGAssembler(const DGAssembler&) = delete;
    DGAssembler& operator=(const DGAssembler&) = delete;
    DGAssembler(DGAssembler&&) noexcept = default;
    DGAssembler& operator=(DGAssembler&&) noexcept = default;
    
    /**
     * @brief Unified assembly method with optional parameters
     * @param source_func Source function for Laplace/Poisson equations
     * @param bc_func Boundary condition function for time-dependent equations
     * @return For time-dependent equations, returns (operator_matrix, boundary_vector)
     */
    void assemble(std::function<double(const Eigen::Vector2d&)> source_func = nullptr,
                  std::function<double(const Eigen::Vector2d&)> bc_func = nullptr);

    
    /**
     * @brief Assemble mass matrix for time-dependent problems
     */
    [[nodiscard]] Eigen::SparseMatrix<double> assemble_mass_matrix();
    
    /**
     * @brief Assemble Euler residual for time stepping (used for Euler equations)
     * @param u_coeffs Solution coefficients for each element (n_elem, n_basis, n_vars)
     * @return Residual for each element (n_elem, n_basis, n_vars)
     */
    [[nodiscard]] std::vector<Eigen::MatrixXd> assemble_euler_residual(
        const std::vector<Eigen::MatrixXd>& u_coeffs);
    
    /**
     * @brief Get the weak formulation type
     */
    [[nodiscard]] std::string get_weak_form_type() const;

    
    /**
     * @brief Get system matrix
     */
    [[nodiscard]] const Eigen::SparseMatrix<double>& get_system_matrix() const noexcept { return system_matrix_; }
    
    /**
     * @brief Get RHS vector
     */
    [[nodiscard]] const Eigen::VectorXd& get_rhs() const noexcept { return rhs_; }
    
    /**
     * @brief Distribute solution vector back to mesh
     */
    void distribute_solution(const Eigen::VectorXd& solution);
    
    // Assembly utilities for weak formulations
    /**
     * @brief Get mesh
     */
    [[nodiscard]] std::shared_ptr<DGMesh> get_mesh() const noexcept { return mesh_; }
    
    /**
     * @brief Get DG space
     */
    [[nodiscard]] std::shared_ptr<DGSpace> get_dg_space() const noexcept { return dg_space_; }
    
    /**
     * @brief Get DOF indices for element
     */
    [[nodiscard]] std::vector<int> get_dof_indices(int elem_id, int var_id = 0) const;
    
    /**
     * @brief Add local matrix to global system
     */
    void add_to_matrix(int elem_i, int elem_j, const Eigen::MatrixXd& K_local);
    
    /**
     * @brief Add to RHS vector
     */
    void add_to_rhs(int elem_id, const Eigen::VectorXd& F_local);
    
    /**
     * @brief Clear assembly data
     */
    void clear_assembly_data();
    
    /**
     * @brief Finalize matrix assembly
     */
    void finalize_assembly();
    
    /**
     * @brief Get the Laplace weak formulation (for backward compatibility)
     */
    [[nodiscard]] std::shared_ptr<LaplaceWeakFormulation> get_laplace_weak_form() const { return laplace_weak_form_; }
    
    /**
     * @brief Get the Advection weak formulation (for backward compatibility)
     */
    [[nodiscard]] std::shared_ptr<AdvectionWeakFormulation> get_advection_weak_form() const { return advection_weak_form_; }
    
    /**
     * @brief Get the Euler weak formulation (for backward compatibility)
     */
    [[nodiscard]] std::shared_ptr<EulerWeakFormulation> get_euler_weak_form() const { return euler_weak_form_; }

private:
    std::shared_ptr<DGMesh> mesh_;
    std::shared_ptr<DGSpace> dg_space_;
    
    // Weak formulation (polymorphic)
    std::shared_ptr<WeakFormulation> weak_form_;
    
    // Type-specific weak formulations (for downcasting)
    std::shared_ptr<LaplaceWeakFormulation> laplace_weak_form_;
    std::shared_ptr<AdvectionWeakFormulation> advection_weak_form_;
    std::shared_ptr<EulerWeakFormulation> euler_weak_form_;
    
    // System data
    int n_dofs_;
    Eigen::SparseMatrix<double> system_matrix_;
    Eigen::VectorXd rhs_;
    
    // Assembly helpers
    std::vector<Eigen::Triplet<double>> triplets_;
    
    /**
     * @brief Set the weak formulation (helper for constructor)
     */
    template<typename WeakFormT>
    void set_weak_form(std::shared_ptr<WeakFormT> weak_form) {
        weak_form_ = weak_form;
        if constexpr (std::is_base_of_v<LaplaceWeakFormulation, WeakFormT> || std::is_same_v<WeakFormT, LaplaceWeakFormulation>) {
            laplace_weak_form_ = std::static_pointer_cast<LaplaceWeakFormulation>(weak_form);
        } else if constexpr (std::is_base_of_v<AdvectionWeakFormulation, WeakFormT> || std::is_same_v<WeakFormT, AdvectionWeakFormulation>) {
            advection_weak_form_ = std::static_pointer_cast<AdvectionWeakFormulation>(weak_form);
        } else if constexpr (std::is_base_of_v<EulerWeakFormulation, WeakFormT> || std::is_same_v<WeakFormT, EulerWeakFormulation>) {
            euler_weak_form_ = std::static_pointer_cast<EulerWeakFormulation>(weak_form);
        }
    }
};

// Template implementation
template<typename WeakFormT>
DGAssembler::DGAssembler(std::shared_ptr<DGMesh> mesh, std::shared_ptr<WeakFormT> weak_form)
    : mesh_(mesh), dg_space_(mesh->get_dg_space()) {
    set_weak_form(weak_form);
    n_dofs_ = mesh->get_n_elements() * dg_space_->get_basis()->get_n_basis() * weak_form->get_n_vars();
    rhs_.resize(n_dofs_);
}

} // namespace dgfem
