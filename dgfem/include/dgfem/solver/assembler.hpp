/**
 * @file assembler.hpp
 * @brief DG system assembler
 */

#pragma once

#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"

#include <variant>

#include <functional>
#include <memory>
#include <vector>

#include "trilinos_types.hpp"
#include "weak_form.hpp"

namespace dgfem {

/**
 * @brief DG system assembler for various weak formulations
 */
class DGAssembler {
public:
    /**
     * @brief Unified constructor for any weak formulation
     */
    template <typename WeakFormT>
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
    void assemble(std::function<double(const Vec2&)> source_func = nullptr,
                  std::function<double(const Vec2&)> bc_func = nullptr);

    /**
     * @brief Assemble mass matrix for time-dependent problems
     */
    [[nodiscard]] Teuchos::RCP<TpetraCrsMatrix> assemble_mass_matrix();

    /**
     * @brief Assemble Euler residual for time stepping (used for Euler equations)
     * @param u_coeffs Solution coefficients for each element (n_elem, n_basis, n_vars)
     * @return Residual for each element (n_elem, n_basis, n_vars)
     */
    void assemble_euler_residual(const std::vector<DView2>& u_coeffs,
                                 std::vector<DView2>& residuals_out);

    /**
     * @brief Get the weak formulation type
     */
    [[nodiscard]] std::string get_weak_form_type() const;

    /**
     * @brief Get system matrix
     */
    [[nodiscard]] Teuchos::RCP<const TpetraCrsMatrix> get_system_matrix() const noexcept {
        return system_matrix_;
    }

    /**
     * @brief Get RHS vector
     */
    [[nodiscard]] Teuchos::RCP<const TpetraMultiVector> get_rhs() const noexcept { return rhs_; }

    /**
     * @brief Distribute solution vector back to mesh
     */
    void distribute_solution(const TpetraMultiVector& solution);

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
    void add_to_matrix(int elem_i, int elem_j, const DView2& K_local);

    /**
     * @brief Add to RHS vector
     */
    void add_to_rhs(int elem_id, const DView1& F_local);

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
    [[nodiscard]] std::shared_ptr<LaplaceWeakFormulation> get_laplace_weak_form() const {
        return laplace_weak_form_;
    }

    /**
     * @brief Get the Advection weak formulation (for backward compatibility)
     */
    [[nodiscard]] std::shared_ptr<AdvectionWeakFormulation> get_advection_weak_form() const {
        return advection_weak_form_;
    }

    /**
     * @brief Get the Euler weak formulation (for backward compatibility)
     */
    [[nodiscard]] std::shared_ptr<EulerWeakFormulation> get_euler_weak_form() const {
        return euler_weak_form_;
    }

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
    Teuchos::RCP<const TpetraMap> map_;
    Teuchos::RCP<TpetraCrsMatrix> system_matrix_;
    Teuchos::RCP<TpetraMultiVector> rhs_;

    // Assembly helper: Tpetra::CrsMatrix requires its per-row capacity up front (its
    // constructor's nnz-per-row argument allocates fixed storage; it does not grow
    // dynamically the way Eigen::SparseMatrix's triplet-based setFromTriplets() does), so the
    // sparsity pattern has to be known before the matrix is built. add_to_matrix() collects
    // contributions here; finalize_assembly() sums duplicates per (row, col) and only then
    // constructs system_matrix_ with an exact per-row entry count.
    struct Triplet {
        int row, col;
        double value;
    };
    std::vector<Triplet> triplets_;

    /**
     * @brief Set the weak formulation (helper for constructor)
     */
    template <typename WeakFormT>
    void set_weak_form(std::shared_ptr<WeakFormT> weak_form) {
        weak_form_ = weak_form;
        if constexpr (std::is_base_of_v<LaplaceWeakFormulation, WeakFormT> ||
                      std::is_same_v<WeakFormT, LaplaceWeakFormulation>) {
            laplace_weak_form_ = std::static_pointer_cast<LaplaceWeakFormulation>(weak_form);
        } else if constexpr (std::is_base_of_v<AdvectionWeakFormulation, WeakFormT> ||
                             std::is_same_v<WeakFormT, AdvectionWeakFormulation>) {
            advection_weak_form_ = std::static_pointer_cast<AdvectionWeakFormulation>(weak_form);
        } else if constexpr (std::is_base_of_v<EulerWeakFormulation, WeakFormT> ||
                             std::is_same_v<WeakFormT, EulerWeakFormulation>) {
            euler_weak_form_ = std::static_pointer_cast<EulerWeakFormulation>(weak_form);
        }
    }
};

// Template implementation
template <typename WeakFormT>
DGAssembler::DGAssembler(std::shared_ptr<DGMesh> mesh, std::shared_ptr<WeakFormT> weak_form)
    : mesh_(mesh), dg_space_(mesh->get_dg_space()) {
    set_weak_form(weak_form);
    n_dofs_ =
        mesh->get_n_elements() * dg_space_->get_basis()->get_n_basis() * weak_form->get_n_vars();
    map_ = make_serial_map(n_dofs_);
    rhs_ = Teuchos::rcp(new TpetraMultiVector(map_, 1));
    // Solvers that never call finalize_assembly() (the matrix-free compressible/Euler/NS path)
    // still need get_system_matrix() to return a safely-queryable (if empty) matrix, matching
    // the old Eigen::SparseMatrix default-constructed-empty behavior rather than a null RCP.
    system_matrix_ = Teuchos::rcp(new TpetraCrsMatrix(map_, size_t(0)));
    system_matrix_->fillComplete();
}

}  // namespace dgfem
