/**
 * @file solution.hpp
 * @brief DG solution storage and manipulation
 */

#pragma once

#include "dgfem/kokkos_math.hpp"

#include <vector>

namespace dgfem {

/**
 * @brief Container for DG solution coefficients
 */
class DGSolution {
public:
    DGSolution(int n_elements, int n_basis, int n_variables = 1);

    // Delete copy, default move
    DGSolution(const DGSolution&) = delete;
    DGSolution& operator=(const DGSolution&) = delete;
    DGSolution(DGSolution&&) noexcept = default;
    DGSolution& operator=(DGSolution&&) noexcept = default;

    /**
     * @brief Get element coefficients for all variables
     * @param elem_id Element index
     * @return Matrix of size (n_basis x n_variables)
     */
    [[nodiscard]] DView2 get_element_coeffs(int elem_id) const;

    /**
     * @brief Set element coefficients for all variables
     */
    void set_element_coeffs(int elem_id, const DView2& coeffs);

    /**
     * @brief Get element coefficients for specific variable
     */
    [[nodiscard]] DView1 get_element_coeffs(int elem_id, int var_id) const;

    /**
     * @brief Set element coefficients for specific variable
     */
    void set_element_coeffs(int elem_id, int var_id, const DView1& coeffs);

    /**
     * @brief Get all coefficients as a flat vector
     */
    [[nodiscard]] DView1 get_global_coeffs() const;

    /**
     * @brief Set all coefficients from a flat vector
     */
    void set_global_coeffs(const DView1& coeffs);

    /**
     * @brief Zero out all coefficients
     */
    void zero();

    // Getters
    [[nodiscard]] int get_n_elements() const noexcept { return n_elements_; }
    [[nodiscard]] int get_n_basis() const noexcept { return n_basis_; }
    [[nodiscard]] int get_n_variables() const noexcept { return n_variables_; }
    [[nodiscard]] int get_total_dofs() const noexcept {
        return n_elements_ * n_basis_ * n_variables_;
    }

    /**
     * @brief Get DOF indices for element and variable
     */
    [[nodiscard]] std::vector<int> get_dof_indices(int elem_id, int var_id = 0) const;

private:
    int n_elements_;
    int n_basis_;
    int n_variables_;

    // Storage: coeffs_[elem][basis][var]
    std::vector<std::vector<std::vector<double>>> coeffs_;

    /**
     * @brief Convert flat index to (elem, basis, var) indices
     */
    [[nodiscard]] std::tuple<int, int, int> flat_to_indices(int flat_idx) const noexcept;

    /**
     * @brief Convert (elem, basis, var) indices to flat index
     */
    [[nodiscard]] int indices_to_flat(int elem_id, int basis_id, int var_id) const noexcept;
};

}  // namespace dgfem