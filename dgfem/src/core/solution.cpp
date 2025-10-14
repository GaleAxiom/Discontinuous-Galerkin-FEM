/**
 * @file solution.cpp
 * @brief Implementation of DG solution storage
 */

#include "dgfem/core/solution.hpp"

#include <stdexcept>

namespace dgfem {

DGSolution::DGSolution(int n_elements, int n_basis, int n_variables)
    : n_elements_(n_elements), n_basis_(n_basis), n_variables_(n_variables) {
    // Initialize storage
    coeffs_.resize(n_elements_);
    for (int elem = 0; elem < n_elements_; ++elem) {
        coeffs_[elem].resize(n_basis_);
        for (int basis = 0; basis < n_basis_; ++basis) {
            coeffs_[elem][basis].resize(n_variables_, 0.0);
        }
    }
}

Eigen::MatrixXd DGSolution::get_element_coeffs(int elem_id) const {
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }

    Eigen::MatrixXd coeffs(n_basis_, n_variables_);
    for (int basis = 0; basis < n_basis_; ++basis) {
        for (int var = 0; var < n_variables_; ++var) {
            coeffs(basis, var) = coeffs_[elem_id][basis][var];
        }
    }
    return coeffs;
}

void DGSolution::set_element_coeffs(int elem_id, const Eigen::MatrixXd& coeffs) {
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    if (coeffs.rows() != n_basis_ || coeffs.cols() != n_variables_) {
        throw std::invalid_argument("Coefficient matrix size mismatch");
    }

    for (int basis = 0; basis < n_basis_; ++basis) {
        for (int var = 0; var < n_variables_; ++var) {
            coeffs_[elem_id][basis][var] = coeffs(basis, var);
        }
    }
}

Eigen::VectorXd DGSolution::get_element_coeffs(int elem_id, int var_id) const {
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    if (var_id < 0 || var_id >= n_variables_) {
        throw std::out_of_range("Variable index out of range");
    }

    Eigen::VectorXd coeffs(n_basis_);
    for (int basis = 0; basis < n_basis_; ++basis) {
        coeffs[basis] = coeffs_[elem_id][basis][var_id];
    }
    return coeffs;
}

void DGSolution::set_element_coeffs(int elem_id, int var_id, const Eigen::VectorXd& coeffs) {
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    if (var_id < 0 || var_id >= n_variables_) {
        throw std::out_of_range("Variable index out of range");
    }
    if (coeffs.size() != n_basis_) {
        throw std::invalid_argument("Coefficient vector size mismatch");
    }

    for (int basis = 0; basis < n_basis_; ++basis) {
        coeffs_[elem_id][basis][var_id] = coeffs[basis];
    }
}

Eigen::VectorXd DGSolution::get_global_coeffs() const {
    Eigen::VectorXd global_coeffs(get_total_dofs());

    for (int elem = 0; elem < n_elements_; ++elem) {
        for (int basis = 0; basis < n_basis_; ++basis) {
            for (int var = 0; var < n_variables_; ++var) {
                int flat_idx = indices_to_flat(elem, basis, var);
                global_coeffs[flat_idx] = coeffs_[elem][basis][var];
            }
        }
    }

    return global_coeffs;
}

void DGSolution::set_global_coeffs(const Eigen::VectorXd& coeffs) {
    if (coeffs.size() != get_total_dofs()) {
        throw std::invalid_argument("Global coefficient vector size mismatch");
    }

    for (int flat_idx = 0; flat_idx < coeffs.size(); ++flat_idx) {
        auto [elem, basis, var] = flat_to_indices(flat_idx);
        coeffs_[elem][basis][var] = coeffs[flat_idx];
    }
}

void DGSolution::zero() {
    for (int elem = 0; elem < n_elements_; ++elem) {
        for (int basis = 0; basis < n_basis_; ++basis) {
            for (int var = 0; var < n_variables_; ++var) {
                coeffs_[elem][basis][var] = 0.0;
            }
        }
    }
}

std::vector<int> DGSolution::get_dof_indices(int elem_id, int var_id) const {
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    if (var_id < 0 || var_id >= n_variables_) {
        throw std::out_of_range("Variable index out of range");
    }

    std::vector<int> indices(n_basis_);
    for (int basis = 0; basis < n_basis_; ++basis) {
        indices[basis] = indices_to_flat(elem_id, basis, var_id);
    }
    return indices;
}

std::tuple<int, int, int> DGSolution::flat_to_indices(int flat_idx) const noexcept {
    int elem = flat_idx / (n_basis_ * n_variables_);
    int remainder = flat_idx % (n_basis_ * n_variables_);
    int basis = remainder / n_variables_;
    int var = remainder % n_variables_;
    return {elem, basis, var};
}

int DGSolution::indices_to_flat(int elem_id, int basis_id, int var_id) const noexcept {
    return elem_id * n_basis_ * n_variables_ + basis_id * n_variables_ + var_id;
}

}  // namespace dgfem