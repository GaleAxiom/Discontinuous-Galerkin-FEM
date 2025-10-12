/**
 * @file time_stepping.cpp
 * @brief Implementation of time-stepping utilities
 */

#include "dgfem/solver/time_stepping.hpp"

namespace dgfem {

BlockMassMatrix::BlockMassMatrix(const std::vector<Eigen::MatrixXd>& mass_blocks)
    : n_elem_(static_cast<int>(mass_blocks.size()))
{
    if (mass_blocks.empty()) {
        throw std::runtime_error("Cannot create BlockMassMatrix from empty input");
    }
    
    n_basis_ = static_cast<int>(mass_blocks[0].rows());
    M_inv_blocks_.reserve(n_elem_);
    
    // Precompute all inverse blocks
    for (const auto& M_block : mass_blocks) {
        if (M_block.rows() != n_basis_ || M_block.cols() != n_basis_) {
            throw std::runtime_error("Inconsistent mass matrix block sizes");
        }
        M_inv_blocks_.push_back(M_block.inverse());
    }
}

Eigen::VectorXd BlockMassMatrix::apply_inverse(const Eigen::VectorXd& vec) const {
    if (vec.size() != n_elem_ * n_basis_) {
        throw std::runtime_error("Vector size mismatch in apply_inverse");
    }
    
    Eigen::VectorXd result(vec.size());
    
    for (int elem_id = 0; elem_id < n_elem_; ++elem_id) {
        int start = elem_id * n_basis_;
        result.segment(start, n_basis_) = 
            M_inv_blocks_[elem_id] * vec.segment(start, n_basis_);
    }
    
    return result;
}

std::vector<Eigen::MatrixXd> BlockMassMatrix::apply_inverse(
    const std::vector<Eigen::MatrixXd>& vec) const 
{
    if (static_cast<int>(vec.size()) != n_elem_) {
        throw std::runtime_error("Number of elements mismatch in apply_inverse");
    }
    
    std::vector<Eigen::MatrixXd> result(n_elem_);
    
    for (int elem_id = 0; elem_id < n_elem_; ++elem_id) {
        if (vec[elem_id].rows() != n_basis_) {
            throw std::runtime_error("Basis size mismatch in apply_inverse");
        }
        result[elem_id] = M_inv_blocks_[elem_id] * vec[elem_id];
    }
    
    return result;
}

} // namespace dgfem
