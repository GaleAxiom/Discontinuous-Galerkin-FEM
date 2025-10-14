/**
 * @file mapping.cpp
 * @brief Implementation of geometric mapping with modern C++20
 */

#include "dgfem/reference/mapping.hpp"

#include <cmath>

#include <sstream>
#include <stdexcept>

namespace dgfem {

GeometricMapping::GeometricMapping(std::shared_ptr<ReferenceElement> ref_element)
    : ref_element_(std::move(ref_element)) {
    if (!ref_element_) {
        throw std::invalid_argument("Reference element cannot be null");
    }
}

MappingData GeometricMapping::compute_mapping(const Eigen::MatrixXd& vertices,
                                              const Eigen::Vector2d& xi) const {
    MappingData data;

    Eigen::VectorXd N;
    Eigen::MatrixXd dN_dxi;
    compute_shape_functions(xi, N, dN_dxi);

    // Compute physical coordinates
    data.x_phys = vertices.transpose() * N;

    // Compute J_transpose = (dx/dxi)^T directly
    Eigen::Matrix2d J_T = dN_dxi.transpose() * vertices;

    // Store J for compatibility
    data.J = J_T.transpose();

    // Compute Jacobian determinant
    data.J_T_det = J_T.determinant();

    if (std::abs(data.J_T_det) < 1e-12) {
        throw std::runtime_error("Singular Jacobian matrix");
    }

    // The transformation matrix for gradients is inv(J_T)
    data.dxi_dx = J_T.inverse();

    return data;
}

Eigen::Vector2d GeometricMapping::map_to_physical(const Eigen::MatrixXd& vertices,
                                                  const Eigen::Vector2d& xi) const {
    Eigen::VectorXd N;
    Eigen::MatrixXd dN_dxi;
    compute_shape_functions(xi, N, dN_dxi);
    return vertices.transpose() * N;
}

std::optional<Eigen::Vector2d>
GeometricMapping::find_reference_coords(const Eigen::MatrixXd& vertices,
                                        const Eigen::Vector2d& x_phys, double tol,
                                        int max_iter) const {
    // Newton iteration to find xi such that F(xi) = x_mapping(xi) - x_phys = 0
    Eigen::Vector2d xi = Eigen::Vector2d::Zero();  // Initial guess

    for (int iter = 0; iter < max_iter; ++iter) {
        MappingData data = compute_mapping(vertices, xi);
        Eigen::Vector2d residual = data.x_phys - x_phys;

        if (residual.norm() < tol) {
            return xi;  // Success - return the result
        }

        // Newton update: xi_new = xi_old - J^(-1) * F(xi_old)
        xi -= data.dxi_dx.transpose() * residual;

        // Project to bounds using polymorphic method
        ref_element_->project_to_bounds(xi);
    }

    // Convergence failed - return empty optional
    return std::nullopt;
}

void GeometricMapping::compute_shape_functions(const Eigen::Vector2d& xi, Eigen::VectorXd& N,
                                               Eigen::MatrixXd& dN_dxi) const {
    // Use polymorphic method from ReferenceElement
    ref_element_->compute_shape_functions(xi, N, dN_dxi);
}

}  // namespace dgfem