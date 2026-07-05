/**
 * @file mapping.cpp
 * @brief Implementation of geometric mapping with modern C++20
 */

#include "dgfem/reference/mapping.hpp"

#include <cmath>

#include <sstream>
#include <stdexcept>

namespace dgfem {

namespace {
// x_phys = vertices^T * N  (vertices: n_vertices x 2, N: n_vertices)
Vec2 apply_shape_functions(const DView2& vertices, const DView1& N) {
    Vec2 result{0.0, 0.0};
    for (int i = 0; i < static_cast<int>(N.extent(0)); ++i) {
        result[0] += vertices(i, 0) * N[i];
        result[1] += vertices(i, 1) * N[i];
    }
    return result;
}

// J_T = dN_dxi^T * vertices  (dN_dxi: n_vertices x 2, vertices: n_vertices x 2) -> 2x2
Mat2 compute_J_transpose(const DView2& dN_dxi, const DView2& vertices) {
    Mat2 J_T{0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < static_cast<int>(dN_dxi.extent(0)); ++i) {
        J_T.m00 += dN_dxi(i, 0) * vertices(i, 0);
        J_T.m01 += dN_dxi(i, 0) * vertices(i, 1);
        J_T.m10 += dN_dxi(i, 1) * vertices(i, 0);
        J_T.m11 += dN_dxi(i, 1) * vertices(i, 1);
    }
    return J_T;
}
}  // namespace

GeometricMapping::GeometricMapping(std::shared_ptr<ReferenceElement> ref_element)
    : ref_element_(std::move(ref_element)) {
    if (!ref_element_) {
        throw std::invalid_argument("Reference element cannot be null");
    }
}

MappingData GeometricMapping::compute_mapping(const DView2& vertices, const Vec2& xi) const {
    MappingData data;

    DView1 N;
    DView2 dN_dxi;
    compute_shape_functions(xi, N, dN_dxi);

    // Compute physical coordinates
    data.x_phys = apply_shape_functions(vertices, N);

    // Compute J_transpose = (dx/dxi)^T directly
    Mat2 J_T = compute_J_transpose(dN_dxi, vertices);

    // Store J for compatibility
    data.J = J_T.transpose();

    // Compute Jacobian determinant
    data.J_T_det = J_T.det();

    if (std::abs(data.J_T_det) < 1e-12) {
        throw std::runtime_error("Singular Jacobian matrix");
    }

    // The transformation matrix for gradients is inv(J_T)
    data.dxi_dx = J_T.inverse();

    return data;
}

Vec2 GeometricMapping::map_to_physical(const DView2& vertices, const Vec2& xi) const {
    DView1 N;
    DView2 dN_dxi;
    compute_shape_functions(xi, N, dN_dxi);
    return apply_shape_functions(vertices, N);
}

std::optional<Vec2> GeometricMapping::find_reference_coords(const DView2& vertices,
                                                            const Vec2& x_phys, double tol,
                                                            int max_iter) const {
    // Newton iteration to find xi such that F(xi) = x_mapping(xi) - x_phys = 0
    Vec2 xi{0.0, 0.0};  // Initial guess

    for (int iter = 0; iter < max_iter; ++iter) {
        MappingData data = compute_mapping(vertices, xi);
        Vec2 residual = data.x_phys - x_phys;

        if (norm(residual) < tol) {
            return xi;  // Success - return the result
        }

        // Newton update: xi_new = xi_old - J^(-1) * F(xi_old)
        xi = xi - data.dxi_dx.transpose().apply(residual);

        // Project to bounds using polymorphic method
        ref_element_->project_to_bounds(xi);
    }

    // Convergence failed - return empty optional
    return std::nullopt;
}

void GeometricMapping::compute_shape_functions(const Vec2& xi, DView1& N, DView2& dN_dxi) const {
    // Use polymorphic method from ReferenceElement
    ref_element_->compute_shape_functions(xi, N, dN_dxi);
}

}  // namespace dgfem
