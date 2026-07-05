/**
 * @file mapping.hpp
 * @brief Geometric mapping between reference and physical elements with modern C++20
 */

#pragma once

#include <optional>

#include <memory>

#include "elements.hpp"

namespace dgfem {

/**
 * @brief Threshold below which a Jacobian determinant is treated as singular (degenerate or
 * inverted element). Shared by GeometricMapping::compute_mapping()'s throw check and
 * MappingData::is_valid() so the two can't drift apart -- compute_mapping() always throws
 * before returning a MappingData with |J_T_det| below this, so is_valid() should agree with it
 * exactly rather than use a separate, tighter bound that could never actually observe false.
 */
inline constexpr double kSingularJacobianTolerance = 1e-12;

/**
 * @brief Geometric mapping data computed at a specific point
 * Uses aggregate initialization
 */
struct MappingData {
    Vec2 x_phys;     ///< Physical coordinates
    Mat2 J;          ///< Jacobian transposed matrix
    double J_T_det;  ///< Jacobian determinant
    Mat2 dxi_dx;     ///< Inverse Jacobian (dx/dxi)^T

    // Utility methods
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return std::abs(J_T_det) > kSingularJacobianTolerance;
    }

    // Separate, intentionally looser check: a Jacobian can be non-singular (is_valid() == true)
    // but still numerically ill-conditioned, e.g. for a very stretched/skewed element.
    [[nodiscard]] constexpr bool is_well_conditioned(double threshold = 1e-6) const noexcept {
        return std::abs(J_T_det) > threshold;
    }
};

/**
 * @brief Geometric mapping between reference and physical elements
 * Modern C++20 design with better const-correctness
 */
class GeometricMapping {
public:
    explicit GeometricMapping(std::shared_ptr<ReferenceElement> ref_element);

    // Delete copy, allow move
    GeometricMapping(const GeometricMapping&) = delete;
    GeometricMapping& operator=(const GeometricMapping&) = delete;
    GeometricMapping(GeometricMapping&&) noexcept = default;
    GeometricMapping& operator=(GeometricMapping&&) noexcept = default;

    /**
     * @brief Compute mapping data at reference point xi
     * @param vertices Physical element vertices
     * @param xi Reference coordinates
     * @return MappingData structure with computed values
     */
    [[nodiscard]] MappingData compute_mapping(const DView2& vertices, const Vec2& xi) const;

    /**
     * @brief Map reference coordinates to physical coordinates
     */
    [[nodiscard]] Vec2 map_to_physical(const DView2& vertices, const Vec2& xi) const;

    /**
     * @brief Find reference coordinates for given physical point (Newton iteration)
     * Returns std::optional - empty if convergence fails
     */
    [[nodiscard]] std::optional<Vec2> find_reference_coords(const DView2& vertices,
                                                            const Vec2& x_phys, double tol = 1e-10,
                                                            int max_iter = 20) const;

    [[nodiscard]] const std::shared_ptr<ReferenceElement>& get_ref_element() const noexcept {
        return ref_element_;
    }

    /**
     * @brief Compute shape functions and their derivatives (public for testing)
     */
    void compute_shape_functions(const Vec2& xi, DView1& N, DView2& dN_dxi) const;

private:
    std::shared_ptr<ReferenceElement> ref_element_;
};

}  // namespace dgfem
