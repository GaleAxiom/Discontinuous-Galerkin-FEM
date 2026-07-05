/**
 * @file elements.hpp
 * @brief Reference elements for DGFEM with modern C++20 features
 *
 * This header contains the reference element classes for triangles and quadrilaterals.
 */

#pragma once

#include "dgfem/kokkos_math.hpp"

#include <optional>

#include <array>
#include <utility>
#include <vector>

namespace dgfem {

/**
 * @brief Base class for reference elements with modern C++20 design
 */
class ReferenceElement {
public:
    explicit ReferenceElement(const DView2& vertices);
    virtual ~ReferenceElement() = default;

    // Delete copy, allow move
    ReferenceElement(const ReferenceElement&) = delete;
    ReferenceElement& operator=(const ReferenceElement&) = delete;
    ReferenceElement(ReferenceElement&&) noexcept = default;
    ReferenceElement& operator=(ReferenceElement&&) noexcept = default;

    [[nodiscard]] virtual bool contains_point(const Vec2& xi) const = 0;
    [[nodiscard]] virtual std::pair<int, int> edge_vertices(int edge_id) const = 0;

    /**
     * @brief Compute shape functions and their derivatives at a reference point
     * @param xi Reference coordinates
     * @param N Output: Shape function values
     * @param dN_dxi Output: Shape function derivatives w.r.t. reference coordinates
     */
    virtual void compute_shape_functions(const Vec2& xi, DView1& N, DView2& dN_dxi) const = 0;

    /**
     * @brief Project a point onto valid reference element bounds
     * @param xi Reference coordinates (will be modified in place)
     */
    virtual void project_to_bounds(Vec2& xi) const = 0;

    // Modern getters with [[nodiscard]]
    [[nodiscard]] const DView2& get_vertices() const noexcept { return vertices_; }
    [[nodiscard]] constexpr int get_dimension() const noexcept { return dim_; }
    [[nodiscard]] constexpr int get_n_vertices() const noexcept { return n_vertices_; }
    [[nodiscard]] constexpr int get_n_edges() const noexcept { return n_edges_; }

protected:
    DView2 vertices_;
    int dim_;
    int n_vertices_;
    int n_edges_;
};

/**
 * @brief Standard reference triangle with vertices at (0,0), (1,0), (0,1)
 * Uses compile-time constants where possible
 */
class ReferenceTriangle : public ReferenceElement {
public:
    ReferenceTriangle();

    [[nodiscard]] bool contains_point(const Vec2& xi) const override;
    [[nodiscard]] std::pair<int, int> edge_vertices(int edge_id) const override;

    void compute_shape_functions(const Vec2& xi, DView1& N, DView2& dN_dxi) const override;

    void project_to_bounds(Vec2& xi) const override;

    // Compile-time constants
    static constexpr int N_VERTICES = 3;
    static constexpr int N_EDGES = 3;
    static constexpr int DIMENSION = 2;
};

/**
 * @brief Standard reference quadrilateral [-1,1] x [-1,1]
 * Uses compile-time constants where possible
 */
class ReferenceQuad : public ReferenceElement {
public:
    ReferenceQuad();

    [[nodiscard]] bool contains_point(const Vec2& xi) const override;
    [[nodiscard]] std::pair<int, int> edge_vertices(int edge_id) const override;

    void compute_shape_functions(const Vec2& xi, DView1& N, DView2& dN_dxi) const override;

    void project_to_bounds(Vec2& xi) const override;

    // Compile-time constants
    static constexpr int N_VERTICES = 4;
    static constexpr int N_EDGES = 4;
    static constexpr int DIMENSION = 2;
};

}  // namespace dgfem
