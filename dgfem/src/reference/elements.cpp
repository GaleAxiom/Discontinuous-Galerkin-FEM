/**
 * @file elements.cpp
 * @brief Implementation of reference elements
 */

#include "dgfem/reference/elements.hpp"

#include <cmath>

#include <iostream>
#include <stdexcept>

namespace dgfem {

namespace {
DView2 make_vertices(std::initializer_list<std::array<double, 2>> rows) {
    DView2 v("ref_vertices", rows.size(), 2);
    int i = 0;
    for (const auto& row : rows) {
        v(i, 0) = row[0];
        v(i, 1) = row[1];
        ++i;
    }
    return v;
}
}  // namespace

ReferenceElement::ReferenceElement(const DView2& vertices)
    : vertices_(vertices), dim_(vertices.extent(1)), n_vertices_(vertices.extent(0)) {}

ReferenceTriangle::ReferenceTriangle()
    : ReferenceElement(make_vertices({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}})) {
    n_edges_ = 3;
}

bool ReferenceTriangle::contains_point(const Vec2& xi) const {
    return xi[0] >= 0 && xi[1] >= 0 && (xi[0] + xi[1]) <= 1;
}

std::pair<int, int> ReferenceTriangle::edge_vertices(int edge_id) const {
    switch (edge_id) {
    case 0:
        return {0, 1};
    case 1:
        return {1, 2};
    case 2:
        return {2, 0};
    default:
        throw std::out_of_range("Invalid edge_id for triangle");
    }
}

void ReferenceTriangle::compute_shape_functions(const Vec2& xi, DView1& N, DView2& dN_dxi) const {
    // Triangle shape functions: N1 = 1-xi-eta, N2 = xi, N3 = eta
    N = DView1("N", 3);
    dN_dxi = DView2("dN_dxi", 3, 2);

    N[0] = 1.0 - xi[0] - xi[1];
    N[1] = xi[0];
    N[2] = xi[1];

    dN_dxi(0, 0) = -1.0;
    dN_dxi(0, 1) = -1.0;
    dN_dxi(1, 0) = 1.0;
    dN_dxi(1, 1) = 0.0;
    dN_dxi(2, 0) = 0.0;
    dN_dxi(2, 1) = 1.0;
}

void ReferenceTriangle::project_to_bounds(Vec2& xi) const {
    xi[0] = std::max(0.0, std::min(1.0, xi[0]));
    xi[1] = std::max(0.0, std::min(1.0 - xi[0], xi[1]));
}

ReferenceQuad::ReferenceQuad()
    : ReferenceElement(make_vertices({{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}})) {
    n_edges_ = 4;
}

bool ReferenceQuad::contains_point(const Vec2& xi) const {
    return std::abs(xi[0]) <= 1.0 && std::abs(xi[1]) <= 1.0;
}

std::pair<int, int> ReferenceQuad::edge_vertices(int edge_id) const {
    if (edge_id < 0 || edge_id > 3) {
        throw std::out_of_range("Invalid edge_id for quad");
    }

    switch (edge_id) {
    case 0:
        return {0, 1};
    case 1:
        return {1, 2};
    case 2:
        return {2, 3};
    case 3:
        return {3, 0};
    default:
        throw std::out_of_range("Invalid edge_id for quad");
    }
}

void ReferenceQuad::compute_shape_functions(const Vec2& xi, DView1& N, DView2& dN_dxi) const {
    // Bilinear shape functions for quad
    N = DView1("N", 4);
    dN_dxi = DView2("dN_dxi", 4, 2);

    N[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
    N[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
    N[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
    N[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);

    dN_dxi(0, 0) = -0.25 * (1.0 - xi[1]);
    dN_dxi(0, 1) = -0.25 * (1.0 - xi[0]);
    dN_dxi(1, 0) = 0.25 * (1.0 - xi[1]);
    dN_dxi(1, 1) = -0.25 * (1.0 + xi[0]);
    dN_dxi(2, 0) = 0.25 * (1.0 + xi[1]);
    dN_dxi(2, 1) = 0.25 * (1.0 + xi[0]);
    dN_dxi(3, 0) = -0.25 * (1.0 + xi[1]);
    dN_dxi(3, 1) = 0.25 * (1.0 - xi[0]);
}

void ReferenceQuad::project_to_bounds(Vec2& xi) const {
    xi[0] = std::max(-1.0, std::min(1.0, xi[0]));
    xi[1] = std::max(-1.0, std::min(1.0, xi[1]));
}

}  // namespace dgfem
