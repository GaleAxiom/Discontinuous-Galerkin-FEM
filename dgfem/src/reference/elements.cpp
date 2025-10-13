/**
 * @file elements.cpp
 * @brief Implementation of reference elements
 */

#include "dgfem/reference/elements.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace dgfem {

ReferenceElement::ReferenceElement(const Eigen::MatrixXd& vertices)
    : vertices_(vertices), dim_(vertices.cols()), n_vertices_(vertices.rows()) {}

ReferenceTriangle::ReferenceTriangle()
    : ReferenceElement((Eigen::MatrixXd(3, 2) << 
                        0.0, 0.0,
                        1.0, 0.0,
                        0.0, 1.0).finished()) {
    n_edges_ = 3;
}

bool ReferenceTriangle::contains_point(const Eigen::Vector2d& xi) const {
    return xi[0] >= 0 && xi[1] >= 0 && (xi[0] + xi[1]) <= 1;
}

std::pair<int, int> ReferenceTriangle::edge_vertices(int edge_id) const {
    switch (edge_id) {
        case 0: return {0, 1};
        case 1: return {1, 2};
        case 2: return {2, 0};
        default: 
            throw std::out_of_range("Invalid edge_id for triangle");
    }
}

void ReferenceTriangle::compute_shape_functions(const Eigen::Vector2d& xi,
                                                Eigen::VectorXd& N,
                                                Eigen::MatrixXd& dN_dxi) const {
    // Triangle shape functions: N1 = 1-xi-eta, N2 = xi, N3 = eta
    N.resize(3);
    dN_dxi.resize(3, 2);
    
    N[0] = 1.0 - xi[0] - xi[1];
    N[1] = xi[0];
    N[2] = xi[1];
    
    dN_dxi(0, 0) = -1.0; dN_dxi(0, 1) = -1.0;
    dN_dxi(1, 0) =  1.0; dN_dxi(1, 1) =  0.0;
    dN_dxi(2, 0) =  0.0; dN_dxi(2, 1) =  1.0;
}

void ReferenceTriangle::project_to_bounds(Eigen::Vector2d& xi) const {
    xi[0] = std::max(0.0, std::min(1.0, xi[0]));
    xi[1] = std::max(0.0, std::min(1.0 - xi[0], xi[1]));
}

ReferenceQuad::ReferenceQuad()
    : ReferenceElement((Eigen::MatrixXd(4, 2) << 
                        -1.0, -1.0,
                         1.0, -1.0,
                         1.0,  1.0,
                        -1.0,  1.0).finished()) {
    n_edges_ = 4;
}

bool ReferenceQuad::contains_point(const Eigen::Vector2d& xi) const {
    return std::abs(xi[0]) <= 1.0 && std::abs(xi[1]) <= 1.0;
}

std::pair<int, int> ReferenceQuad::edge_vertices(int edge_id) const {
    std::cout << "DEBUG [ReferenceQuad::edge_vertices]: edge_id=" << edge_id << std::endl;
    
    if (edge_id < 0 || edge_id > 3) {
        std::cerr << "ERROR [ReferenceQuad::edge_vertices]: Invalid edge_id " << edge_id 
                  << " (must be 0-3)" << std::endl;
        throw std::out_of_range("Invalid edge_id for quad");
    }
    
    std::pair<int, int> result;
    switch (edge_id) {
        case 0: result = {0, 1}; break;
        case 1: result = {1, 2}; break;
        case 2: result = {2, 3}; break;
        case 3: result = {3, 0}; break;
        default: 
            throw std::out_of_range("Invalid edge_id for quad");
    }
    
    std::cout << "DEBUG [ReferenceQuad::edge_vertices]: Returning (" << result.first << ", " << result.second << ")" << std::endl;
    return result;
}

void ReferenceQuad::compute_shape_functions(const Eigen::Vector2d& xi,
                                            Eigen::VectorXd& N,
                                            Eigen::MatrixXd& dN_dxi) const {
    // Bilinear shape functions for quad
    N.resize(4);
    dN_dxi.resize(4, 2);
    
    N[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
    N[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
    N[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
    N[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);
    
    dN_dxi(0, 0) = -0.25 * (1.0 - xi[1]); dN_dxi(0, 1) = -0.25 * (1.0 - xi[0]);
    dN_dxi(1, 0) =  0.25 * (1.0 - xi[1]); dN_dxi(1, 1) = -0.25 * (1.0 + xi[0]);
    dN_dxi(2, 0) =  0.25 * (1.0 + xi[1]); dN_dxi(2, 1) =  0.25 * (1.0 + xi[0]);
    dN_dxi(3, 0) = -0.25 * (1.0 + xi[1]); dN_dxi(3, 1) =  0.25 * (1.0 - xi[0]);
}

void ReferenceQuad::project_to_bounds(Eigen::Vector2d& xi) const {
    xi[0] = std::max(-1.0, std::min(1.0, xi[0]));
    xi[1] = std::max(-1.0, std::min(1.0, xi[1]));
}

} // namespace dgfem
