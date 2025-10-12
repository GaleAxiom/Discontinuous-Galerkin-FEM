/**
 * @file space.cpp
 * @brief Implementation of DG space
 */

#include "dgfem/core/space.hpp"
#include "dgfem/basis/legendre.hpp"
#include "dgfem/basis/monomial.hpp"
#include "dgfem/basis/dubiner.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cmath>

namespace dgfem {

DGSpace::DGSpace(std::string_view element_type, int order)
    : order_(order), element_type_(element_type) {
    
    if (order < 1 || order > 10) {
        std::ostringstream oss;
        oss << "Element order must be between 1 and 10, got: " << order;
        throw std::invalid_argument(oss.str());
    }
    
    std::cout << "Initializing DG space: " << element_type_ << " elements, order " << order << std::endl;
    
    // Create components
    ref_element_ = create_ref_element(element_type_);
    basis_ = create_basis(element_type_, order);
    mapping_ = std::make_shared<GeometricMapping>(ref_element_);
    
    // Create quadrature rules
    if (element_type == "triangle") {
        // Rule must be exact for polynomials of degree 2*p for the mass matrix.
        // (Stiffness matrix needs 2*(p-1) but mass matrix needs 2*p)
        int quad_order = 2 * order;
        volume_quad_ = QuadratureFactory::dunavant_triangle(quad_order);
        
        // N points integrate up to degree 2N-1. We need to integrate up to 2*p.
        // 2N-1 >= 2p  => N >= p + 0.5 => N = p+1
        int face_points = order + 1;
        face_quad_ = QuadratureFactory::gauss_legendre_1d(face_points);
    } else if (element_type == "quad") {
        // For tensor product elements, we need p+1 points in each direction.
        int quad_points_1d = order + 1;
        volume_quad_ = QuadratureFactory::gauss_legendre_quad(quad_points_1d);
        
        // Face rule is the same as for triangles.
        int face_points = order + 1;
        face_quad_ = QuadratureFactory::gauss_legendre_1d(face_points);
    } else {
        std::ostringstream oss;
        oss << "Unknown element type: " << element_type_;
        throw std::invalid_argument(oss.str());
    }
    
    std::cout << "  Basis functions: " << basis_->get_n_basis() << std::endl;
    std::cout << "  Volume quad points: " << volume_quad_->size() << std::endl;
    std::cout << "  Face quad points: " << face_quad_->size() << std::endl;
    
    // Precompute basis values
    precompute_basis_values();
}

std::map<std::string, Eigen::MatrixXd> DGSpace::compute_element_data(
    const Eigen::MatrixXd& vertices,
    const std::vector<std::pair<int, int>>& face_neighbors) const {
    
    std::map<std::string, Eigen::MatrixXd> data;
    
    int n_vol_quad = volume_quad_->size();
    int n_basis = basis_->get_n_basis();
    
    // Storage for Jacobian determinants and transformed gradients
    Eigen::VectorXd J_det_vol(n_vol_quad);
    Eigen::MatrixXd dphi_dx_vol(n_vol_quad * n_basis, 2);
    
    // Compute at each volume quadrature point
    for (int q = 0; q < n_vol_quad; ++q) {
        Eigen::Vector2d xi = volume_quad_->points.row(q);
        
        // Compute mapping
        MappingData mapping_data = mapping_->compute_mapping(vertices, xi);
        J_det_vol[q] = mapping_data.J_T_det;
        
        // Transform basis gradients
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_ref = dphi_vol_[q].row(i);
            Eigen::Vector2d grad_phys = mapping_data.dxi_dx * grad_ref;
            dphi_dx_vol.row(q * n_basis + i) = grad_phys;
        }
    }
    
    data["vertices"] = vertices;
    data["J_det_vol"] = J_det_vol;
    data["dphi_dx_vol"] = dphi_dx_vol;
    
    return data;
}

std::map<std::string, Eigen::VectorXd> DGSpace::compute_face_data(
    const Eigen::MatrixXd& vertices,
    int face_id,
    const std::pair<int, int>& neighbor_info) const {
    
    std::map<std::string, Eigen::VectorXd> data;
    
    // Get face vertex indices
    auto [v1_idx, v2_idx] = ref_element_->edge_vertices(face_id);
    Eigen::Vector2d v1 = vertices.row(v1_idx);
    Eigen::Vector2d v2 = vertices.row(v2_idx);
    
    // Compute face properties
    Eigen::Vector2d tangent = v2 - v1;
    double edge_length = tangent.norm();
    // Compute normal: rotate tangent 90 degrees clockwise (right-hand rule for outward normal)
    // normal = (tangent_y, -tangent_x)
    Eigen::Vector2d normal(tangent[1], -tangent[0]);
    normal /= edge_length;
    
    // Ensure outward normal by checking against centroid
    Eigen::Vector2d centroid = vertices.colwise().mean();
    Eigen::Vector2d edge_midpoint = 0.5 * (v1 + v2);
    Eigen::Vector2d from_centroid_to_edge = edge_midpoint - centroid;
    
    // The outward normal should point AWAY from the centroid
    // So it should point opposite to the vector from edge to centroid
    // Which means it should have positive dot product with (edge - centroid)
    // If dot product is negative, the normal points inward, so flip it
    if (normal.dot(from_centroid_to_edge) < 0) {
        normal = -normal;
    }
    
    // Compute quadrature points on face
    int n_face_quad = face_quad_->size();
    Eigen::MatrixXd quad_points(n_face_quad, 2);
    int n_basis = basis_->get_n_basis();
    Eigen::VectorXd dphi_dx_face(n_face_quad * n_basis * 2);
    
    for (int q = 0; q < n_face_quad; ++q) {
        double s = face_quad_->points(q, 0);  // 1D quadrature point in [-1, 1]
        double t = 0.5 * (s + 1.0);  // Map to [0, 1]
        quad_points.row(q) = (1.0 - t) * v1 + t * v2;

        Eigen::Vector2d xi_face = map_face_quad_point(face_id, s);
        auto mapping_data = mapping_->compute_mapping(vertices, xi_face);
        Eigen::MatrixXd dphi_dxi = basis_->evaluate_gradient(xi_face);
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phys = mapping_data.dxi_dx * dphi_dxi.row(i).transpose();
            int row_offset = q * n_basis + i;
            dphi_dx_face[row_offset * 2] = grad_phys[0];
            dphi_dx_face[row_offset * 2 + 1] = grad_phys[1];
        }
    }
    
    // Store face data
    data["normal"] = normal;
    data["length"] = Eigen::VectorXd::Constant(1, edge_length);
    data["quad_points"] = Eigen::Map<Eigen::VectorXd>(quad_points.data(), quad_points.size());
    data["neighbor_elem"] = Eigen::VectorXd::Constant(1, neighbor_info.first);
    data["neighbor_face"] = Eigen::VectorXd::Constant(1, neighbor_info.second);
    data["is_boundary"] = Eigen::VectorXd::Constant(1, (neighbor_info.first < 0) ? 1.0 : 0.0);
    data["dphi_dx_face"] = dphi_dx_face;
    
    return data;
}

Eigen::VectorXi DGSpace::compute_face_permutation(
    int elem1_face, const Eigen::MatrixXd& elem1_vertices,
    int elem2_face, const Eigen::MatrixXd& elem2_vertices) const {
    
    // Get face vertices
    auto [v1_1, v1_2] = ref_element_->edge_vertices(elem1_face);
    auto [v2_1, v2_2] = ref_element_->edge_vertices(elem2_face);
    
    Eigen::Vector2d edge1_v1 = elem1_vertices.row(v1_1);
    Eigen::Vector2d edge1_v2 = elem1_vertices.row(v1_2);
    Eigen::Vector2d edge2_v1 = elem2_vertices.row(v2_1);
    Eigen::Vector2d edge2_v2 = elem2_vertices.row(v2_2);
    
    double tol = 1e-10;
    int n_quad = face_quad_->size();
    
    // Compute edge tangent vectors
    Eigen::Vector2d tangent1 = edge1_v2 - edge1_v1;
    Eigen::Vector2d tangent2 = edge2_v2 - edge2_v1;
    
    // Check if edges are oriented in opposite directions
    // First check geometric coincidence (for non-periodic faces)
    bool vertices_match_reversed = (edge1_v1 - edge2_v2).norm() < tol && 
                                    (edge1_v2 - edge2_v1).norm() < tol;
    bool vertices_match_same = (edge1_v1 - edge2_v1).norm() < tol && 
                                (edge1_v2 - edge2_v2).norm() < tol;
    
    bool reverse_orientation = false;
    
    if (vertices_match_reversed || vertices_match_same) {
        // Non-periodic case: use geometric vertex matching
        reverse_orientation = vertices_match_reversed;
    } else {
        // Periodic case: edges are geometrically far apart
        // Check if tangent vectors point in opposite directions
        // For periodic boundaries, tangents should be parallel or anti-parallel
        tangent1.normalize();
        tangent2.normalize();
        double dot_product = tangent1.dot(tangent2);
        
        // If dot product is negative, edges point in opposite directions
        reverse_orientation = (dot_product < 0.0);
    }
    
    if (reverse_orientation) {
        // Reverse order
        Eigen::VectorXi perm(n_quad);
        for (int i = 0; i < n_quad; ++i) {
            perm[i] = n_quad - 1 - i;
        }
        return perm;
    } else {
        // Same order
        return Eigen::VectorXi::LinSpaced(n_quad, 0, n_quad - 1);
    }
}

void DGSpace::precompute_basis_values() {
    int n_vol_quad = volume_quad_->size();
    int n_face_quad = face_quad_->size();
    int n_basis = basis_->get_n_basis();
    int n_faces = ref_element_->get_n_edges();
    
    // Volume basis values and gradients
    phi_vol_.resize(n_vol_quad, n_basis);
    dphi_vol_.resize(n_vol_quad);
    
    for (int q = 0; q < n_vol_quad; ++q) {
        Eigen::Vector2d xi = volume_quad_->points.row(q);
        phi_vol_.row(q) = basis_->evaluate(xi);
        dphi_vol_[q] = basis_->evaluate_gradient(xi);
    }
    
    // Face basis values
    phi_face_.resize(n_faces);
    for (int face = 0; face < n_faces; ++face) {
        phi_face_[face].resize(n_face_quad, n_basis);
        
        for (int q = 0; q < n_face_quad; ++q) {
            double s = face_quad_->points(q, 0);
            Eigen::Vector2d xi_face = map_face_quad_point(face, s);
            phi_face_[face].row(q) = basis_->evaluate(xi_face);
        }
    }
}

Eigen::Vector2d DGSpace::map_face_quad_point(int face_id, double s) const {
    if (element_type_ == "triangle") {
        double t = 0.5 * (s + 1.0);  // Map [-1,1] to [0,1]
        switch (face_id) {
            case 0: return Eigen::Vector2d(t, 0.0);        // Bottom edge
            case 1: return Eigen::Vector2d(1.0 - t, t);    // Diagonal edge
            case 2: return Eigen::Vector2d(0.0, 1.0 - t);  // Left edge
            default: throw std::out_of_range("Invalid face for triangle");
        }
    } else if (element_type_ == "quad") {
        switch (face_id) {
            case 0: return Eigen::Vector2d(s, -1.0);   // Bottom edge
            case 1: return Eigen::Vector2d(1.0, s);    // Right edge
            case 2: return Eigen::Vector2d(-s, 1.0);   // Top edge (note -s)
            case 3: return Eigen::Vector2d(-1.0, -s);  // Left edge (note -s)
            default: throw std::out_of_range("Invalid face for quad");
        }
    } else {
        throw std::invalid_argument("Unknown element type");
    }
}

std::vector<int> DGSpace::get_dof_to_vertex_map() const {
    // For a P1 space, the DoFs are located at the vertices.
    // This assumes the ordering of DoFs in the basis matches the
    // ordering of vertices in the reference element.
    if (order_ != 1) {
        throw std::runtime_error("get_dof_to_vertex_map is only implemented for order 1 elements.");
    }
    
    // Get number of dofs directly from the basis to avoid recursion
    int n_dofs = basis_->get_n_basis();
    std::vector<int> map(n_dofs);
    for (int i = 0; i < n_dofs; ++i) {
        map[i] = i;
    }
    return map;
}

std::shared_ptr<OrthogonalBasis> DGSpace::create_basis(std::string_view element_type, int order) const {
    if (element_type == "triangle") {
        // For triangles, use monomial basis (could also use Dubiner)
        return std::make_shared<MonomialBasisTriangle>(order);
    } else if (element_type == "quad") {
        return std::make_shared<LegendreBasis>(order);
    } else {
        std::ostringstream oss;
        oss << "Unknown element type: " << element_type;
        throw std::invalid_argument(oss.str());
    }
}

std::shared_ptr<ReferenceElement> DGSpace::create_ref_element(std::string_view element_type) const {
    if (element_type == "triangle") {
        return std::make_shared<ReferenceTriangle>();
    } else if (element_type == "quad") {
        return std::make_shared<ReferenceQuad>();
    } else {
        std::ostringstream oss;
        oss << "Unknown element type: " << element_type;
        throw std::invalid_argument(oss.str());
    }
}

} // namespace dgfem