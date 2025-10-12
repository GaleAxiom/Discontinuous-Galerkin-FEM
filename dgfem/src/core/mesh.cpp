/**
 * @file mesh.cpp
 * @brief Implementation of DG mesh
 */

#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/boundary/conditions.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <set>

namespace dgfem {

DGMesh::DGMesh(const Eigen::MatrixXd& vertices, 
               const Eigen::MatrixXi& elements,
               const Eigen::VectorXi& element_tags,
               const std::map<std::string, int>& boundary_tags,
               const std::map<int, std::vector<std::pair<int, int>>>& boundary_edges)
    : vertices_(vertices), elements_(elements), 
      element_tags_(element_tags), boundary_tags_(boundary_tags),
      n_elements_(elements.rows()) {
    
    std::cout << "DEBUG [DGMesh::DGMesh]: Constructor entry - vertices: " 
              << vertices.rows() << "x" << vertices.cols() 
              << ", elements: " << elements.rows() << "x" << elements.cols() << std::endl;
    
    // Determine element type
    int n_nodes_per_elem = elements.cols();
    std::cout << "DEBUG [DGMesh::DGMesh]: Nodes per element = " << n_nodes_per_elem << std::endl;
    
    if (n_nodes_per_elem == 3) {
        element_type_ = "triangle";
        n_faces_per_elem_ = 3;
    } else if (n_nodes_per_elem == 4) {
        element_type_ = "quad";
        n_faces_per_elem_ = 4;
    } else {
        std::cerr << "ERROR [DGMesh::DGMesh]: Unknown element type with " << n_nodes_per_elem << " vertices" << std::endl;
        throw std::invalid_argument("Unknown element type with " + 
                                  std::to_string(n_nodes_per_elem) + " vertices");
    }
    std::cout << "DEBUG [DGMesh::DGMesh]: Element type = " << element_type_ << ", faces per element = " << n_faces_per_elem_ << std::endl;
    
    // Build connectivity
    std::cout << "DEBUG [DGMesh::DGMesh]: Building face connectivity..." << std::endl;
    build_face_connectivity();
    std::cout << "DEBUG [DGMesh::DGMesh]: Face connectivity built" << std::endl;
    
    std::cout << "DEBUG [DGMesh::DGMesh]: Identifying boundary faces..." << std::endl;
    identify_boundary_faces(boundary_edges);
    std::cout << "DEBUG [DGMesh::DGMesh]: Boundary faces identified, count = " << boundary_faces_.size() << std::endl;

    // Build node-to-element mapping
    std::cout << "DEBUG [DGMesh::DGMesh]: Building node-to-element mapping..." << std::endl;
    for (int i = 0; i < n_elements_; ++i) {
        for (int j = 0; j < elements_.cols(); ++j) {
            int global_node_idx = elements_(i, j);
            node_to_elements_[global_node_idx].push_back({i, j});
        }
    }
    std::cout << "DEBUG [DGMesh::DGMesh]: Node-to-element mapping built" << std::endl;
    
    std::cout << "Created DGMesh with " << n_elements_ << " " << element_type_ 
              << " elements and " << boundary_faces_.size() << " boundary faces" << std::endl;
    std::cout << "DEBUG [DGMesh::DGMesh]: Constructor exit" << std::endl;
}

void DGMesh::initialize_dg_space(std::shared_ptr<DGSpace> dg_space, int n_variables) {
    dg_space_ = dg_space;
    solution_ = std::make_shared<DGSolution>(n_elements_, dg_space->get_basis()->get_n_basis(), n_variables);
    
    // Compute element and face data
    element_data_.resize(n_elements_);
    face_data_.resize(n_elements_);
    
    for (int elem_id = 0; elem_id < n_elements_; ++elem_id) {
        // Get element vertices
        Eigen::MatrixXd elem_vertices(elements_.cols(), 2);
        for (int i = 0; i < elements_.cols(); ++i) {
            elem_vertices.row(i) = vertices_.row(elements_(elem_id, i));
        }
        
        // Get face neighbors
        std::vector<std::pair<int, int>> neighbors = get_element_neighbors(elem_id);
        
        // Compute element data
        element_data_[elem_id] = dg_space_->compute_element_data(elem_vertices, neighbors);
        
        // Compute face data
        face_data_[elem_id].resize(n_faces_per_elem_);
        for (int face_id = 0; face_id < n_faces_per_elem_; ++face_id) {
            std::pair<int, int> neighbor = (face_id < neighbors.size()) ? neighbors[face_id] : std::make_pair(-1, -1);
            face_data_[elem_id][face_id] = dg_space_->compute_face_data(elem_vertices, face_id, neighbor);
        }
    }
    
    std::cout << "Initialized DG space with order " << dg_space->get_order() 
              << ", " << dg_space->get_basis()->get_n_basis() << " basis functions per element" << std::endl;
    
    // Build precomputed face connectivity for efficient assembly
    build_precomputed_faces();
}

std::vector<std::pair<int, int>> DGMesh::get_element_neighbors(int elem_id) const {
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    
    std::vector<std::pair<int, int>> neighbors(n_faces_per_elem_);
    for (int face = 0; face < n_faces_per_elem_; ++face) {
        neighbors[face] = {face_neighbors_(elem_id, face * 2), face_neighbors_(elem_id, face * 2 + 1)};
    }
    return neighbors;
}

const std::map<std::string, Eigen::MatrixXd>& DGMesh::get_element_data(int elem_id) const {
    if (!dg_space_) {
        throw std::runtime_error("DG space not initialized");
    }
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    return element_data_[elem_id];
}

const std::map<std::string, Eigen::VectorXd>& DGMesh::get_face_data(int elem_id, int face_id) const {
    if (!dg_space_) {
        throw std::runtime_error("DG space not initialized");
    }
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    if (face_id < 0 || face_id >= n_faces_per_elem_) {
        throw std::out_of_range("Face index out of range");
    }
    return face_data_[elem_id][face_id];
}

const std::map<std::string, Eigen::MatrixXd>& DGMesh::get_element_face_data(int elem_id, int face_id) const {
    if (!dg_space_) {
        throw std::runtime_error("DG space not initialized");
    }
    if (elem_id < 0 || elem_id >= n_elements_) {
        throw std::out_of_range("Element index out of range");
    }
    if (face_id < 0 || face_id >= n_faces_per_elem_) {
        throw std::out_of_range("Face index out of range");
    }
    
    // Lazily compute face_data_matrix_ if not already done
    if (face_data_matrix_.empty()) {
        const_cast<DGMesh*>(this)->face_data_matrix_.resize(n_elements_);
        for (int e = 0; e < n_elements_; ++e) {
            const_cast<DGMesh*>(this)->face_data_matrix_[e].resize(n_faces_per_elem_);
        }
    }
    
    // Convert face_data_ (VectorXd) to face_data_matrix_ (MatrixXd) format if needed
    if (face_data_matrix_[elem_id][face_id].empty()) {
        const auto& face_vec_data = face_data_[elem_id][face_id];
        auto& face_mat_data = const_cast<DGMesh*>(this)->face_data_matrix_[elem_id][face_id];
        
        for (const auto& pair : face_vec_data) {
            const std::string& key = pair.first;
            const Eigen::VectorXd& vec = pair.second;
            
            if (key == "dphi_dx_face") {
                int n_face_quad = dg_space_->get_face_quad()->size();
                int n_basis = dg_space_->get_basis()->get_n_basis();
                if (vec.size() != n_face_quad * n_basis * 2) {
                    throw std::runtime_error("Unexpected size for dphi_dx_face data");
                }
                face_mat_data[key] = Eigen::Map<const Eigen::MatrixXd>(vec.data(), n_face_quad * n_basis, 2);
            } else {
                // Convert vector to single-column matrix
                face_mat_data[key] = vec.reshaped(vec.size(), 1);
            }
        }
        
        // Add phi (basis function values at face quadrature points) from DGSpace
        const auto& phi_face_all = dg_space_->get_face_basis_values();
        if (face_id < static_cast<int>(phi_face_all.size())) {
            face_mat_data["phi"] = phi_face_all[face_id];
        }
        
        // Add quadrature weights as a column vector in MatrixXd format
        const auto& face_quad = dg_space_->get_face_quad();
        int n_quad = face_quad->weights.size();
        Eigen::MatrixXd weights_mat(n_quad, 1);
        weights_mat.col(0) = face_quad->weights;
        face_mat_data["weights"] = weights_mat;
    }
    
    return face_data_matrix_[elem_id][face_id];
}

void DGMesh::set_boundary_condition(std::string_view tag_name, 
                                   std::shared_ptr<BoundaryCondition> bc) {
    std::string tag_str(tag_name);
    if (boundary_tags_.find(tag_str) == boundary_tags_.end()) {
        throw std::invalid_argument("Unknown boundary tag: " + tag_str);
    }
    boundary_conditions_[tag_str] = std::move(bc);
}

std::shared_ptr<BoundaryCondition> DGMesh::get_boundary_condition(int elem_id, int face_id) const {
    auto key = std::make_pair(elem_id, face_id);
    auto it = boundary_face_tags_.find(key);
    if (it == boundary_face_tags_.end()) {
        return nullptr;  // Not a boundary face
    }
    
    int tag = it->second;
    for (const auto& bt : boundary_tags_) {
        if (bt.second == tag) {
            auto bc_it = boundary_conditions_.find(bt.first);
            return (bc_it != boundary_conditions_.end()) ? bc_it->second : nullptr;
        }
    }
    return nullptr;
}

void DGMesh::set_boundary_condition_euler(std::string_view tag_name, 
                                          std::shared_ptr<BoundaryConditionEuler> bc) {
    std::string tag_str(tag_name);
    if (boundary_tags_.find(tag_str) == boundary_tags_.end()) {
        throw std::invalid_argument("Unknown boundary tag: " + tag_str);
    }
    boundary_conditions_euler_[tag_str] = bc;
    
    // Update precomputed boundary face data
    for (auto& face_data : boundary_face_data_) {
        if (face_data.bc_tag == tag_str) {
            face_data.bc_euler = bc;
        }
    }
}

std::shared_ptr<BoundaryConditionEuler> DGMesh::get_boundary_condition_euler(int elem_id, int face_id) const {
    std::string tag_name = get_face_boundary_tag(elem_id, face_id);
    if (tag_name.empty()) {
        return nullptr;  // Not a boundary face
    }
    return get_boundary_condition_euler(tag_name);
}

std::shared_ptr<BoundaryConditionEuler> DGMesh::get_boundary_condition_euler(std::string_view tag_name) const {
    std::string tag_str(tag_name);
    auto it = boundary_conditions_euler_.find(tag_str);
    if (it != boundary_conditions_euler_.end()) {
        return it->second;
    }
    return nullptr;
}

std::string DGMesh::get_face_boundary_tag(int elem_id, int face_id) const {
    auto key = std::make_pair(elem_id, face_id);
    
    // Check if already cached
    auto it = boundary_face_tag_names_.find(key);
    if (it != boundary_face_tag_names_.end()) {
        return it->second;
    }
    
    // Look up the tag ID
    auto tag_it = boundary_face_tags_.find(key);
    if (tag_it == boundary_face_tags_.end()) {
        return "";  // Not a boundary face
    }
    
    int tag_id = tag_it->second;
    
    // Find the tag name
    for (const auto& bt : boundary_tags_) {
        if (bt.second == tag_id) {
            // Cache the result
            const_cast<DGMesh*>(this)->boundary_face_tag_names_[key] = bt.first;
            return bt.first;
        }
    }
    
    return "";  // Tag ID not found in boundary_tags_
}

bool DGMesh::is_boundary_face(int elem_id, int face_id) const noexcept {
    return face_neighbors_(elem_id, face_id * 2) == -1;
}

void DGMesh::build_face_connectivity() {
    std::cout << "DEBUG [build_face_connectivity]: Entry - n_elements=" << n_elements_ 
              << ", n_faces_per_elem=" << n_faces_per_elem_ << std::endl;
    
    std::cout << "DEBUG [build_face_connectivity]: Initializing face_neighbors matrix to " 
              << n_elements_ << "x" << (n_faces_per_elem_ * 2) << std::endl;
    face_neighbors_ = Eigen::MatrixXi::Constant(n_elements_, n_faces_per_elem_ * 2, -1);
    std::cout << "DEBUG [build_face_connectivity]: face_neighbors matrix initialized" << std::endl;
    
    std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> face_map;
    
    std::cout << "DEBUG [build_face_connectivity]: Building face signature map..." << std::endl;
    // Build face signature map
    for (int elem_id = 0; elem_id < n_elements_; ++elem_id) {
        if (elem_id % 500 == 0) {
            std::cout << "DEBUG [build_face_connectivity]: Processing element " << elem_id << "/" << n_elements_ << std::endl;
        }
        for (int face_id = 0; face_id < n_faces_per_elem_; ++face_id) {
            // Get face vertices
            int v1, v2;
            if (element_type_ == "triangle") {
                v1 = elements_(elem_id, face_id);
                v2 = elements_(elem_id, (face_id + 1) % 3);
            } else { // quad
                v1 = elements_(elem_id, face_id);
                v2 = elements_(elem_id, (face_id + 1) % 4);
            }
            
            // Create face signature (sorted)
            std::pair<int, int> face_sig = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);
            face_map[face_sig].push_back({elem_id, face_id});
            face_to_vertices_[{elem_id, face_id}] = {v1, v2};
        }
    }
    std::cout << "DEBUG [build_face_connectivity]: Face signature map built, unique faces = " << face_map.size() << std::endl;
    
    std::cout << "DEBUG [build_face_connectivity]: Setting up connectivity..." << std::endl;
    // Set up connectivity
    int interior_faces = 0;
    for (const auto& entry : face_map) {
        const auto& face_list = entry.second;
        if (face_list.size() == 2) {
            // Interior face
            interior_faces++;
            int elem1 = face_list[0].first, face1 = face_list[0].second;
            int elem2 = face_list[1].first, face2 = face_list[1].second;
            
            face_neighbors_(elem1, face1 * 2) = elem2;
            face_neighbors_(elem1, face1 * 2 + 1) = face2;
            face_neighbors_(elem2, face2 * 2) = elem1;
            face_neighbors_(elem2, face2 * 2 + 1) = face1;
        }
        // Boundary faces remain -1
    }
    std::cout << "DEBUG [build_face_connectivity]: Connectivity setup complete, interior_faces = " << interior_faces << std::endl;
    std::cout << "DEBUG [build_face_connectivity]: Exit" << std::endl;
}

void DGMesh::identify_boundary_faces(const std::map<int, std::vector<std::pair<int, int>>>& boundary_edges) {
    std::cout << "DEBUG [identify_boundary_faces]: Entry - boundary_edges groups = " << boundary_edges.size() << std::endl;
    
    boundary_faces_.clear();
    boundary_face_tags_.clear();

    // Create a map from a sorted vertex pair (face signature) to the boundary tag
    std::map<std::pair<int, int>, int> sig_to_tag;
    std::cout << "DEBUG [identify_boundary_faces]: Building signature-to-tag map..." << std::endl;
    for (const auto& pair : boundary_edges) {
        int tag = pair.first;
        const auto& edges = pair.second;
        std::cout << "DEBUG [identify_boundary_faces]: Processing tag " << tag << " with " << edges.size() << " edges" << std::endl;
        for (const auto& edge : edges) {
            std::pair<int, int> sig = (edge.first < edge.second) ? edge : std::make_pair(edge.second, edge.first);
            sig_to_tag[sig] = tag;
        }
    }
    std::cout << "DEBUG [identify_boundary_faces]: sig_to_tag map built, size = " << sig_to_tag.size() << std::endl;

    // Identify all boundary faces and assign tags
    std::cout << "DEBUG [identify_boundary_faces]: Identifying boundary faces..." << std::endl;
    for (int elem_id = 0; elem_id < n_elements_; ++elem_id) {
        if (elem_id % 500 == 0 && elem_id > 0) {
            std::cout << "DEBUG [identify_boundary_faces]: Checked " << elem_id << "/" << n_elements_ 
                      << " elements, found " << boundary_faces_.size() << " boundary faces so far" << std::endl;
        }
        for (int face_id = 0; face_id < n_faces_per_elem_; ++face_id) {
            if (face_neighbors_(elem_id, face_id * 2) == -1) {
                boundary_faces_.push_back({elem_id, face_id});

                // Get face vertices to create a signature
                const auto& vertices = face_to_vertices_.at({elem_id, face_id});
                int v1 = vertices[0];
                int v2 = vertices[1];
                std::pair<int, int> face_sig = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);

                int tag = 0; // Default tag if not found
                if (sig_to_tag.count(face_sig)) {
                    tag = sig_to_tag.at(face_sig);
                }
                boundary_face_tags_[{elem_id, face_id}] = tag;
            }
        }
    }
    std::cout << "DEBUG [identify_boundary_faces]: Boundary face identification complete, total = " << boundary_faces_.size() << std::endl;
    std::cout << "DEBUG [identify_boundary_faces]: Exit" << std::endl;
}

void DGMesh::build_precomputed_faces() {
    if (!dg_space_) {
        throw std::runtime_error("DG space must be initialized before building precomputed faces");
    }

    std::cout << "Building precomputed face connectivity..." << std::endl;
    
    interior_faces_.clear();
    boundary_face_data_.clear();
    
    // Track processed interior faces to avoid duplicates
    std::set<std::pair<int, int>> processed_interior_faces;
    
    // Process all faces
    for (int elem_L = 0; elem_L < n_elements_; ++elem_L) {
        auto neighbors = get_element_neighbors(elem_L);
        
        // Get element L vertices once
        Eigen::MatrixXd vertices_L(elements_.cols(), 2);
        for (int i = 0; i < elements_.cols(); ++i) {
            vertices_L.row(i) = vertices_.row(elements_(elem_L, i));
        }
        
        for (int face_L = 0; face_L < static_cast<int>(neighbors.size()); ++face_L) {
            int elem_R = neighbors[face_L].first;
            int face_R = neighbors[face_L].second;
            
            if (elem_R >= 0) {
                // Interior face - only process once
                std::pair<int, int> face_sig = (elem_L < elem_R) ? 
                    std::make_pair(elem_L, elem_R) : std::make_pair(elem_R, elem_L);
                
                if (processed_interior_faces.find(face_sig) == processed_interior_faces.end()) {
                    processed_interior_faces.insert(face_sig);
                    
                    FaceConnectivity fc;
                    fc.elem_L = elem_L;
                    fc.elem_R = elem_R;
                    fc.face_L = face_L;
                    fc.face_R = face_R;
                    fc.is_boundary = false;
                    fc.vertices_L = vertices_L;
                    
                    // Get element R vertices
                    fc.vertices_R.resize(elements_.cols(), 2);
                    for (int i = 0; i < elements_.cols(); ++i) {
                        fc.vertices_R.row(i) = vertices_.row(elements_(elem_R, i));
                    }
                    
                    // Compute and cache permutation
                    fc.permutation = dg_space_->compute_face_permutation(
                        face_L, fc.vertices_L, face_R, fc.vertices_R);
                    
                    interior_faces_.push_back(fc);
                }
            } else {
                // Boundary face
                FaceConnectivity fc;
                fc.elem_L = elem_L;
                fc.face_L = face_L;
                fc.is_boundary = true;
                fc.vertices_L = vertices_L;
                
                // Get boundary tag
                fc.bc_tag = get_face_boundary_tag(elem_L, face_L);
                
                // Note: BC pointer will be set when BCs are assigned
                fc.bc_euler = nullptr;
                
                boundary_face_data_.push_back(fc);
            }
        }
    }
    
    std::cout << "  Precomputed " << interior_faces_.size() << " interior faces and " 
              << boundary_face_data_.size() << " boundary faces" << std::endl;
}

void DGMesh::set_periodic_boundaries(std::string_view tag_name_1, std::string_view tag_name_2) {
    std::string tag_str_1(tag_name_1);
    std::string tag_str_2(tag_name_2);
    
    // Verify both tags exist
    if (boundary_tags_.find(tag_str_1) == boundary_tags_.end()) {
        throw std::invalid_argument("Unknown boundary tag: " + tag_str_1);
    }
    if (boundary_tags_.find(tag_str_2) == boundary_tags_.end()) {
        throw std::invalid_argument("Unknown boundary tag: " + tag_str_2);
    }
    
    int tag_id_1 = boundary_tags_.at(tag_str_1);
    int tag_id_2 = boundary_tags_.at(tag_str_2);
    
    // Collect faces for each boundary
    std::vector<std::pair<int, int>> faces_1;
    std::vector<std::pair<int, int>> faces_2;
    
    for (const auto& [key, tag] : boundary_face_tags_) {
        if (tag == tag_id_1) {
            faces_1.push_back(key);
        } else if (tag == tag_id_2) {
            faces_2.push_back(key);
        }
    }
    
    if (faces_1.empty() || faces_2.empty()) {
        throw std::runtime_error("Cannot set periodic boundaries: one or both boundaries have no faces");
    }
    
    std::cout << "Setting up periodic boundaries between '" << tag_str_1 
              << "' (" << faces_1.size() << " faces) and '" << tag_str_2 
              << "' (" << faces_2.size() << " faces) ..." << std::endl;
    
    // Match faces based on their positions
    // For a rectangular domain, we match faces with same y-coordinate for left/right
    // and same x-coordinate for top/bottom
    
    for (const auto& face_1 : faces_1) {
        int elem_1 = face_1.first;
        int local_face_1 = face_1.second;
        
        // Get face center for face_1
        const auto& verts_1 = face_to_vertices_.at(face_1);
        Eigen::Vector2d center_1 = 0.5 * (vertices_.row(verts_1[0]) + vertices_.row(verts_1[1]));
        
        // Find matching face on boundary 2
        double min_dist = std::numeric_limits<double>::max();
        std::pair<int, int> best_match = {-1, -1};
        
        for (const auto& face_2 : faces_2) {
            int elem_2 = face_2.first;
            int local_face_2 = face_2.second;
            
            const auto& verts_2 = face_to_vertices_.at(face_2);
            Eigen::Vector2d center_2 = 0.5 * (vertices_.row(verts_2[0]) + vertices_.row(verts_2[1]));
            
            // For periodic boundaries, we match faces with similar transverse coordinate
            // Left-Right: match y-coordinates
            // Top-Bottom: match x-coordinates
            double dist;
            if (tag_str_1 == "Left" || tag_str_1 == "Right" || 
                tag_str_2 == "Left" || tag_str_2 == "Right") {
                dist = std::abs(center_1.y() - center_2.y());
            } else {
                dist = std::abs(center_1.x() - center_2.x());
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                best_match = face_2;
            }
        }
        
        if (best_match.first >= 0) {
            // Set up periodic connection
            periodic_face_map_[face_1] = best_match;
            periodic_face_map_[best_match] = face_1;
            
            // Update face neighbors to point to periodic partner
            int elem_2 = best_match.first;
            int local_face_2 = best_match.second;
            
            face_neighbors_(elem_1, local_face_1 * 2) = elem_2;
            face_neighbors_(elem_1, local_face_1 * 2 + 1) = local_face_2;
            face_neighbors_(elem_2, local_face_2 * 2) = elem_1;
            face_neighbors_(elem_2, local_face_2 * 2 + 1) = local_face_1;
        }
    }
    
    std::cout << "  Set up " << periodic_face_map_.size() / 2 << " periodic face pairs" << std::endl;
    
    // Mark periodic boundary conditions
    Eigen::Vector4d zero_vec = Eigen::Vector4d::Zero();
    auto periodic_bc = std::make_shared<BoundaryConditionEuler>(
        BCTypeEuler::PERIODIC, zero_vec);
    boundary_conditions_euler_[tag_str_1] = periodic_bc;
    boundary_conditions_euler_[tag_str_2] = periodic_bc;
    
    // Rebuild precomputed faces if DG space is initialized
    if (dg_space_) {
        build_precomputed_faces();
    }
}

} // namespace dgfem