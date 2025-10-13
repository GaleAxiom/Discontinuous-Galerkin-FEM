/**
 * @file mesh_creation.cpp
 * @brief Implementation of GMSH mesh creation utilities
 */

#include "dgfem/utils/mesh_creation.hpp"
#include "dgfem/core/mesh.hpp"
#include <gmsh.h>
#include <iostream>
#include <stdexcept>
#include <optional>

namespace dgfem {

void MeshCreator::initialize_gmsh() {
    // Initialize GMSH if not already initialized
    // Note: gmsh::isInitialized() is not available in all GMSH versions
    // We use a try-catch approach instead
    try {
        gmsh::initialize();
    } catch (...) {
        // Already initialized, clear any leftover models
        gmsh::clear();
    }
    
    // Disable GMSH terminal output (keep errors only)
    gmsh::option::setNumber("General.Terminal", 1);
    gmsh::option::setNumber("General.Verbosity", 2);  // Errors only
    
    // Disable automatic file saving
    gmsh::option::setNumber("Mesh.SaveAll", 0);
    
    // Ensure GMSH doesn't try to use GUI or graphics
    gmsh::option::setNumber("General.GraphicsWidth", 0);
    gmsh::option::setNumber("General.GraphicsHeight", 0);
    
    // Disable multithreading for CI stability
    gmsh::option::setNumber("General.NumThreads", 1);
}

void MeshCreator::finalize_gmsh() {
    gmsh::finalize();
}

std::shared_ptr<DGMesh> MeshCreator::create_rectangular_mesh(
    double dx, bool use_triangles, double xmin, double xmax, double ymin, double ymax) {
    
    // Remove any existing models to prevent conflicts
    std::vector<std::string> existing_models;
    gmsh::model::list(existing_models);
    
    if (!existing_models.empty()) {
        gmsh::clear();
    }
    
    // Create new model
    gmsh::model::add("rectangular_mesh");
    
    // Setup geometry (transfinite only for quads)
    setup_rectangular_geometry(xmin, xmax, ymin, ymax, dx, use_triangles);
    
    // Configure mesh generation
    configure_mesh_generation(use_triangles);
    
    // Generate mesh
    gmsh::model::mesh::generate(2);
    
    // Create DGMesh from current model
    return create_dg_mesh_from_gmsh();
}

std::shared_ptr<DGMesh> MeshCreator::create_euler_mesh(
    double xmin, double xmax, double ymin, double ymax, double dx, bool use_triangles) {
    
    // Remove any existing models to prevent conflicts
    std::vector<std::string> existing_models;
    gmsh::model::list(existing_models);
    
    if (!existing_models.empty()) {
        gmsh::clear();
    }
    
    // Create new model
    gmsh::model::add("euler_mesh");
    
    // Setup geometry (transfinite only for quads)
    setup_rectangular_geometry(xmin, xmax, ymin, ymax, dx, use_triangles);
    
    // Configure mesh generation
    configure_mesh_generation(use_triangles);
    
    // Generate mesh
    gmsh::model::mesh::generate(2);
    
    // Create DGMesh from current model
    return create_dg_mesh_from_gmsh();
}

std::shared_ptr<DGMesh> MeshCreator::create_dg_mesh_from_gmsh() {
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi elements;
    Eigen::VectorXi element_tags;
    std::map<std::string, int> boundary_tags;
    std::map<int, std::vector<std::pair<int, int>>> boundary_edges;
    
    extract_mesh_data(vertices, elements, element_tags, boundary_tags, boundary_edges);
    
    return std::make_shared<DGMesh>(vertices, elements, element_tags, boundary_tags, boundary_edges);
}

void MeshCreator::setup_rectangular_geometry(
    double xmin, double xmax, double ymin, double ymax, double dx, bool use_triangles) {
    
    // Calculate number of divisions
    int nx = static_cast<int>((xmax - xmin) / dx);
    int ny = static_cast<int>((ymax - ymin) / dx);
    
    // Add points
    int p1 = gmsh::model::geo::addPoint(xmin, ymin, 0, dx);
    int p2 = gmsh::model::geo::addPoint(xmax, ymin, 0, dx);
    int p3 = gmsh::model::geo::addPoint(xmax, ymax, 0, dx);
    int p4 = gmsh::model::geo::addPoint(xmin, ymax, 0, dx);
    
    // Add lines
    int l1 = gmsh::model::geo::addLine(p1, p2);  // Bottom
    int l2 = gmsh::model::geo::addLine(p2, p3);  // Right
    int l3 = gmsh::model::geo::addLine(p3, p4);  // Top
    int l4 = gmsh::model::geo::addLine(p4, p1);  // Left
    
    // Add curve loop and surface
    int cl = gmsh::model::geo::addCurveLoop({l1, l2, l3, l4});
    int surf = gmsh::model::geo::addPlaneSurface({cl});
    
    // Use transfinite meshing for quads (structured mesh)
    // This avoids the crash-prone Frontal-Delaunay algorithm
    if (!use_triangles) {
        gmsh::model::geo::mesh::setTransfiniteCurve(l1, nx + 1);
        gmsh::model::geo::mesh::setTransfiniteCurve(l2, ny + 1);
        gmsh::model::geo::mesh::setTransfiniteCurve(l3, nx + 1);
        gmsh::model::geo::mesh::setTransfiniteCurve(l4, ny + 1);
        gmsh::model::geo::mesh::setTransfiniteSurface(surf);
        gmsh::model::geo::mesh::setRecombine(2, surf);
    }
    
    // Synchronize
    gmsh::model::geo::synchronize();
    
    // Add physical groups (without names for compatibility with older GMSH versions)
    gmsh::model::addPhysicalGroup(2, {surf}, 1);
    gmsh::model::addPhysicalGroup(1, {l1}, 2);
    gmsh::model::addPhysicalGroup(1, {l2}, 3);
    gmsh::model::addPhysicalGroup(1, {l3}, 4);
    gmsh::model::addPhysicalGroup(1, {l4}, 5);
    
    // Set physical group names separately for better compatibility
    gmsh::model::setPhysicalName(2, 1, "Domain");
    gmsh::model::setPhysicalName(1, 2, "Bottom");
    gmsh::model::setPhysicalName(1, 3, "Right");
    gmsh::model::setPhysicalName(1, 4, "Top");
    gmsh::model::setPhysicalName(1, 5, "Left");
    
    // Final synchronization
    gmsh::model::geo::synchronize();
}

void MeshCreator::configure_mesh_generation(bool use_triangles) {
    if (!use_triangles) {
        // Transfinite meshing is already configured in setup_rectangular_geometry
        gmsh::option::setNumber("Mesh.RecombineAll", 1);
    }
}

void MeshCreator::extract_mesh_data(
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& elements, 
    Eigen::VectorXi& element_tags,
    std::map<std::string, int>& boundary_tags,
    std::map<int, std::vector<std::pair<int, int>>>& boundary_edges) {
    
    // Get nodes
    std::vector<std::size_t> node_tags;
    std::vector<double> coords;
    std::vector<double> parametric_coords;
    
    gmsh::model::mesh::getNodes(node_tags, coords, parametric_coords);
    
    // Convert nodes to Eigen format
    int n_nodes = node_tags.size();
    vertices.resize(n_nodes, 2);
    std::map<std::size_t, int> node_map;
    
    for (int i = 0; i < n_nodes; ++i) {
        node_map[node_tags[i]] = i;
        vertices(i, 0) = coords[3*i];
        vertices(i, 1) = coords[3*i + 1];
    }
    
    // Get 2D elements
    std::vector<int> elem_types;
    std::vector<std::vector<std::size_t>> elem_tags_vec;
    std::vector<std::vector<std::size_t>> node_tags_vec;
    
    gmsh::model::mesh::getElements(elem_types, elem_tags_vec, node_tags_vec, 2);
    
    if (elem_types.empty()) {
        throw std::runtime_error("No 2D elements found in GMSH model");
    }
    
    int elem_type = elem_types[0];
    int n_nodes_per_elem = (elem_type == 2) ? 3 : (elem_type == 3) ? 4 : 0;
    
    if (n_nodes_per_elem == 0) {
        throw std::runtime_error("Unsupported element type: " + std::to_string(elem_type));
    }
    
    // Convert elements to Eigen format
    int n_elements = elem_tags_vec[0].size();
    elements.resize(n_elements, n_nodes_per_elem);
    element_tags.resize(n_elements);
    
    for (int i = 0; i < n_elements; ++i) {
        element_tags[i] = elem_tags_vec[0][i];
        for (int j = 0; j < n_nodes_per_elem; ++j) {
            std::size_t global_node = node_tags_vec[0][i * n_nodes_per_elem + j];
            elements(i, j) = node_map[global_node];
        }
    }
    
    // Get physical groups for boundary identification
    std::vector<std::pair<int, int>> physical_groups;
    gmsh::model::getPhysicalGroups(physical_groups);
    
    for (auto& pg : physical_groups) {
        int dim = pg.first;
        int tag = pg.second;
        
        if (dim == 1) {  // Boundary physical groups
            std::string name;
            gmsh::model::getPhysicalName(dim, tag, name);
            boundary_tags[name] = tag;

            std::vector<int> entities;
            gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);

            for (int entity : entities) {
                std::vector<int> edge_types;
                std::vector<std::vector<std::size_t>> edge_tags_vec;
                std::vector<std::vector<std::size_t>> edge_node_tags_vec;
                gmsh::model::mesh::getElements(edge_types, edge_tags_vec, edge_node_tags_vec, 1, entity);

                if (!edge_types.empty()) {
                    const auto& edge_node_tags = edge_node_tags_vec[0];
                    for (size_t i = 0; i < edge_node_tags.size() / 2; ++i) {
                        int v1 = node_map.at(edge_node_tags[2 * i]);
                        int v2 = node_map.at(edge_node_tags[2 * i + 1]);
                        boundary_edges[tag].push_back({v1, v2});
                    }
                }
            }
        }
    }
    
    // Print mesh statistics
    std::cout << "Generated mesh with " << n_nodes << " nodes and " << n_elements << " elements." << std::endl;
    if (elem_type == 2) {
        std::cout << "Element type: 3-node triangles" << std::endl;
    } else if (elem_type == 3) {
        std::cout << "Element type: 4-node quadrilaterals" << std::endl;
    }
}

std::optional<Eigen::Vector2d> find_reference_coords(
    const Eigen::Vector2d& x_phys,
    const Eigen::MatrixXd& vertices,
    std::string_view element_type,
    double tol, int max_iter) {
    
    // Newton iteration to find reference coordinates
    Eigen::Vector2d xi = Eigen::Vector2d::Zero();  // Initial guess
    
    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::Vector2d x_mapped;
        Eigen::Matrix2d J;
        
        if (element_type == "triangle") {
            // Triangle mapping: x = v0*(1-xi-eta) + v1*xi + v2*eta
            x_mapped = vertices.row(0) * (1.0 - xi[0] - xi[1]) + 
                      vertices.row(1) * xi[0] + 
                      vertices.row(2) * xi[1];
            
            J.col(0) = vertices.row(1) - vertices.row(0);
            J.col(1) = vertices.row(2) - vertices.row(0);
            
        } else if (element_type == "quad") {
            // Bilinear quad mapping
            double xi_val = xi[0], eta_val = xi[1];
            Eigen::Vector4d N;
            N[0] = 0.25 * (1.0 - xi_val) * (1.0 - eta_val);
            N[1] = 0.25 * (1.0 + xi_val) * (1.0 - eta_val);
            N[2] = 0.25 * (1.0 + xi_val) * (1.0 + eta_val);
            N[3] = 0.25 * (1.0 - xi_val) * (1.0 + eta_val);
            
            x_mapped = vertices.transpose() * N;
            
            Eigen::Matrix<double, 4, 2> dN_dxi;
            dN_dxi(0, 0) = -0.25 * (1.0 - eta_val); dN_dxi(0, 1) = -0.25 * (1.0 - xi_val);
            dN_dxi(1, 0) =  0.25 * (1.0 - eta_val); dN_dxi(1, 1) = -0.25 * (1.0 + xi_val);
            dN_dxi(2, 0) =  0.25 * (1.0 + eta_val); dN_dxi(2, 1) =  0.25 * (1.0 + xi_val);
            dN_dxi(3, 0) = -0.25 * (1.0 + eta_val); dN_dxi(3, 1) =  0.25 * (1.0 - xi_val);
            
            J = vertices.transpose() * dN_dxi;
        } else {
            throw std::invalid_argument("Unknown element type: " + std::string(element_type));
        }
        
        Eigen::Vector2d residual = x_mapped - x_phys;
        
        if (residual.norm() < tol) {
            return xi;
        }
        
        // Newton update
        xi -= J.inverse() * residual;
        
        // Project to reference element bounds
        if (element_type == "triangle") {
            xi[0] = std::max(0.0, std::min(1.0, xi[0]));
            xi[1] = std::max(0.0, std::min(1.0 - xi[0], xi[1]));
        } else if (element_type == "quad") {
            xi[0] = std::max(-1.0, std::min(1.0, xi[0]));
            xi[1] = std::max(-1.0, std::min(1.0, xi[1]));
        }
    }
    
    throw std::runtime_error("Newton iteration failed to converge");
}

} // namespace dgfem