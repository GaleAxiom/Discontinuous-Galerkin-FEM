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
    std::cout << "DEBUG [initialize_gmsh]: Initializing GMSH..." << std::endl;
    
    // Check if GMSH is already initialized (from a crashed previous test)
    bool was_initialized = gmsh::isInitialized();
    std::cout << "DEBUG [initialize_gmsh]: GMSH was_initialized = " << was_initialized << std::endl;
    
    if (was_initialized) {
        std::cout << "DEBUG [initialize_gmsh]: GMSH already initialized, clearing state..." << std::endl;
        gmsh::clear();  // Clear any leftover models
        std::cout << "DEBUG [initialize_gmsh]: GMSH state cleared" << std::endl;
    } else {
        std::cout << "DEBUG [initialize_gmsh]: Calling gmsh::initialize()..." << std::endl;
        gmsh::initialize();
        std::cout << "DEBUG [initialize_gmsh]: gmsh::initialize() completed" << std::endl;
    }
    
    // Enable verbose GMSH output to see where crashes occur
    std::cout << "DEBUG [initialize_gmsh]: Setting GMSH options..." << std::endl;
    std::cout.flush();  // Force flush before GMSH operations
    
    gmsh::option::setNumber("General.Terminal", 1);
    gmsh::option::setNumber("General.Verbosity", 5);  // Maximum verbosity (0-5)
    
    // Force unbuffered output so we see exactly where crashes occur
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    // Disable automatic file saving which might fail in CI
    gmsh::option::setNumber("Mesh.SaveAll", 0);
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    
    // Ensure GMSH doesn't try to use GUI or graphics
    gmsh::option::setNumber("General.GraphicsWidth", 0);
    gmsh::option::setNumber("General.GraphicsHeight", 0);
    
    // Disable multithreading which can cause issues in CI
    gmsh::option::setNumber("General.NumThreads", 1);
    gmsh::option::setNumber("Mesh.MaxNumThreads1D", 1);
    gmsh::option::setNumber("Mesh.MaxNumThreads2D", 1);
    gmsh::option::setNumber("Mesh.MaxNumThreads3D", 1);
    
    std::cout << "DEBUG [initialize_gmsh]: GMSH initialized with CI-safe options (single-threaded, verbosity=5)" << std::endl;
}

void MeshCreator::finalize_gmsh() {
    std::cout << "DEBUG [finalize_gmsh]: Finalizing GMSH..." << std::endl;
    gmsh::finalize();
    std::cout << "DEBUG [finalize_gmsh]: GMSH finalized" << std::endl;
}

std::shared_ptr<DGMesh> MeshCreator::create_rectangular_mesh(
    double dx, bool use_triangles, double xmin, double xmax, double ymin, double ymax) {
    
    std::cout << "DEBUG [create_rectangular_mesh]: Entry - domain=[" << xmin << "," << xmax 
              << "] x [" << ymin << "," << ymax << "], dx=" << dx 
              << ", use_triangles=" << use_triangles << std::endl;
    
    // Remove any existing models to prevent conflicts
    std::cout << "DEBUG [create_rectangular_mesh]: Clearing existing GMSH models..." << std::endl;
    std::vector<std::string> existing_models;
    gmsh::model::list(existing_models);
    std::cout << "DEBUG [create_rectangular_mesh]: Found " << existing_models.size() << " existing models" << std::endl;
    
    // Remove all models by clearing them, not by iterating (safer)
    if (!existing_models.empty()) {
        std::cout << "DEBUG [create_rectangular_mesh]: Clearing all GMSH models..." << std::endl;
        gmsh::clear();
        std::cout << "DEBUG [create_rectangular_mesh]: All models cleared" << std::endl;
    }
    
    // Create new model
    std::cout << "DEBUG [create_rectangular_mesh]: Adding GMSH model 'rectangular_mesh'" << std::endl;
    gmsh::model::add("rectangular_mesh");
    
    // Setup geometry
    std::cout << "DEBUG [create_rectangular_mesh]: Setting up rectangular geometry" << std::endl;
    setup_rectangular_geometry(xmin, xmax, ymin, ymax, dx);
    
    // Configure mesh generation
    std::cout << "DEBUG [create_rectangular_mesh]: Configuring mesh generation" << std::endl;
    configure_mesh_generation(use_triangles);
    
    // Generate mesh with error handling
    std::cout << "DEBUG [create_rectangular_mesh]: Generating 2D mesh..." << std::endl;
    std::cout.flush();  // Force output before potentially crashing operation
    std::cerr.flush();
    try {
        std::cout << "DEBUG [create_rectangular_mesh]: Calling gmsh::model::mesh::generate(2)..." << std::endl;
        std::cout.flush();
        gmsh::model::mesh::generate(2);
        std::cout << "DEBUG [create_rectangular_mesh]: gmsh::model::mesh::generate(2) returned" << std::endl;
        std::cout.flush();
        std::cout << "DEBUG [create_rectangular_mesh]: Mesh generation completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR [create_rectangular_mesh]: GMSH mesh generation failed: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "ERROR [create_rectangular_mesh]: GMSH mesh generation failed with unknown exception" << std::endl;
        throw std::runtime_error("GMSH mesh generation failed");
    }
    
    // Create DGMesh from current model
    std::cout << "DEBUG [create_rectangular_mesh]: Creating DGMesh from GMSH model" << std::endl;
    auto result = create_dg_mesh_from_gmsh();
    std::cout << "DEBUG [create_rectangular_mesh]: DGMesh created, returning" << std::endl;
    return result;
}

std::shared_ptr<DGMesh> MeshCreator::create_euler_mesh(
    double xmin, double xmax, double ymin, double ymax, double dx, bool use_triangles) {
    
    std::cout << "DEBUG [create_euler_mesh]: Entry - domain=[" << xmin << "," << xmax 
              << "] x [" << ymin << "," << ymax << "], dx=" << dx 
              << ", use_triangles=" << use_triangles << std::endl;
    
    // Remove any existing models to prevent conflicts
    std::cout << "DEBUG [create_euler_mesh]: Clearing existing GMSH models..." << std::endl;
    std::vector<std::string> existing_models;
    gmsh::model::list(existing_models);
    std::cout << "DEBUG [create_euler_mesh]: Found " << existing_models.size() << " existing models" << std::endl;
    
    // Remove all models by clearing them, not by iterating (safer)
    if (!existing_models.empty()) {
        std::cout << "DEBUG [create_euler_mesh]: Clearing all GMSH models..." << std::endl;
        gmsh::clear();
        std::cout << "DEBUG [create_euler_mesh]: All models cleared" << std::endl;
    }
    
    // Create new model
    std::cout << "DEBUG [create_euler_mesh]: Adding GMSH model 'euler_mesh'" << std::endl;
    gmsh::model::add("euler_mesh");
    
    // Setup geometry
    std::cout << "DEBUG [create_euler_mesh]: Setting up rectangular geometry" << std::endl;
    setup_rectangular_geometry(xmin, xmax, ymin, ymax, dx);
    
    // Configure mesh generation
    std::cout << "DEBUG [create_euler_mesh]: Configuring mesh generation" << std::endl;
    configure_mesh_generation(use_triangles);
    
    // Generate mesh with error handling
    std::cout << "DEBUG [create_euler_mesh]: Generating 2D mesh..." << std::endl;
    std::cout.flush();  // Force output before potentially crashing operation
    std::cerr.flush();
    try {
        std::cout << "DEBUG [create_euler_mesh]: Calling gmsh::model::mesh::generate(2)..." << std::endl;
        std::cout.flush();
        gmsh::model::mesh::generate(2);
        std::cout << "DEBUG [create_euler_mesh]: gmsh::model::mesh::generate(2) returned" << std::endl;
        std::cout.flush();
        std::cout << "DEBUG [create_euler_mesh]: Mesh generation completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR [create_euler_mesh]: GMSH mesh generation failed: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "ERROR [create_euler_mesh]: GMSH mesh generation failed with unknown exception" << std::endl;
        throw std::runtime_error("GMSH mesh generation failed");
    }
    
    // Create DGMesh from current model
    std::cout << "DEBUG [create_euler_mesh]: Creating DGMesh from GMSH model" << std::endl;
    auto result = create_dg_mesh_from_gmsh();
    std::cout << "DEBUG [create_euler_mesh]: DGMesh created, returning" << std::endl;
    return result;
}

std::shared_ptr<DGMesh> MeshCreator::create_dg_mesh_from_gmsh() {
    std::cout << "DEBUG [create_dg_mesh_from_gmsh]: Entry" << std::endl;
    
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi elements;
    Eigen::VectorXi element_tags;
    std::map<std::string, int> boundary_tags;
    std::map<int, std::vector<std::pair<int, int>>> boundary_edges;
    
    std::cout << "DEBUG [create_dg_mesh_from_gmsh]: Extracting mesh data from GMSH" << std::endl;
    extract_mesh_data(vertices, elements, element_tags, boundary_tags, boundary_edges);
    
    std::cout << "DEBUG [create_dg_mesh_from_gmsh]: Creating DGMesh object - vertices: " 
              << vertices.rows() << "x" << vertices.cols() 
              << ", elements: " << elements.rows() << "x" << elements.cols() << std::endl;
    
    auto mesh = std::make_shared<DGMesh>(vertices, elements, element_tags, boundary_tags, boundary_edges);
    std::cout << "DEBUG [create_dg_mesh_from_gmsh]: DGMesh object created successfully" << std::endl;
    return mesh;
}

void MeshCreator::setup_rectangular_geometry(
    double xmin, double xmax, double ymin, double ymax, double dx) {
    
    std::cout << "DEBUG [setup_rectangular_geometry]: Entry - bounds=[" << xmin << "," << xmax 
              << "] x [" << ymin << "," << ymax << "], dx=" << dx << std::endl;
    
    // Add points
    std::cout << "DEBUG [setup_rectangular_geometry]: Adding geometry points..." << std::endl;
    int p1 = gmsh::model::geo::addPoint(xmin, ymin, 0, dx);
    int p2 = gmsh::model::geo::addPoint(xmax, ymin, 0, dx);
    int p3 = gmsh::model::geo::addPoint(xmax, ymax, 0, dx);
    int p4 = gmsh::model::geo::addPoint(xmin, ymax, 0, dx);
    std::cout << "DEBUG [setup_rectangular_geometry]: Points added: p1=" << p1 << ", p2=" << p2 
              << ", p3=" << p3 << ", p4=" << p4 << std::endl;
    
    // Add lines
    std::cout << "DEBUG [setup_rectangular_geometry]: Adding lines..." << std::endl;
    int l1 = gmsh::model::geo::addLine(p1, p2);  // Bottom
    int l2 = gmsh::model::geo::addLine(p2, p3);  // Right
    int l3 = gmsh::model::geo::addLine(p3, p4);  // Top
    int l4 = gmsh::model::geo::addLine(p4, p1);  // Left
    std::cout << "DEBUG [setup_rectangular_geometry]: Lines added: l1=" << l1 << ", l2=" << l2 
              << ", l3=" << l3 << ", l4=" << l4 << std::endl;
    
    // Add curve loop and surface
    std::cout << "DEBUG [setup_rectangular_geometry]: Adding curve loop and surface..." << std::endl;
    int cl = gmsh::model::geo::addCurveLoop({l1, l2, l3, l4});
    int surf = gmsh::model::geo::addPlaneSurface({cl});
    std::cout << "DEBUG [setup_rectangular_geometry]: Curve loop=" << cl << ", surface=" << surf << std::endl;
    
    // Synchronize
    std::cout << "DEBUG [setup_rectangular_geometry]: Synchronizing geometry..." << std::endl;
    gmsh::model::geo::synchronize();
    std::cout << "DEBUG [setup_rectangular_geometry]: Geometry synchronized" << std::endl;
    
    // Add physical groups
    std::cout << "DEBUG [setup_rectangular_geometry]: Adding physical groups..." << std::endl;
    gmsh::model::addPhysicalGroup(2, {surf}, 1, "Domain");
    gmsh::model::addPhysicalGroup(1, {l1}, 2, "Bottom");
    gmsh::model::addPhysicalGroup(1, {l2}, 3, "Right");
    gmsh::model::addPhysicalGroup(1, {l3}, 4, "Top");
    gmsh::model::addPhysicalGroup(1, {l4}, 5, "Left");
    std::cout << "DEBUG [setup_rectangular_geometry]: Physical groups added" << std::endl;
    
    // Final synchronization
    std::cout << "DEBUG [setup_rectangular_geometry]: Final synchronization..." << std::endl;
    gmsh::model::geo::synchronize();
    std::cout << "DEBUG [setup_rectangular_geometry]: Exit" << std::endl;
}

void MeshCreator::configure_mesh_generation(bool use_triangles) {
    if (!use_triangles) {
        // Recombine triangles into quadrilaterals
        gmsh::model::mesh::setRecombine(2, 1);  // Surface tag 1
        
        // Set mesh algorithm for quads - use transfinite for structured mesh
        gmsh::option::setNumber("Mesh.RecombineAll", 1);
        gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 1);  // Blossom
        gmsh::option::setNumber("Mesh.Algorithm", 8);  // Frontal-Delaunay for quads
        std::cout << "Mesh configuration: QUADS" << std::endl;
    } else {
        std::cout << "Mesh configuration: TRIANGLES" << std::endl;
    }
}

void MeshCreator::extract_mesh_data(
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& elements, 
    Eigen::VectorXi& element_tags,
    std::map<std::string, int>& boundary_tags,
    std::map<int, std::vector<std::pair<int, int>>>& boundary_edges) {
    
    std::cout << "DEBUG [extract_mesh_data]: Entry" << std::endl;
    
    // Get nodes
    std::vector<std::size_t> node_tags;
    std::vector<double> coords;
    std::vector<double> parametric_coords;
    
    std::cout << "DEBUG [extract_mesh_data]: Getting nodes from GMSH" << std::endl;
    gmsh::model::mesh::getNodes(node_tags, coords, parametric_coords);
    std::cout << "DEBUG [extract_mesh_data]: Retrieved " << node_tags.size() << " nodes" << std::endl;
    
    // Convert nodes to Eigen format
    int n_nodes = node_tags.size();
    vertices.resize(n_nodes, 2);
    std::map<std::size_t, int> node_map;
    
    std::cout << "DEBUG [extract_mesh_data]: Converting nodes to Eigen format" << std::endl;
    for (int i = 0; i < n_nodes; ++i) {
        node_map[node_tags[i]] = i;
        vertices(i, 0) = coords[3*i];
        vertices(i, 1) = coords[3*i + 1];
    }
    std::cout << "DEBUG [extract_mesh_data]: Node conversion complete" << std::endl;
    
    // Get 2D elements
    std::vector<int> elem_types;
    std::vector<std::vector<std::size_t>> elem_tags_vec;
    std::vector<std::vector<std::size_t>> node_tags_vec;
    
    std::cout << "DEBUG [extract_mesh_data]: Getting 2D elements from GMSH" << std::endl;
    gmsh::model::mesh::getElements(elem_types, elem_tags_vec, node_tags_vec, 2);
    std::cout << "DEBUG [extract_mesh_data]: Retrieved " << elem_types.size() << " element types" << std::endl;
    
    if (elem_types.empty()) {
        std::cerr << "ERROR [extract_mesh_data]: No 2D elements found in GMSH model" << std::endl;
        throw std::runtime_error("No 2D elements found in GMSH model");
    }
    
    int elem_type = elem_types[0];
    std::cout << "DEBUG [extract_mesh_data]: Element type = " << elem_type << std::endl;
    int n_nodes_per_elem = (elem_type == 2) ? 3 : (elem_type == 3) ? 4 : 0;
    std::cout << "DEBUG [extract_mesh_data]: Nodes per element = " << n_nodes_per_elem << std::endl;
    
    if (n_nodes_per_elem == 0) {
        std::cerr << "ERROR [extract_mesh_data]: Unsupported element type: " << elem_type << std::endl;
        throw std::runtime_error("Unsupported element type: " + std::to_string(elem_type));
    }
    
    // Convert elements to Eigen format
    int n_elements = elem_tags_vec[0].size();
    std::cout << "DEBUG [extract_mesh_data]: Number of elements = " << n_elements << std::endl;
    std::cout << "DEBUG [extract_mesh_data]: Resizing elements matrix to " << n_elements << "x" << n_nodes_per_elem << std::endl;
    elements.resize(n_elements, n_nodes_per_elem);
    element_tags.resize(n_elements);
    
    std::cout << "DEBUG [extract_mesh_data]: Converting elements to Eigen format" << std::endl;
    for (int i = 0; i < n_elements; ++i) {
        element_tags[i] = elem_tags_vec[0][i];
        for (int j = 0; j < n_nodes_per_elem; ++j) {
            std::size_t global_node = node_tags_vec[0][i * n_nodes_per_elem + j];
            elements(i, j) = node_map[global_node];
        }
    }
    std::cout << "DEBUG [extract_mesh_data]: Element conversion complete" << std::endl;
    
    // Get physical groups for boundary identification
    std::vector<std::pair<int, int>> physical_groups;
    std::cout << "DEBUG [extract_mesh_data]: Getting physical groups" << std::endl;
    gmsh::model::getPhysicalGroups(physical_groups);
    std::cout << "DEBUG [extract_mesh_data]: Found " << physical_groups.size() << " physical groups" << std::endl;
    
    for (auto& pg : physical_groups) {
        int dim = pg.first;
        int tag = pg.second;
        
        std::cout << "DEBUG [extract_mesh_data]: Processing physical group - dim=" << dim << ", tag=" << tag << std::endl;
        
        if (dim == 1) {  // Boundary physical groups
            std::string name;
            gmsh::model::getPhysicalName(dim, tag, name);
            std::cout << "DEBUG [extract_mesh_data]: Boundary group name = '" << name << "'" << std::endl;
            boundary_tags[name] = tag;

            std::vector<int> entities;
            gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);
            std::cout << "DEBUG [extract_mesh_data]: Found " << entities.size() << " entities for this group" << std::endl;

            for (int entity : entities) {
                std::vector<int> edge_types;
                std::vector<std::vector<std::size_t>> edge_tags_vec;
                std::vector<std::vector<std::size_t>> edge_node_tags_vec;
                gmsh::model::mesh::getElements(edge_types, edge_tags_vec, edge_node_tags_vec, 1, entity);

                if (!edge_types.empty()) {
                    const auto& edge_node_tags = edge_node_tags_vec[0];
                    std::cout << "DEBUG [extract_mesh_data]: Processing " << edge_node_tags.size()/2 << " edges for entity " << entity << std::endl;
                    for (size_t i = 0; i < edge_node_tags.size() / 2; ++i) {
                        int v1 = node_map.at(edge_node_tags[2 * i]);
                        int v2 = node_map.at(edge_node_tags[2 * i + 1]);
                        boundary_edges[tag].push_back({v1, v2});
                    }
                }
            }
        }
    }
    
    std::cout << "DEBUG [extract_mesh_data]: Boundary processing complete" << std::endl;
    
    // Print mesh statistics
    std::cout << "Generated mesh with " << n_nodes << " nodes and " << n_elements << " elements." << std::endl;
    if (elem_type == 2) {
        std::cout << "Element type: 3-node triangles" << std::endl;
    } else if (elem_type == 3) {
        std::cout << "Element type: 4-node quadrilaterals" << std::endl;
    }
    std::cout << "DEBUG [extract_mesh_data]: Exit" << std::endl;
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