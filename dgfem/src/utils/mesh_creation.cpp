/**
 * @file mesh_creation.cpp
 * @brief Implementation of GMSH mesh creation utilities
 */

#include "dgfem/utils/mesh_creation.hpp"

#include "dgfem/core/mesh.hpp"

#include <optional>

#include <iostream>
#include <stdexcept>

#include <gmsh.h>

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

std::shared_ptr<DGMesh> MeshCreator::create_rectangular_mesh(double dx, bool use_triangles,
                                                             double xmin, double xmax, double ymin,
                                                             double ymax) {
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

std::shared_ptr<DGMesh> MeshCreator::create_euler_mesh(double xmin, double xmax, double ymin,
                                                       double ymax, double dx, bool use_triangles) {
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

std::shared_ptr<DGMesh> MeshCreator::create_cylinder_channel_mesh(double length, double height,
                                                                  double radius, const Vec2& center,
                                                                  double dx_channel,
                                                                  double dx_cylinder,
                                                                  bool use_triangles) {
    // Remove any existing models to prevent conflicts
    std::vector<std::string> existing_models;
    gmsh::model::list(existing_models);

    if (!existing_models.empty()) {
        gmsh::clear();
    }

    gmsh::model::add("cylinder_channel_mesh");

    double xmin = 0.0;
    double xmax = length;
    double ymin = 0.0;
    double ymax = height;

    // Channel corner points
    int p1 = gmsh::model::geo::addPoint(xmin, ymin, 0.0, dx_channel);
    int p2 = gmsh::model::geo::addPoint(xmax, ymin, 0.0, dx_channel);
    int p3 = gmsh::model::geo::addPoint(xmax, ymax, 0.0, dx_channel);
    int p4 = gmsh::model::geo::addPoint(xmin, ymax, 0.0, dx_channel);

    // Cylinder center and points on the circumference (four quadrature points)
    int pc = gmsh::model::geo::addPoint(center[0], center[1], 0.0, dx_cylinder);
    int p5 = gmsh::model::geo::addPoint(center[0] + radius, center[1], 0.0, dx_cylinder);
    int p6 = gmsh::model::geo::addPoint(center[0], center[1] + radius, 0.0, dx_cylinder);
    int p7 = gmsh::model::geo::addPoint(center[0] - radius, center[1], 0.0, dx_cylinder);
    int p8 = gmsh::model::geo::addPoint(center[0], center[1] - radius, 0.0, dx_cylinder);

    // Channel boundary lines (counter-clockwise)
    int l1 = gmsh::model::geo::addLine(p1, p2);  // Lower wall
    int l2 = gmsh::model::geo::addLine(p2, p3);  // Outlet
    int l3 = gmsh::model::geo::addLine(p3, p4);  // Upper wall
    int l4 = gmsh::model::geo::addLine(p4, p1);  // Inlet

    // Cylinder boundary (counter-clockwise arcs)
    int c1 = gmsh::model::geo::addCircleArc(p5, pc, p6);
    int c2 = gmsh::model::geo::addCircleArc(p6, pc, p7);
    int c3 = gmsh::model::geo::addCircleArc(p7, pc, p8);
    int c4 = gmsh::model::geo::addCircleArc(p8, pc, p5);

    int outer_loop = gmsh::model::geo::addCurveLoop({l1, l2, l3, l4});
    int inner_loop = gmsh::model::geo::addCurveLoop({c1, c2, c3, c4});
    int surface = gmsh::model::geo::addPlaneSurface({outer_loop, inner_loop});

    // Synchronize the CAD kernel with the model
    gmsh::model::geo::synchronize();

    // Tag physical groups for boundary condition mapping
    gmsh::model::addPhysicalGroup(2, {surface}, 1);
    gmsh::model::setPhysicalName(2, 1, "Domain");

    gmsh::model::addPhysicalGroup(1, {l4}, 2);
    gmsh::model::setPhysicalName(1, 2, "Inlet");

    gmsh::model::addPhysicalGroup(1, {l2}, 3);
    gmsh::model::setPhysicalName(1, 3, "Outlet");

    gmsh::model::addPhysicalGroup(1, {l1}, 4);
    gmsh::model::setPhysicalName(1, 4, "LowerWall");

    gmsh::model::addPhysicalGroup(1, {l3}, 5);
    gmsh::model::setPhysicalName(1, 5, "UpperWall");

    gmsh::model::addPhysicalGroup(1, {c1, c2, c3, c4}, 6);
    gmsh::model::setPhysicalName(1, 6, "Cylinder");

    // Configure mesh generation strategy
    configure_mesh_generation(use_triangles);

    // Refine the mesh near the cylinder using a distance-based background field
    int distance_field = gmsh::model::mesh::field::add("Distance");
    std::vector<double> cylinder_curve_ids{static_cast<double>(c1), static_cast<double>(c2),
                                           static_cast<double>(c3), static_cast<double>(c4)};
    gmsh::model::mesh::field::setNumbers(distance_field, "CurvesList", cylinder_curve_ids);
    gmsh::model::mesh::field::setNumber(distance_field, "NumPointsPerCurve", 50);

    int threshold_field = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(threshold_field, "InField", distance_field);
    gmsh::model::mesh::field::setNumber(threshold_field, "SizeMin", dx_cylinder);
    gmsh::model::mesh::field::setNumber(threshold_field, "SizeMax", dx_channel);
    gmsh::model::mesh::field::setNumber(threshold_field, "DistMin", radius * 0.4);
    gmsh::model::mesh::field::setNumber(threshold_field, "DistMax", radius * 5.0);
    gmsh::model::mesh::field::setAsBackgroundMesh(threshold_field);

    gmsh::model::mesh::generate(2);

    // Clear mesh fields to avoid side-effects in subsequent mesh generations
    gmsh::model::mesh::field::remove(distance_field);
    gmsh::model::mesh::field::remove(threshold_field);

    return create_dg_mesh_from_gmsh();
}

std::shared_ptr<DGMesh> MeshCreator::create_dg_mesh_from_gmsh() {
    DView2 vertices;
    IView2 elements;
    IView1 element_tags;
    std::map<std::string, int> boundary_tags;
    std::map<int, std::vector<std::pair<int, int>>> boundary_edges;

    extract_mesh_data(vertices, elements, element_tags, boundary_tags, boundary_edges);

    return std::make_shared<DGMesh>(vertices, elements, element_tags, boundary_tags,
                                    boundary_edges);
}

void MeshCreator::setup_rectangular_geometry(double xmin, double xmax, double ymin, double ymax,
                                             double dx, bool use_triangles) {
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
    DView2& vertices, IView2& elements, IView1& element_tags,
    std::map<std::string, int>& boundary_tags,
    std::map<int, std::vector<std::pair<int, int>>>& boundary_edges) {
    // Get nodes
    std::vector<std::size_t> node_tags;
    std::vector<double> coords;
    std::vector<double> parametric_coords;

    gmsh::model::mesh::getNodes(node_tags, coords, parametric_coords);

    // Convert nodes to view format
    int n_nodes = static_cast<int>(node_tags.size());
    vertices = DView2("vertices", n_nodes, 2);
    std::map<std::size_t, int> node_map;

    for (int i = 0; i < n_nodes; ++i) {
        node_map[node_tags[i]] = i;
        vertices(i, 0) = coords[3 * i];
        vertices(i, 1) = coords[3 * i + 1];
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

    // Convert elements to view format
    int n_elements = static_cast<int>(elem_tags_vec[0].size());
    elements = IView2("elements", n_elements, n_nodes_per_elem);
    element_tags = IView1("element_tags", n_elements);

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
                gmsh::model::mesh::getElements(edge_types, edge_tags_vec, edge_node_tags_vec, 1,
                                               entity);

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
    std::cout << "Generated mesh with " << n_nodes << " nodes and " << n_elements << " elements."
              << std::endl;
    if (elem_type == 2) {
        std::cout << "Element type: 3-node triangles" << std::endl;
    } else if (elem_type == 3) {
        std::cout << "Element type: 4-node quadrilaterals" << std::endl;
    }
}

std::optional<Vec2> find_reference_coords(const Vec2& x_phys, const DView2& vertices,
                                          std::string_view element_type, double tol, int max_iter) {
    // Newton iteration to find reference coordinates
    Vec2 xi = Vec2{0.0, 0.0};  // Initial guess

    for (int iter = 0; iter < max_iter; ++iter) {
        Vec2 x_mapped;
        Mat2 J{};

        if (element_type == "triangle") {
            // Triangle mapping: x = v0*(1-xi-eta) + v1*xi + v2*eta
            Vec2 v0 = row2(vertices, 0);
            Vec2 v1 = row2(vertices, 1);
            Vec2 v2 = row2(vertices, 2);
            x_mapped = v0 * (1.0 - xi[0] - xi[1]) + v1 * xi[0] + v2 * xi[1];

            Vec2 col0 = v1 - v0;
            Vec2 col1 = v2 - v0;
            J = Mat2{col0[0], col1[0], col0[1], col1[1]};

        } else if (element_type == "quad") {
            // Bilinear quad mapping
            double xi_val = xi[0], eta_val = xi[1];
            Vec4 N;
            N[0] = 0.25 * (1.0 - xi_val) * (1.0 - eta_val);
            N[1] = 0.25 * (1.0 + xi_val) * (1.0 - eta_val);
            N[2] = 0.25 * (1.0 + xi_val) * (1.0 + eta_val);
            N[3] = 0.25 * (1.0 - xi_val) * (1.0 + eta_val);

            x_mapped = Vec2{0.0, 0.0};
            for (int i = 0; i < 4; ++i) {
                x_mapped = x_mapped + row2(vertices, i) * N[i];
            }

            double dN_dxi[4][2];
            dN_dxi[0][0] = -0.25 * (1.0 - eta_val);
            dN_dxi[0][1] = -0.25 * (1.0 - xi_val);
            dN_dxi[1][0] = 0.25 * (1.0 - eta_val);
            dN_dxi[1][1] = -0.25 * (1.0 + xi_val);
            dN_dxi[2][0] = 0.25 * (1.0 + eta_val);
            dN_dxi[2][1] = 0.25 * (1.0 + xi_val);
            dN_dxi[3][0] = -0.25 * (1.0 + eta_val);
            dN_dxi[3][1] = 0.25 * (1.0 - xi_val);

            // J = vertices^T * dN_dxi, a (2x4)*(4x2) contraction.
            double j00 = 0.0, j01 = 0.0, j10 = 0.0, j11 = 0.0;
            for (int i = 0; i < 4; ++i) {
                j00 += vertices(i, 0) * dN_dxi[i][0];
                j01 += vertices(i, 0) * dN_dxi[i][1];
                j10 += vertices(i, 1) * dN_dxi[i][0];
                j11 += vertices(i, 1) * dN_dxi[i][1];
            }
            J = Mat2{j00, j01, j10, j11};
        } else {
            throw std::invalid_argument("Unknown element type: " + std::string(element_type));
        }

        Vec2 residual = x_mapped - x_phys;

        if (norm(residual) < tol) {
            return xi;
        }

        // Newton update
        xi = xi - J.inverse().apply(residual);

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

}  // namespace dgfem