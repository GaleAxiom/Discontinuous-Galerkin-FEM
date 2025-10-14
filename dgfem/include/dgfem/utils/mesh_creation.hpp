/**
 * @file mesh_creation.hpp
 * @brief GMSH mesh creation utilities for DGFEM
 */

#pragma once

#include <Eigen/Dense>
#include <optional>

#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace dgfem {

// Forward declarations
class DGMesh;

/**
 * @brief GMSH mesh creation utilities
 */
class MeshCreator {
public:
    /**
     * @brief Create a simple rectangular mesh using GMSH
     * @param dx Target mesh size
     * @param use_triangles If true, use triangles; otherwise quads
     * @param xmin Left boundary
     * @param xmax Right boundary
     * @param ymin Bottom boundary
     * @param ymax Top boundary
     * @return Shared pointer to created DGMesh
     */
    [[nodiscard]] static std::shared_ptr<DGMesh>
    create_rectangular_mesh(double dx = 0.5, bool use_triangles = true, double xmin = 0.0,
                            double xmax = 1.0, double ymin = 0.0, double ymax = 1.0);

    /**
     * @brief Create a mesh for Euler equations (larger domain)
     * @param xmin Left boundary
     * @param xmax Right boundary
     * @param ymin Bottom boundary
     * @param ymax Top boundary
     * @param dx Target mesh size
     * @param use_triangles If true, use triangles; otherwise quads
     * @return Shared pointer to created DGMesh
     */
    [[nodiscard]] static std::shared_ptr<DGMesh>
    create_euler_mesh(double xmin = -5.0, double xmax = 5.0, double ymin = -5.0, double ymax = 5.0,
                      double dx = 0.2, bool use_triangles = false);

    /**
     * @brief Create a channel mesh with a cylindrical obstacle, suitable for von Kármán vortex street
     * simulations
     * @param length Channel length in the streamwise direction
     * @param height Channel height in the cross-stream direction
     * @param radius Cylinder radius
     * @param center Cylinder center position
     * @param dx_channel Target mesh size away from the cylinder
     * @param dx_cylinder Target mesh size on and near the cylinder boundary
     * @param use_triangles Generate triangles (default) or recombined quads
     * @return Shared pointer to created DGMesh
     */
    [[nodiscard]] static std::shared_ptr<DGMesh>
    create_cylinder_channel_mesh(double length = 2.2, double height = 0.41, double radius = 0.05,
                                 const Eigen::Vector2d& center = Eigen::Vector2d(0.2, 0.2),
                                 double dx_channel = 0.08, double dx_cylinder = 0.02,
                                 bool use_triangles = true);

    /**
     * @brief Create DGMesh from current GMSH model
     * @return Shared pointer to created DGMesh
     */
    [[nodiscard]] static std::shared_ptr<DGMesh> create_dg_mesh_from_gmsh();

    /**
     * @brief Initialize GMSH (must be called before mesh creation)
     */
    static void initialize_gmsh();

    /**
     * @brief Finalize GMSH (should be called after mesh creation)
     */
    static void finalize_gmsh();

private:
    /**
     * @brief Extract mesh data from GMSH
     */
    static void extract_mesh_data(Eigen::MatrixXd& vertices, Eigen::MatrixXi& elements,
                                  Eigen::VectorXi& element_tags,
                                  std::map<std::string, int>& boundary_tags,
                                  std::map<int, std::vector<std::pair<int, int>>>& boundary_edges);

    /**
     * @brief Set up rectangular geometry in GMSH
     */
    static void setup_rectangular_geometry(double xmin, double xmax, double ymin, double ymax,
                                           double dx, bool use_triangles = false);

    /**
     * @brief Configure mesh generation (triangles vs quads)
     */
    static void configure_mesh_generation(bool use_triangles);
};

/**
 * @brief Find reference coordinates for physical point (Newton iteration)
 * @param x_phys Physical coordinates
 * @param vertices Element vertices
 * @param element_type Element type ("triangle" or "quad")
 * @param tol Tolerance for convergence
 * @param max_iter Maximum iterations
 * @return Reference coordinates
 */
[[nodiscard]] std::optional<Eigen::Vector2d>
find_reference_coords(const Eigen::Vector2d& x_phys, const Eigen::MatrixXd& vertices,
                      std::string_view element_type, double tol = 1e-10, int max_iter = 20);

}  // namespace dgfem