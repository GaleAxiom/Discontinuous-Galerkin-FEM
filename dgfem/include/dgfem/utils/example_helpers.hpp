/**
 * @file example_helpers.hpp
 * @brief Utility functions and classes to simplify example code
 *
 * This header provides common functionality used across examples,
 * reducing code duplication and improving readability.
 */

#pragma once

#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/solver/weak_form.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "mesh_creation.hpp"

namespace dgfem {

/**
 * @brief Simple RAII timer for performance measurements
 *
 * Usage:
 *   {
 *     Timer t("Operation name");
 *     // ... code to time ...
 *   } // Prints elapsed time on destruction
 */
class Timer {
public:
    explicit Timer(const std::string& name)
        : label_(name), start_(std::chrono::high_resolution_clock::now()) {
        std::cout << "[TIMER] Starting: " << label_ << std::endl;
    }

    ~Timer() {
        auto duration =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_)
                .count();
        std::cout << "[TIMER] " << label_ << " took " << std::fixed << std::setprecision(4)
                  << duration << " seconds" << std::endl;
    }

    /**
     * @brief Get elapsed time without destroying the timer
     */
    [[nodiscard]] double elapsed() const {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_)
            .count();
    }

private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Helper for setting up standard mesh and DG space configuration
 */
class MeshSetup {
public:
    /**
     * @brief Initialize mesh with standard configuration
     * @param use_triangles Use triangles (true) or quads (false)
     * @param order Polynomial order
     * @param dx Mesh spacing
     * @param n_vars Number of solution variables (default=1 for scalar)
     */
    static std::shared_ptr<DGMesh> create_standard_mesh(bool use_triangles, int order, double dx,
                                                        int n_vars = 1, double xmin = -1.0,
                                                        double xmax = 1.0, double ymin = -1.0,
                                                        double ymax = 1.0) {
        // Initialize GMSH
        MeshCreator::initialize_gmsh();

        // Configure
        auto& config = Config::instance();
        config.use_triangles = use_triangles;
        config.order = order;
        config.dx = dx;
        config.xmin = xmin;
        config.xmax = xmax;
        config.ymin = ymin;
        config.ymax = ymax;

        // Create mesh
        auto mesh = MeshCreator::create_rectangular_mesh(dx, use_triangles, xmin, xmax, ymin, ymax);

        // Create DG space
        auto dg_space = std::make_shared<DGSpace>(mesh->get_element_type(), order);

        // Initialize mesh with DG space
        mesh->initialize_dg_space(dg_space, n_vars);

        return mesh;
    }

    /**
     * @brief Create a DG-ready mesh for a channel with a cylindrical obstacle.
     */
    static std::shared_ptr<DGMesh>
    create_cylinder_channel_mesh(int order, double dx_channel, double dx_cylinder, int n_vars = 4,
                                 double length = 2.2, double height = 0.41,
                                 const Eigen::Vector2d& center = Eigen::Vector2d(0.2, 0.2),
                                 double radius = 0.05, bool use_triangles = true) {
        MeshCreator::initialize_gmsh();

        auto& config = Config::instance();
        config.use_triangles = use_triangles;
        config.order = order;
        config.dx = dx_channel;
        config.xmin = 0.0;
        config.xmax = length;
        config.ymin = 0.0;
        config.ymax = height;

        auto mesh = MeshCreator::create_cylinder_channel_mesh(
            length, height, radius, center, dx_channel, dx_cylinder, use_triangles);
        auto dg_space = std::make_shared<DGSpace>(mesh->get_element_type(), order);
        mesh->initialize_dg_space(dg_space, n_vars);

        return mesh;
    }

    /**
     * @brief Print mesh and space information
     */
    static void print_info(const std::shared_ptr<DGMesh>& mesh) {
        auto dg_space = mesh->get_dg_space();

        std::cout << "\n--- Mesh Information ---" << std::endl;
        std::cout << "  Element type: " << mesh->get_element_type() << std::endl;
        std::cout << "  Number of elements: " << mesh->get_n_elements() << std::endl;
        std::cout << "  Number of vertices: " << mesh->get_vertices().rows() << std::endl;
        std::cout << "  Polynomial order: " << dg_space->get_basis()->get_order() << std::endl;
        std::cout << "  Basis functions per element: " << dg_space->get_basis()->get_n_basis()
                  << std::endl;
        std::cout << "  Total DOFs: "
                  << mesh->get_n_elements() * dg_space->get_basis()->get_n_basis() *
                         mesh->get_solution()->get_n_variables()
                  << std::endl;
    }
};

}  // namespace dgfem
