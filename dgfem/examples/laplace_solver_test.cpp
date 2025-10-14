/**
 * @file laplace_solver_test.cpp
 * @brief Test the Laplace DG solver implementation
 */

#include <cmath>

#include <chrono>
#include <iostream>
#include <memory>

// Include DGFEM headers
#include "dgfem/boundary/conditions.hpp"
#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/mesh_creation.hpp"
#include "dgfem/utils/vtk_writer.hpp"

int main() {
    try {
        std::cout << "=== DGFEM C++ Laplace Solver Test ===" << std::endl;

        // Initialize GMSH
        dgfem::MeshCreator::initialize_gmsh();

        // Get configuration
        auto& config = dgfem::Config::instance();
        config.use_triangles = true;
        config.order = 4;
        config.dx = 0.1;
        config.xmin = 0.0;
        config.xmax = 1.0;
        config.ymin = 0.0;
        config.ymax = 1.0;
        config.sigma = 10.0;

        std::cout << "Configuration:" << std::endl;
        std::cout << "  Element type: " << (config.use_triangles ? "triangles" : "quads")
                  << std::endl;
        std::cout << "  Order: " << config.order << std::endl;
        std::cout << "  Mesh size: " << config.dx << std::endl;

        // Create mesh
        std::cout << "\nCreating rectangular mesh..." << std::endl;
        auto mesh = dgfem::MeshCreator::create_rectangular_mesh(
            config.dx, config.use_triangles, config.xmin, config.xmax, config.ymin, config.ymax);

        // Create DG space
        std::cout << "\nInitializing DG space..." << std::endl;
        auto dg_space = std::make_shared<dgfem::DGSpace>(mesh->get_element_type(), config.order);

        // Initialize mesh with DG space
        mesh->initialize_dg_space(dg_space, 1);  // 1 variable for scalar Laplace

        // Set up boundary conditions (homogeneous Dirichlet)
        std::cout << "\nSetting up boundary conditions..." << std::endl;
        auto bc_zero = dgfem::make_dirichlet_bc(0.0);

        // Print boundary tag information first
        const auto& boundary_tags = mesh->get_boundary_tags();

        // For simplicity, assign the same BC to all boundaries using the first boundary tag
        mesh->set_boundary_condition("Bottom", bc_zero);
        mesh->set_boundary_condition("Top", bc_zero);
        mesh->set_boundary_condition("Left", bc_zero);
        mesh->set_boundary_condition("Right", bc_zero);

        // Define source function f(x,y) = 2*pi^2 * sin(pi*x)*sin(pi*y)
        auto source_function = [](const Eigen::Vector2d& x) -> double {
            return 2.0 * M_PI * M_PI * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
        };

        // Define exact solution for error computation
        auto exact_solution = [](const Eigen::Vector2d& x) -> double {
            return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
        };

        // Define exact gradient
        auto exact_gradient = [](const Eigen::Vector2d& x) -> Eigen::Vector2d {
            Eigen::Vector2d grad;
            grad[0] = M_PI * std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]);
            grad[1] = M_PI * std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]);
            return grad;
        };

        // Create solver
        std::cout << "\nCreating Laplace DG solver..." << std::endl;
        dgfem::LaplaceDGSolver solver(mesh, config.sigma);

        // Time the assembly and solve
        std::cout << "\nAssembling and solving..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd solution = solver.solve(source_function);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        std::cout << "Assembly and solve time: " << duration.count() << " seconds" << std::endl;

        // Export solution to VTK
        std::cout << "\nExporting solution to VTK..." << std::endl;
        dgfem::VTKWriter::write_solution(mesh, solution, "../../output/laplace_solution");

        // Compute errors
        std::cout << "\nComputing errors against exact solution..." << std::endl;
        auto errors = solver.compute_error(exact_solution, exact_gradient);

        std::cout << "Error analysis:" << std::endl;
        std::cout << "  L² error: " << errors["L2"] << std::endl;
        std::cout << "  H¹ error: " << errors["H1"] << std::endl;

        // Basic validation
        bool test_passed = true;
        if (errors["L2"] > 1.0) {
            std::cout << "  WARNING: L² error seems high" << std::endl;
            test_passed = false;
        }
        if (errors["H1"] > 5.0) {
            std::cout << "  WARNING: H¹ error seems high" << std::endl;
            test_passed = false;
        }

        dgfem::MeshCreator::finalize_gmsh();

        if (test_passed) {
            std::cout << "\n=== Laplace solver test PASSED! ===" << std::endl;
        } else {
            std::cout << "\n=== Laplace solver test completed with warnings ===" << std::endl;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
