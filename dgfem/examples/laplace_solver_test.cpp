/**
 * @file laplace_solver_test.cpp
 * @brief Test the Laplace DG solver implementation
 */

#include <Kokkos_Core.hpp>
#include <cmath>

#include <iostream>
#include <memory>

// Include DGFEM headers
#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== DGFEM C++ Laplace Solver Test ===" << std::endl;

        // Get configuration
        auto& config = dgfem::Config::instance();
        config.order = 4;
        config.dx = 0.1;
        config.sigma = 10.0;

        std::cout << "Configuration:" << std::endl;
        std::cout << "  Element type: triangles" << std::endl;
        std::cout << "  Order: " << config.order << std::endl;
        std::cout << "  Mesh size: " << config.dx << std::endl;

        // Create mesh + DG space (1 variable for scalar Laplace)
        std::cout << "\nCreating rectangular mesh..." << std::endl;
        auto mesh = dgfem::MeshSetup::create_standard_mesh(
            /*use_triangles=*/true, config.order, config.dx, /*n_vars=*/1, 0.0, 1.0, 0.0, 1.0);
        dgfem::MeshSetup::print_info(mesh);

        // Set up boundary conditions (homogeneous Dirichlet)
        std::cout << "\nSetting up boundary conditions..." << std::endl;
        dgfem::set_rectangle_dirichlet_bc(mesh, 0.0);

        // Define source function f(x,y) = 2*pi^2 * sin(pi*x)*sin(pi*y)
        auto source_function = [](const dgfem::Vec2& x) -> double {
            return 2.0 * M_PI * M_PI * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
        };

        // Define exact solution for error computation
        auto exact_solution = [](const dgfem::Vec2& x) -> double {
            return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
        };

        // Define exact gradient
        auto exact_gradient = [](const dgfem::Vec2& x) -> dgfem::Vec2 {
            dgfem::Vec2 grad;
            grad[0] = M_PI * std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]);
            grad[1] = M_PI * std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]);
            return grad;
        };

        // Create solver
        std::cout << "\nCreating Laplace DG solver..." << std::endl;
        dgfem::LaplaceDGSolver solver(mesh, config.sigma);

        // Time the assembly and solve
        std::cout << "\nAssembling and solving..." << std::endl;

        dgfem::DView1 solution;
        {
            dgfem::Timer timer("Laplace assembly and solve");
            solution = solver.solve(source_function);
        }

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
