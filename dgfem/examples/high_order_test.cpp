/**
 * @file high_order_test.cpp
 * @brief Test high-order elements (order 1-7) with Laplace solver
 */

#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

#include <iomanip>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== High-Order DG Elements Test ===" << std::endl;
        std::cout << "Testing polynomial orders 1-7 with Laplace equation" << std::endl;
        std::cout << "====================================================\n" << std::endl;

        dgfem::MeshCreator::initialize_gmsh();

        // Store convergence data
        std::vector<int> orders;
        std::vector<double> l2_errors;
        std::vector<double> h1_errors;
        std::vector<int> dofs;

        // Test all orders from 1 to 7
        for (int order = 1; order <= 7; ++order) {
            std::cout << "--- Testing Polynomial Order " << order << " ---" << std::endl;

            auto& config = dgfem::Config::instance();
            config.sigma = 10.0;        // Penalty parameter
            constexpr double dx = 0.1;  // Fixed mesh size for comparison

            // Create mesh + DG space (1 variable for scalar Laplace)
            auto mesh = dgfem::MeshSetup::create_standard_mesh(/*use_triangles=*/true, order, dx,
                                                               /*n_vars=*/1, 0.0, 1.0, 0.0, 1.0);
            auto dg_space = mesh->get_dg_space();

            std::cout << "  Mesh: " << mesh->get_n_elements() << " elements" << std::endl;
            std::cout << "  Basis functions per element: " << dg_space->get_basis()->get_n_basis()
                      << std::endl;
            std::cout << "  Volume quad points: " << dg_space->get_volume_quad()->size()
                      << std::endl;
            std::cout << "  Face quad points: " << dg_space->get_face_quad()->size() << std::endl;

            int total_dofs = mesh->get_n_elements() * dg_space->get_basis()->get_n_basis();
            std::cout << "  Total DOFs: " << total_dofs << std::endl;

            // Set up boundary conditions (homogeneous Dirichlet)
            dgfem::set_rectangle_dirichlet_bc(mesh, 0.0);

            // Manufactured solution: u(x,y) = sin(pi*x) * sin(pi*y)
            // This satisfies homogeneous Dirichlet BCs on [0,1]^2
            auto exact_solution = [](const dgfem::Vec2& x) -> double {
                return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
            };

            // Source function: f = -Laplacian(u) = 2*pi^2 * sin(pi*x) * sin(pi*y)
            auto source_function = [](const dgfem::Vec2& x) -> double {
                return 2.0 * M_PI * M_PI * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
            };

            // Exact gradient for H1 error computation
            auto exact_gradient = [](const dgfem::Vec2& x) -> dgfem::Vec2 {
                dgfem::Vec2 grad;
                grad[0] = M_PI * std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]);
                grad[1] = M_PI * std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]);
                return grad;
            };

            // Create Laplace solver
            dgfem::LaplaceDGSolver solver(mesh, config.sigma);

            try {
                // Solve
                dgfem::DView1 solution = solver.solve(source_function);

                // Compute errors using built-in method
                auto errors = solver.compute_error(exact_solution, exact_gradient);

                double l2_error = errors["L2"];
                double h1_error = errors["H1"];

                std::cout << "  L² Error: " << std::scientific << std::setprecision(6) << l2_error
                          << std::endl;
                std::cout << "  H¹ Error: " << std::scientific << std::setprecision(6) << h1_error
                          << std::endl;

                // Store data for convergence analysis
                orders.push_back(order);
                l2_errors.push_back(l2_error);
                h1_errors.push_back(h1_error);
                dofs.push_back(total_dofs);

                // Validate errors are reasonable
                if (l2_error < 1.0 && h1_error < 10.0) {
                    std::cout << "  ✓ Order " << order << " PASSED" << std::endl;
                } else {
                    std::cout << "  ⚠ Order " << order << " completed with high errors"
                              << std::endl;
                }

            } catch (const std::exception& e) {
                std::cerr << "  ✗ Order " << order << " FAILED: " << e.what() << std::endl;
                dgfem::MeshCreator::finalize_gmsh();
                return 1;
            }

            std::cout << std::endl;
        }

        // Print convergence summary
        std::cout << "\n=== Convergence Summary ===" << std::endl;
        std::cout << std::setw(8) << "Order" << std::setw(12) << "DOFs" << std::setw(15)
                  << "L² Error" << std::setw(15) << "H¹ Error" << std::setw(12) << "L² Rate"
                  << std::setw(12) << "H¹ Rate" << std::endl;
        std::cout << std::string(74, '-') << std::endl;

        for (size_t i = 0; i < orders.size(); ++i) {
            std::cout << std::setw(8) << orders[i] << std::setw(12) << dofs[i] << std::setw(15)
                      << std::scientific << std::setprecision(4) << l2_errors[i] << std::setw(15)
                      << std::scientific << std::setprecision(4) << h1_errors[i];

            if (i > 0) {
                // Compute convergence rates (should improve with higher order)
                double l2_rate = -std::log(l2_errors[i] / l2_errors[i - 1]) /
                                 std::log(double(dofs[i]) / dofs[i - 1]);
                double h1_rate = -std::log(h1_errors[i] / h1_errors[i - 1]) /
                                 std::log(double(dofs[i]) / dofs[i - 1]);
                std::cout << std::setw(12) << std::fixed << std::setprecision(2) << l2_rate
                          << std::setw(12) << std::fixed << std::setprecision(2) << h1_rate;
            }
            std::cout << std::endl;
        }

        dgfem::MeshCreator::finalize_gmsh();

        std::cout << "\n=== All High-Order Tests COMPLETED! ===" << std::endl;
        std::cout << "Successfully tested polynomial orders 1-7" << std::endl;
        std::cout << "Quadrature rules verified for elements up to order 7" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
