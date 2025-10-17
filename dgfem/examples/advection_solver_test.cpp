/**
 * @file advection_solver_test.cpp
 * @brief Test the Advection DG solver with a Gaussian pulse
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <cmath>

#include <iostream>

int main() {
    try {
        std::cout << "=== DGFEM Advection Solver Test ===" << std::endl;

        // Create mesh (triangles, order 2, spacing 0.01, 1 variable, domain [0,1]x[0,1])
        auto mesh = dgfem::MeshSetup::create_standard_mesh(true, 2, 0.01, 1, 0.0, 1.0, 0.0, 1.0);
        dgfem::MeshSetup::print_info(mesh);

        // Advection velocity (diagonal flow)
        Eigen::Vector2d velocity(0.1, 0.1);
        std::cout << "  Advection velocity: [" << velocity(0) << ", " << velocity(1) << "]"
                  << std::endl;

        // Boundary conditions: Inflow (Left/Bottom) = 0, Outflow (Right/Top) = natural
        auto bc_zero = dgfem::make_dirichlet_bc(0.0);
        mesh->set_boundary_condition("Bottom", bc_zero);
        mesh->set_boundary_condition("Left", bc_zero);

        // Initial condition: Gaussian pulse at center
        auto initial_condition = [](const Eigen::Vector2d& x) -> double {
            double r2 = std::pow(x(0) - 0.5, 2) + std::pow(x(1) - 0.5, 2);
            return std::exp(-r2 / (2.0 * 0.05 * 0.05));
        };

        // Time stepping
        constexpr double T_final = 2.0;
        constexpr double dt = 0.001;
        constexpr int save_every = 100;

        std::cout << "\n--- Solving ---" << std::endl;
        std::cout << "  T_final = " << T_final << ", dt = " << dt
                  << ", frames = " << static_cast<int>(T_final / dt) / save_every + 1 << std::endl;

        dgfem::Timer timer("Advection solve");
        dgfem::AdvectionDGSolver solver(mesh, velocity);
        auto solutions = solver.solve(initial_condition, T_final, dt, bc_zero, save_every);

        // Export and validate
        std::cout << "\n--- Exporting " << solutions.size() << " frames ---" << std::endl;
        for (size_t i = 0; i < solutions.size(); ++i) {
            dgfem::VTKWriter::write_solution(mesh, solutions[i],
                                             "output/advection_solution_" + std::to_string(i));
            if (i % 5 == 0 || i == solutions.size() - 1)
                std::cout << "  Frame " << i << std::endl;
        }

        // Statistics
        double norm_ratio = solutions.back().norm() / solutions[0].norm();
        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "  Solution norm ratio (final/initial): " << norm_ratio << std::endl;
        std::cout << "  Mass change: "
                  << std::abs(solutions.back().sum() - solutions[0].sum()) /
                         std::abs(solutions[0].sum()) * 100
                  << "%" << std::endl;

        bool passed = !std::isnan(solutions.back().norm()) && norm_ratio < 2.0;
        dgfem::MeshCreator::finalize_gmsh();

        std::cout << "\n=== Test " << (passed ? "PASSED" : "FAILED") << " ===" << std::endl;
        return passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
