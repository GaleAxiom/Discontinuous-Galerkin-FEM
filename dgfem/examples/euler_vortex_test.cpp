/**
 * @file euler_vortex_test.cpp
 * @brief Test the Euler DG solver with isentropic vortex
 *
 * This test implements an isentropic vortex - a smooth exact solution
 * of the Euler equations useful for testing accuracy and stability.
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <cmath>

#include <iostream>

/**
 * @brief Isentropic vortex initial condition
 *
 * This analytical solution features a Gaussian vortex with specified strength.
 * Parameters are captured by value for thread-safety and simplicity.
 */
auto create_vortex_initial_condition(double gamma, double x_c, double y_c, double R, double sigma,
                                     double alpha, double beta, double M_inf, double u_inf,
                                     double v_inf) {
    return [=](const Eigen::Vector2d& x) -> Eigen::Vector4d {
        // Compute vortex perturbation based on distance from center
        double dx = x[0] - x_c;
        double dy = x[1] - y_c;
        double r2 = (dx * dx + dy * dy) / (R * R);

        // Gaussian profile for vortex strength
        double omega = beta * std::exp(-r2 / (2.0 * sigma * sigma));

        // Velocity and temperature perturbations
        double du = -dy / R * omega;
        double dv = dx / R * omega;
        double dT = -(gamma - 1.0) / 2.0 * omega * omega;

        // Isentropic relations for primitive variables
        double rho = std::pow(1.0 + dT, 1.0 / (gamma - 1.0));
        double u = M_inf * std::cos(alpha) + du + u_inf;
        double v = M_inf * std::sin(alpha) + dv + v_inf;
        double p = (1.0 / gamma) * std::pow(1.0 + dT, gamma / (gamma - 1.0));

        return dgfem::primitive_to_conserved(Eigen::Vector4d(rho, u, v, p), gamma);
    };
}

int main() {
    try {
        std::cout << "=== DGFEM C++ Euler Vortex Test ===" << std::endl;

        // Physics and vortex parameters
        const double gamma = 1.4;  // Ratio of specific heats
        const double beta = 5.0 / (2.0 * M_PI * std::sqrt(gamma)) * std::exp(0.5);

        // Create mesh using helper (triangles, order 3, spacing 0.1, 4 variables)
        auto mesh = dgfem::MeshSetup::create_standard_mesh(true, 3, 0.1, 4, -5.0, 5.0, -5.0, 5.0);
        dgfem::MeshSetup::print_info(mesh);

        // Set periodic boundaries for stationary vortex
        mesh->set_periodic_boundaries("Left", "Right");
        mesh->set_periodic_boundaries("Bottom", "Top");
        std::cout << "  Boundary conditions: PERIODIC on all sides" << std::endl;

        // Time stepping parameters
        const double dt = 0.0025;
        const double T_final = 100.0;
        const int save_every = 100;

        std::cout << "\n--- Time Stepping ---" << std::endl;
        std::cout << "  dt = " << dt << ", T_final = " << T_final
                  << ", frames = " << static_cast<int>(T_final / dt) / save_every + 1 << std::endl;

        // Initial condition: stationary vortex at origin
        auto ic =
            create_vortex_initial_condition(gamma, 0.0, 0.0, 1.0, 1.0, 0.0, beta, 0.0, 1.0, 1.0);

        // Solve
        dgfem::Timer solve_timer("Euler solve");
        dgfem::EulerDGSolver solver(mesh, gamma);
        auto solutions = solver.solve(ic, T_final, dt, save_every);

        std::cout << "\n--- Exporting " << solutions.size() << " frames to VTK ---" << std::endl;
        for (size_t i = 0; i < solutions.size(); ++i) {
            dgfem::VTKWriter::write_euler_solution(
                mesh, solutions[i], "output/euler_vortex_" + std::to_string(i), gamma, 2);
            if (i % 5 == 0 || i == solutions.size() - 1)
                std::cout << "  Frame " << i << "/" << solutions.size() - 1 << std::endl;
        }

        dgfem::MeshCreator::finalize_gmsh();
        std::cout << "\n=== Test COMPLETED! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
