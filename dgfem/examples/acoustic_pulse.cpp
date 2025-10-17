/**
 * @file acoustic_pulse.cpp
 * @brief Acoustic pulse propagation test for the Euler DG solver
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <Eigen/Dense>
#include <cmath>

#include <iomanip>
#include <iostream>

/**
 * @brief Create Gaussian acoustic pulse initial condition in primitive variables
 */
auto create_gaussian_pulse_ic(double gamma, double rho0, double p0, double amplitude, double sigma,
                              const Eigen::Vector2d& center) {
    // Linear acoustic relation: delta_p = c^2 * delta_rho
    const double sound_speed2 = gamma * p0 / rho0;

    return [=](const Eigen::Vector2d& x) -> Eigen::Vector4d {
        const double dx = x[0] - center[0];
        const double dy = x[1] - center[1];
        const double r2 = dx * dx + dy * dy;

        const double delta_p = amplitude * std::exp(-r2 / (2.0 * sigma * sigma));
        const double p = p0 + delta_p;
        const double rho = rho0 + delta_p / sound_speed2;

        // Primitive variables: [rho, u, v, p]
        Eigen::Vector4d W(rho, 0.0, 0.0, p);
        return dgfem::primitive_to_conserved(W, gamma);
    };
}

int main() {
    try {
        std::cout << "=== DGFEM Acoustic Pulse Test ===" << std::endl;

        // Gas parameters (dimensionless ideal gas)
        constexpr double gamma = 1.4;
        constexpr double rho0 = 1.0;
        constexpr double p0 = 1.0;
        const double c0 = std::sqrt(gamma * p0 / rho0);

        // Domain: unit square to exercise all outlet faces symmetrically
        constexpr double x_min = 0.0;
        constexpr double x_max = 1.0;
        constexpr double y_min = 0.0;
        constexpr double y_max = 1.0;

        // Mesh configuration
        const int order = 1;
        const double dx = 1.0 / 80.0;  // ~80 first-order elements across domain length
        const bool use_triangles = true;

        auto mesh = dgfem::MeshSetup::create_standard_mesh(use_triangles, order, dx, 4, x_min,
                                                           x_max, y_min, y_max);
        dgfem::MeshSetup::print_info(mesh);

        // Acoustic pulse parameters
        const double amplitude = 5.0e-4;         // Small pressure perturbation
        const double sigma = 0.10;               // Pulse width
        const Eigen::Vector2d center(0.5, 0.5);  // Pulse centered in the box

        std::cout << "\n--- Initial Pulse ---" << std::endl;
        std::cout << "  Amplitude: " << amplitude << std::endl;
        std::cout << "  Width (sigma): " << sigma << std::endl;
        std::cout << "  Center: (" << center.transpose() << ")" << std::endl;

        // Boundary conditions: use outlet BC on all edges with ambient state
        const Eigen::Vector4d W_ambient(rho0, 0.0, 0.0, p0);
        const Eigen::Vector4d U_ambient = dgfem::primitive_to_conserved(W_ambient, gamma);
        auto bc_outlet =
            std::make_shared<dgfem::BoundaryConditionEuler>(dgfem::BCTypeEuler::OUTLET, U_ambient);
        const Eigen::Vector4d zero_state = Eigen::Vector4d::Zero();
        auto bc_slip = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::SLIP_WALL, zero_state);
        mesh->set_boundary_condition_euler("Left", bc_outlet);
        mesh->set_boundary_condition_euler("Right", bc_outlet);
        mesh->set_boundary_condition_euler("Bottom", bc_outlet);
        mesh->set_boundary_condition_euler("Top", bc_outlet);

        std::cout << "\n--- Boundary Conditions ---" << std::endl;
        std::cout << "  Left/Right: OUTLET (ambient state)" << std::endl;
        std::cout << "  Top/Bottom: SLIP_WALL" << std::endl;

        // Time stepping parameters based on acoustic CFL limit
        const double CFL = 0.1;
        const double dt = CFL * dx / c0;
        const double T_final = 1.0;  // Stop before reflections dominate
        const int save_every = 8;

        std::cout << "\n--- Time Stepping ---" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  CFL = " << CFL << std::endl;
        std::cout << "  dt  = " << dt << std::endl;
        std::cout << "  T_final = " << T_final << std::endl;
        std::cout << "  Total steps ≈ " << static_cast<int>(std::ceil(T_final / dt)) << std::endl;

        // Create initial condition and solve
        auto ic = create_gaussian_pulse_ic(gamma, rho0, p0, amplitude, sigma, center);

        std::cout << "\n--- Solving ---" << std::endl;
        dgfem::Timer timer("Acoustic pulse solve");
        dgfem::EulerDGSolver solver(mesh, gamma);
        auto solutions = solver.solve(ic, T_final, dt, save_every);

        std::cout << "\n--- Exporting " << solutions.size() << " frames ---" << std::endl;
        for (size_t i = 0; i < solutions.size(); ++i) {
            dgfem::VTKWriter::write_euler_solution(
                mesh, solutions[i], "output/acoustic_pulse_" + std::to_string(i), gamma, order);
            if (i % 5 == 0 || i == solutions.size() - 1) {
                std::cout << "  Frame " << i << "/" << solutions.size() - 1
                          << " (t = " << i * save_every * dt << ")" << std::endl;
            }
        }

        dgfem::MeshCreator::finalize_gmsh();
        std::cout << "\n=== Acoustic Pulse Test COMPLETED ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
