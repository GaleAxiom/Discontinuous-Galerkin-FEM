/**
 * @file von_karman_test.cpp
 * @brief Test the Euler DG solver with von Kármán vortex street
 *
 * This test simulates flow past a circular cylinder using the Euler equations.
 * At moderate Reynolds numbers (inviscid limit), the flow develops the famous
 * von Kármán vortex street - alternating vortices shed from the cylinder.
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <cmath>

#include <iostream>

/**
 * @brief Create uniform flow initial condition
 *
 * Sets up a uniform flow field with specified free-stream conditions.
 * A small perturbation can be added to trigger vortex shedding.
 */
auto create_uniform_flow_ic(double rho_inf, double u_inf, double v_inf, double p_inf,
                            double gamma) {
    return [=](const Eigen::Vector2d& x) -> Eigen::Vector4d {
        // Uniform flow with primitive variables: [rho, u, v, p]
        Eigen::Vector4d W_inf(rho_inf, u_inf, v_inf, p_inf);

        // Convert to conserved variables: [rho, rho*u, rho*v, E]
        return dgfem::primitive_to_conserved(W_inf, gamma);
    };
}

/**
 * @brief Create initial condition with small perturbation to trigger vortex shedding
 */
auto create_perturbed_flow_ic(double rho_inf, double u_inf, double v_inf, double p_inf,
                              double gamma, const Eigen::Vector2d& cylinder_center,
                              double cylinder_radius, double perturbation_amplitude = 0.01) {
    return [=](const Eigen::Vector2d& x) -> Eigen::Vector4d {
        double dx = x[0] - cylinder_center[0];
        double dy = x[1] - cylinder_center[1];
        double r = std::sqrt(dx * dx + dy * dy);

        // Add small perturbation near the cylinder to break symmetry
        double v_pert = 0.0;
        if (r > cylinder_radius && r < cylinder_radius + 3.0 * cylinder_radius) {
            double decay = std::exp(-(r - cylinder_radius) / cylinder_radius);
            v_pert = perturbation_amplitude * u_inf * decay * std::sin(5.0 * std::atan2(dy, dx));
        }

        // Primitive variables with perturbation
        Eigen::Vector4d W(rho_inf, u_inf, v_inf + v_pert, p_inf);

        return dgfem::primitive_to_conserved(W, gamma);
    };
}

int main() {
    try {
        std::cout << "=== DGFEM C++ von Kármán Vortex Street Test ===" << std::endl;

        // Physics parameters
        const double gamma = 1.4;  // Ratio of specific heats

        // Free-stream conditions (dimensionless)
        const double rho_inf = 1.0;                               // Density
        const double u_inf = 0.3;                                 // Streamwise velocity (subsonic)
        const double v_inf = 0.0;                                 // Cross-stream velocity
        const double p_inf = 10.0;                                // Pressure
        const double T_inf = p_inf / rho_inf;                     // Temperature (ideal gas)
        const double c_inf = std::sqrt(gamma * p_inf / rho_inf);  // Speed of sound
        const double M_inf = u_inf / c_inf;                       // Mach number

        std::cout << "\n--- Flow Conditions ---" << std::endl;
        std::cout << "  Mach number: " << M_inf << std::endl;
        std::cout << "  Density: " << rho_inf << std::endl;
        std::cout << "  Velocity: (" << u_inf << ", " << v_inf << ")" << std::endl;
        std::cout << "  Pressure: " << p_inf << std::endl;
        std::cout << "  Temperature: " << T_inf << std::endl;

        // Geometric parameters for the channel and cylinder
        const double channel_length = 2.2;                // Streamwise length
        const double channel_height = 0.41;               // Cross-stream height
        const double cylinder_radius = 0.05;              // Cylinder radius
        const Eigen::Vector2d cylinder_center(0.2, 0.2);  // Cylinder center

        // Mesh parameters
        const int order = 2;              // Polynomial order
        const double dx_channel = 0.04;   // Mesh size away from cylinder
        const double dx_cylinder = 0.02;  // Mesh size near cylinder
        const bool use_triangles = true;  // Use triangular elements

        std::cout << "\n--- Geometry ---" << std::endl;
        std::cout << "  Channel: " << channel_length << " × " << channel_height << std::endl;
        std::cout << "  Cylinder: center (" << cylinder_center[0] << ", " << cylinder_center[1]
                  << "), radius " << cylinder_radius << std::endl;

        // Create mesh with cylinder
        auto mesh = dgfem::MeshSetup::create_cylinder_channel_mesh(
            order, dx_channel, dx_cylinder, 4,  // 4 variables for Euler
            channel_length, channel_height, cylinder_center, cylinder_radius, use_triangles);
        dgfem::MeshSetup::print_info(mesh);

        // Set boundary conditions
        std::cout << "\n--- Boundary Conditions ---" << std::endl;

        // Inlet: prescribed far-field state
        Eigen::Vector4d W_inlet(rho_inf, u_inf, v_inf, p_inf);
        Eigen::Vector4d U_inlet = dgfem::primitive_to_conserved(W_inlet, gamma);
        auto bc_inlet =
            std::make_shared<dgfem::BoundaryConditionEuler>(dgfem::BCTypeEuler::INLET, U_inlet);
        mesh->set_boundary_condition_euler("Inlet", bc_inlet);
        std::cout << "  Inlet: INLET with U = [" << U_inlet.transpose() << "]" << std::endl;

        // Outlet: prescribed pressure
        Eigen::Vector4d W_outlet(rho_inf, u_inf, v_inf, p_inf);
        Eigen::Vector4d U_outlet = dgfem::primitive_to_conserved(W_outlet, gamma);
        auto bc_outlet =
            std::make_shared<dgfem::BoundaryConditionEuler>(dgfem::BCTypeEuler::OUTLET, U_outlet);
        mesh->set_boundary_condition_euler("Outlet", bc_outlet);
        std::cout << "  Outlet: OUTLET with P = " << p_inf << std::endl;

        // Top and bottom walls: slip wall (inviscid)
        Eigen::Vector4d dummy_value(0.0, 0.0, 0.0, 0.0);  // Not used for slip wall
        auto bc_slip = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::SLIP_WALL, dummy_value);
        mesh->set_boundary_condition_euler("UpperWall", bc_slip);
        mesh->set_boundary_condition_euler("LowerWall", bc_slip);
        std::cout << "  UpperWall/LowerWall: SLIP_WALL" << std::endl;

        // Cylinder surface: slip wall (inviscid)
        auto bc_cylinder = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::SLIP_WALL, dummy_value);
        mesh->set_boundary_condition_euler("Cylinder", bc_cylinder);
        std::cout << "  Cylinder: SLIP_WALL" << std::endl;

        // Time stepping parameters
        // CFL condition: dt <= CFL * h / (|u| + c)
        const double char_length = dx_cylinder;
        const double max_wave_speed = u_inf + c_inf;
        const double CFL = 0.01;
        const double dt = CFL * char_length / max_wave_speed;

        // Simulation time - need long enough to see vortex shedding develop
        // Typical shedding period ~ 10 * D / U (rough estimate)
        const double shedding_period = 10.0 * (2.0 * cylinder_radius) / u_inf;
        const double T_final = 3.;

        const int save_every = 100;  // Save every N time steps

        std::cout << "\n--- Time Stepping ---" << std::endl;
        std::cout << "  CFL number: " << CFL << std::endl;
        std::cout << "  Time step: dt = " << dt << std::endl;
        std::cout << "  Final time: T_final = " << T_final << std::endl;
        std::cout << "  Est. shedding period: " << shedding_period << std::endl;
        std::cout << "  Total steps: " << static_cast<int>(T_final / dt) << std::endl;
        std::cout << "  Output frames: " << static_cast<int>(T_final / dt) / save_every + 1
                  << std::endl;

        // Initial condition with small perturbation to trigger vortex shedding
        auto ic = create_perturbed_flow_ic(rho_inf, u_inf, v_inf, p_inf, gamma, cylinder_center,
                                           cylinder_radius, 0.05);

        // Solve
        std::cout << "\n--- Starting Simulation ---" << std::endl;
        dgfem::Timer solve_timer("von Kármán vortex simulation");
        dgfem::EulerDGSolver solver(mesh, gamma);

        auto solutions = solver.solve(ic, T_final, dt, save_every);

        std::cout << "\n--- Exporting " << solutions.size() << " frames to VTK ---" << std::endl;
        for (size_t i = 0; i < solutions.size(); ++i) {
            dgfem::VTKWriter::write_euler_solution(
                mesh, solutions[i], "output/von_karman_" + std::to_string(i), gamma, order);
            if (i % 10 == 0 || i == solutions.size() - 1) {
                std::cout << "  Frame " << i << "/" << solutions.size() - 1
                          << " (t = " << i * save_every * dt << ")" << std::endl;
            }
        }

        dgfem::MeshCreator::finalize_gmsh();
        std::cout << "\n=== von Kármán Vortex Test COMPLETED! ===" << std::endl;
        std::cout << "\nVisualization tips:" << std::endl;
        std::cout << "  - Load output/von_karman_*.vtk in ParaView" << std::endl;
        std::cout << "  - Visualize vorticity magnitude: curl of velocity field" << std::endl;
        std::cout << "  - Look for alternating vortices in the wake" << std::endl;
        std::cout << "  - Plot pressure or density contours" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
