/**
 * @file von_karman_vortices_example.cpp
 * @brief Laminar flow past a cylinder generating von Kármán vortices using the Navier-Stokes DG
 * solver.
 *
 * The setup follows the classical flow-around-a-cylinder benchmark (Re ≈ 100) inside a confined
 * channel. A uniform inflow profile drives the flow, producing periodic vortex shedding in the
 * cylinder wake. The example demonstrates how to set boundary conditions for walls, obstacles, and
 * inflow/outflow using the laminar Navier-Stokes formulation.
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <Kokkos_Core.hpp>
#include <filesystem>

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

struct CylinderChannelConfig {
    double length = 3.0;           ///< Channel length
    double height = 1.;            ///< Channel height
    double radius = 0.05;          ///< Cylinder radius
    dgfem::Vec2 center{0.2, 0.5};  ///< Cylinder center
    double dx_channel = 0.025;     ///< Characteristic mesh size away from the cylinder
    double dx_cylinder = 0.01;     ///< Refined size near the cylinder boundary
};

struct UniformInflowProfile {
    double gamma;
    double density;
    double max_velocity;
    double pressure;
    double channel_height;

    [[nodiscard]] double velocity(double y) const {
        (void)y;
        return max_velocity;
    }

    [[nodiscard]] double mean_velocity() const { return max_velocity; }

    [[nodiscard]] dgfem::Vec4 primitive(const dgfem::Vec2& x) const {
        double u = velocity(x[1]);
        return dgfem::Vec4{density, u, 0.0, pressure};
    }

    [[nodiscard]] dgfem::Vec4 conserved(const dgfem::Vec2& x) const {
        return dgfem::primitive_to_conserved(primitive(x), gamma);
    }
};

std::shared_ptr<dgfem::BoundaryConditionEuler> make_no_slip_bc(double density, double pressure) {
    dgfem::Vec4 W{density, 0.0, 0.0, pressure};
    return std::make_shared<dgfem::BoundaryConditionEuler>(dgfem::BCTypeEuler::NO_SLIP_WALL,
                                                           [W](const dgfem::Vec2&) { return W; });
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== DGFEM Navier-Stokes von Kármán Vortex Street Example ===" << std::endl;

        constexpr double gamma = 1.4;
        constexpr double density = 1.0;
        constexpr double pressure = 100.0;
        constexpr double mu = 1.0e-5;       // Dynamic viscosity (Re ≈ 100 with chosen scales)
        constexpr double prandtl = 0.72;    // Standard air value
        constexpr double penalty = 1000.0;  // SIPG penalty prefactor for viscous terms

        CylinderChannelConfig config;
        UniformInflowProfile inflow{gamma, density, 0.01, pressure, config.height};

        const int poly_order = 3;
        auto mesh = dgfem::MeshSetup::create_cylinder_channel_mesh(
            poly_order, config.dx_channel, config.dx_cylinder, /*n_vars=*/4, config.length,
            config.height, config.center, config.radius, /*use_triangles=*/true);
        dgfem::MeshSetup::print_info(mesh);

        double diameter = 2.0 * config.radius;
        double reynolds = density * inflow.mean_velocity() * diameter / mu;
        std::cout << "  Reynolds number (mean velocity based) ≈ " << std::setprecision(4)
                  << reynolds << std::endl;

        // Boundary conditions
        auto inlet_bc = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::FAR_FIELD,
            [inflow](const dgfem::Vec2& x) { return inflow.conserved(x); });

        auto outlet_bc = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::FAR_FIELD, [=](const dgfem::Vec2& x) {
                // Mirror the uniform inflow profile to minimise reflections at the outlet
                return inflow.conserved(x);
            });

        auto wall_bc = make_no_slip_bc(density, pressure);
        auto cylinder_bc = make_no_slip_bc(density, pressure);

        const auto& boundary_tags = mesh->get_boundary_tags();
        auto ensure_tag = [&boundary_tags](const std::string& tag) {
            if (boundary_tags.find(tag) == boundary_tags.end()) {
                throw std::runtime_error("Required boundary tag missing: " + tag);
            }
        };

        ensure_tag("Inlet");
        ensure_tag("Outlet");
        ensure_tag("LowerWall");
        ensure_tag("UpperWall");
        ensure_tag("Cylinder");

        mesh->build_precomputed_faces();
        mesh->set_boundary_condition_euler("Inlet", inlet_bc);
        mesh->set_boundary_condition_euler("Outlet", outlet_bc);
        mesh->set_boundary_condition_euler("LowerWall", wall_bc);
        mesh->set_boundary_condition_euler("UpperWall", wall_bc);
        mesh->set_boundary_condition_euler("Cylinder", cylinder_bc);

        // Time integration settings tailored for vortex shedding development
        constexpr double dt = 1.0e-5;
        constexpr double T_final = 1.0;
        constexpr int save_every = 25;

        std::cout << "\n--- Solving laminar Navier-Stokes ---" << std::endl;
        std::cout << "  dt = " << dt << ", T_final = " << T_final << std::endl;

        dgfem::Timer solve_timer("von Kármán vortex solve");
        dgfem::NavierStokesDGSolver solver(mesh, gamma, mu, prandtl, penalty);
        auto solution_frames =
            solver.solve([inflow](const dgfem::Vec2& x) { return inflow.conserved(x); }, T_final,
                         dt, save_every);

        if (solution_frames.empty()) {
            throw std::runtime_error("Navier-Stokes solver did not return any solution frames.");
        }

        std::cout << "Generated " << solution_frames.size() << " solution frames." << std::endl;

        std::filesystem::path output_dir("output");
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "Warning: failed to create output directory ('" << output_dir.string()
                      << "'): " << ec.message() << std::endl;
        }

        std::cout << "\n--- Exporting frames to VTK ---" << std::endl;
        for (size_t i = 0; i < solution_frames.size(); ++i) {
            auto filename = (output_dir / ("von_karman_frame_" + std::to_string(i))).string();
            dgfem::VTKWriter::write_euler_solution(mesh, solution_frames[i], filename, gamma,
                                                   /*refinement=*/1);
            if (i % 5 == 0 || i == solution_frames.size() - 1) {
                std::cout << "  Frame " << i << "/" << solution_frames.size() - 1 << std::endl;
            }
        }

        dgfem::MeshCreator::finalize_gmsh();
        std::cout << "\n=== Example COMPLETED successfully ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
