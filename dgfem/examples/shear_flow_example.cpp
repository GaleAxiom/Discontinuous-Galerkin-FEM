/**
 * @file shear_flow_example.cpp
 * @brief Viscous shear flow example using the Navier-Stokes DG solver.
 *
 * This example mirrors the analytic shear flow used in the unit tests for
 * the Navier-Stokes weak formulation. The manufactured solution is
 * \(u(x, y) = (y, 0)\) with constant density and pressure. Viscous stresses
 * are balanced by the boundary data, so the solution is steady; the example
 * integrates it in time to demonstrate stability and provides basic error
 * metrics against the analytic profile.
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <cmath>
#include <filesystem>

#include <array>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

/**
 * @brief Analytic shear flow profile used for ICs and boundary conditions.
 * 
 * The velocity profile is u(y) = U_ref * y, with constant density and pressure.
 * To ensure low Mach number (incompressible-like behavior), we use:
 * - U_ref = 0.1 (reference velocity)
 * - p = 10.0 (high pressure to reduce Mach number)
 * - rho = 1.0 (reference density)
 * This gives Mach ≈ 0.1 * 1 / sqrt(1.4 * 10) ≈ 0.027, which is nearly incompressible.
 */
struct ShearFlowAnalytic {
    double gamma;
    double U_ref = 0.1;   // Reference velocity scale (reduced for low Mach)
    double p_ref = 10.0;  // Reference pressure (increased for low Mach)
    double rho_ref = 1.0; // Reference density

    [[nodiscard]] Eigen::Vector4d primitive(const Eigen::Vector2d& x) const {
        Eigen::Vector4d W;
        W << rho_ref, U_ref * x[1], 0.0, p_ref;
        return W;
    }

    [[nodiscard]] Eigen::Vector4d conserved(const Eigen::Vector2d& x) const {
        return dgfem::primitive_to_conserved(primitive(x), gamma);
    }
};

/**
 * @brief Compute an L2-like error for a primitive variable against the analytic state.
 */
double compute_variable_error(const std::shared_ptr<dgfem::DGMesh>& mesh,
                              const Eigen::MatrixXd& numerical_sol, const ShearFlowAnalytic& exact,
                              double gamma, int var_idx) {
    auto space = mesh->get_dg_space();
    auto mapping = space->get_mapping();
    const auto& phi = space->get_volume_basis_values();
    const auto& quad_pts = space->get_volume_quad()->points;
    const auto& quad_wts = space->get_volume_quad()->weights;

    int n_basis = space->get_basis()->get_n_basis();
    double error_sq = 0.0;
    double norm_sq = 0.0;

    for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");

        Eigen::MatrixXd vertices(mesh->get_elements().cols(), 2);
        for (int i = 0; i < vertices.rows(); ++i) {
            vertices.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem, i));
        }

        for (int q = 0; q < quad_wts.size(); ++q) {
            Eigen::Vector2d x_phys = mapping->map_to_physical(vertices, quad_pts.row(q));

            Eigen::Vector4d U_num = Eigen::Vector4d::Zero();
            for (int i = 0; i < n_basis; ++i) {
                for (int v = 0; v < 4; ++v) {
                    U_num[v] += numerical_sol(elem, i * 4 + v) * phi(q, i);
                }
            }

            Eigen::Vector4d W_num = dgfem::conserved_to_primitive(U_num, gamma);
            Eigen::Vector4d W_exact = exact.primitive(x_phys);

            double weight = quad_wts(q) * std::abs(J_det(q));
            double diff = W_num[var_idx] - W_exact[var_idx];
            error_sq += diff * diff * weight;
            norm_sq += W_exact[var_idx] * W_exact[var_idx] * weight;
        }
    }

    if (norm_sq < 1e-14) {
        return std::sqrt(error_sq);
    }
    return std::sqrt(error_sq / norm_sq);
}

}  // namespace

int main() {
    try {
        std::cout << "=== DGFEM Navier-Stokes Shear Flow Example ===" << std::endl;

        // Physical parameters matching the analytic test case
        constexpr double gamma = 1.4;
        constexpr double mu = 5.0e-6;
        constexpr double prandtl = 0.72;
        constexpr double penalty = 10000.0;

        // Build a modest triangular mesh over [0, 1] x [0, 1] with P1 basis
        auto mesh = dgfem::MeshSetup::create_standard_mesh(
            /*use_triangles=*/true,
            /*order=*/1,
            /*dx=*/0.05,
            /*n_vars=*/4,
            /*xmin=*/0.0,
            /*xmax=*/1.0,
            /*ymin=*/0.0,
            /*ymax=*/1.0);
        dgfem::MeshSetup::print_info(mesh);

        ShearFlowAnalytic exact{gamma};

        auto inlet_bc = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::INLET,
            [exact](const Eigen::Vector2d& x) { return exact.primitive(x); });
        auto outlet_bc = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::OUTLET,
            [exact](const Eigen::Vector2d& x) { return exact.primitive(x); });
        auto wall_bc = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::NO_SLIP_WALL,
            [exact](const Eigen::Vector2d& x) { return exact.primitive(x); });

        auto apply_channel_bcs = [&](const std::shared_ptr<dgfem::BoundaryConditionEuler>& bc,
                                     std::string_view tag) {
            const auto& tags = mesh->get_boundary_tags();
            auto it = tags.find(std::string(tag));
            if (it != tags.end()) {
                mesh->set_boundary_condition_euler(it->first, bc);
            } else {
                std::cerr << "Warning: boundary tag '" << tag
                          << "' not found; skipping BC assignment." << std::endl;
            }
        };

        apply_channel_bcs(inlet_bc, "Left");
        apply_channel_bcs(outlet_bc, "Right");
        apply_channel_bcs(wall_bc, "Bottom");
        apply_channel_bcs(wall_bc, "Top");

        mesh->build_precomputed_faces();

        apply_channel_bcs(inlet_bc, "Left");
        apply_channel_bcs(outlet_bc, "Right");
        apply_channel_bcs(wall_bc, "Bottom");
        apply_channel_bcs(wall_bc, "Top");

        // Time integration settings
        constexpr double dt = 1e-5;
        constexpr double T_final = 0.02;
        constexpr int save_every = 20;

        std::cout << "\n--- Solving viscous shear flow ---" << std::endl;
        std::cout << "  dt = " << dt << ", T_final = " << T_final << std::endl;

        dgfem::Timer solve_timer("Shear flow solve");
        dgfem::NavierStokesDGSolver solver(mesh, gamma, mu, prandtl, penalty);
        auto solutions =
            solver.solve([exact](const Eigen::Vector2d& x) { return exact.conserved(x); }, T_final,
                         dt, save_every);

        if (solutions.empty()) {
            throw std::runtime_error("Navier-Stokes solver did not return any solution frames.");
        }

        const auto& final_sol = solutions.back();
        std::array<std::string, 4> var_names{"rho", "u", "v", "p"};

        std::cout << "\n--- L2 relative errors vs analytic profile ---" << std::endl;
        for (int v = 0; v < 4; ++v) {
            double err = compute_variable_error(mesh, final_sol, exact, gamma, v);
            std::cout << "  " << std::setw(3) << var_names[v] << ": " << std::scientific
                      << std::setprecision(4) << err << std::endl;
        }

        std::cout << "\n--- Exporting " << solutions.size() << " frames to VTK ---" << std::endl;
        std::filesystem::path output_dir("output");
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "Warning: failed to create output directory ('" << output_dir.string()
                      << "'): " << ec.message() << std::endl;
        }

        for (size_t i = 0; i < solutions.size(); ++i) {
            auto filename = (output_dir / ("shear_flow_" + std::to_string(i))).string();
            dgfem::VTKWriter::write_euler_solution(mesh, solutions[i], filename, gamma,
                                                   /*refinement=*/1);
            if (i % 5 == 0 || i == solutions.size() - 1) {
                std::cout << "  Frame " << i << "/" << solutions.size() - 1 << std::endl;
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
