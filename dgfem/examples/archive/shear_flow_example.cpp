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

#include <Kokkos_Core.hpp>
#include <cmath>
#include <filesystem>

#include <array>
#include <iomanip>
#include <iostream>

namespace {

/**
 * @brief Analytic shear flow profile used for ICs and boundary conditions.
 */
struct ShearFlowAnalytic {
    double gamma;

    [[nodiscard]] dgfem::Vec4 primitive(const dgfem::Vec2& x) const {
        return dgfem::Vec4{1.0, x[1], 0.0, 1.0};
    }

    [[nodiscard]] dgfem::Vec4 conserved(const dgfem::Vec2& x) const {
        return dgfem::primitive_to_conserved(primitive(x), gamma);
    }
};

/**
 * @brief Compute an L2-like error for a primitive variable against the analytic state.
 */
double compute_variable_error(const std::shared_ptr<dgfem::DGMesh>& mesh,
                              const dgfem::DView2& numerical_sol, const ShearFlowAnalytic& exact,
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
        const dgfem::DView2& J_det = elem_data.at("J_det_vol");

        dgfem::DView2 vertices = mesh->get_element_vertices(elem);

        for (int q = 0; q < static_cast<int>(quad_wts.size()); ++q) {
            dgfem::Vec2 x_phys = mapping->map_to_physical(vertices, dgfem::row2(quad_pts, q));

            dgfem::Vec4 U_num{0.0, 0.0, 0.0, 0.0};
            for (int i = 0; i < n_basis; ++i) {
                for (int v = 0; v < 4; ++v) {
                    U_num[v] += numerical_sol(elem, i * 4 + v) * phi(q, i);
                }
            }

            dgfem::Vec4 W_num = dgfem::conserved_to_primitive(U_num, gamma);
            dgfem::Vec4 W_exact = exact.primitive(x_phys);

            double weight = quad_wts(q) * std::abs(J_det(q, 0));
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

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== DGFEM Navier-Stokes Shear Flow Example ===" << std::endl;

        // Physical parameters matching the analytic test case
        constexpr double gamma = 1.4;
        constexpr double mu = 5.0e-6;
        constexpr double prandtl = 0.72;
        constexpr double penalty = 10000.0;

        // Build a modest triangular mesh over [0, 1] x [0, 1] with P1 basis
        auto mesh = dgfem::MeshSetup::create_standard_mesh(
            /*use_triangles=*/false,
            /*order=*/2,
            /*dx=*/0.05,
            /*n_vars=*/4,
            /*xmin=*/0.0,
            /*xmax=*/1.0,
            /*ymin=*/0.0,
            /*ymax=*/1.0);
        dgfem::MeshSetup::print_info(mesh);

        ShearFlowAnalytic exact{gamma};

        // Apply shear-consistent far-field boundary conditions on all faces
        auto shear_bc = std::make_shared<dgfem::BoundaryConditionEuler>(
            dgfem::BCTypeEuler::FAR_FIELD,
            [exact](const dgfem::Vec2& x) { return exact.conserved(x); });
        for (const auto& [name, _] : mesh->get_boundary_tags()) {
            mesh->set_boundary_condition_euler(name, shear_bc);
        }
        mesh->build_precomputed_faces();
        for (const auto& [name, _] : mesh->get_boundary_tags()) {
            mesh->set_boundary_condition_euler(name, shear_bc);
        }

        // Time integration settings
        constexpr double dt = 1e-5;
        constexpr double T_final = 0.02;
        constexpr int save_every = 20;

        std::cout << "\n--- Solving viscous shear flow ---" << std::endl;
        std::cout << "  dt = " << dt << ", T_final = " << T_final << std::endl;

        dgfem::Timer solve_timer("Shear flow solve");
        dgfem::NavierStokesDGSolver solver(mesh, gamma, mu, prandtl, penalty);
        auto solutions = solver.solve([exact](const dgfem::Vec2& x) { return exact.conserved(x); },
                                      T_final, dt, save_every);

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
        std::filesystem::path output_dir("../../output");
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
