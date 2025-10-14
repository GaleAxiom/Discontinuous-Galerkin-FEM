/**
 * @file taylor_green_test.cpp
 * @brief Test Euler solver with Taylor-Green vortex benchmark
 *
 * The Taylor-Green vortex is a standard test case for incompressible/low-Mach flow.
 * For inviscid Euler at low Mach number, it provides an analytical benchmark.
 */

#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <cmath>

#include <iomanip>
#include <iostream>

/**
 * @brief Taylor-Green vortex analytical solution
 */
struct TaylorGreenVortex {
    double rho0, U0, kx, ky, nu, gamma, p0;

    TaylorGreenVortex(double rho0_, double U0_, double kx_, double ky_, double nu_, double gamma_,
                      double p0_)
        : rho0(rho0_), U0(U0_), kx(kx_), ky(ky_), nu(nu_), gamma(gamma_), p0(p0_) {}

    // Primitive variables [rho, u, v, p] at position (x, y) and time t
    Eigen::Vector4d operator()(double x, double y, double t) const {
        double decay = std::exp(-2.0 * nu * (kx * kx + ky * ky) * t);
        double u = -U0 * std::cos(kx * x) * std::sin(ky * y) * decay;
        double v = U0 * std::sin(kx * x) * std::cos(ky * y) * decay;

        double p_dyn = -rho0 * U0 * U0 / 4.0 * (std::cos(2.0 * kx * x) + std::cos(2.0 * ky * y)) *
                       std::exp(-4.0 * nu * (kx * kx + ky * ky) * t);
        double p = p0 + p_dyn;
        double rho = rho0 * std::pow(p / p0, 1.0 / gamma);  // Isentropic

        return dgfem::primitive_to_conserved(Eigen::Vector4d(rho, u, v, p), gamma);
    }
};

/**
 * @brief Compute L2 error for a specific variable
 */
double compute_variable_error(const std::shared_ptr<dgfem::DGMesh>& mesh,
                              const Eigen::MatrixXd& numerical_sol, const TaylorGreenVortex& exact,
                              double time, double gamma, int var_idx) {
    auto dg_space = mesh->get_dg_space();
    auto mapping = dg_space->get_mapping();
    const auto& phi = dg_space->get_volume_basis_values();
    const auto& quad_pts = dg_space->get_volume_quad()->points;
    const auto& quad_wts = dg_space->get_volume_quad()->weights;

    double error_sq = 0.0, norm_sq = 0.0;
    int n_basis = dg_space->get_basis()->get_n_basis();

    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        const auto& J_det = mesh->get_element_data(e).at("J_det_vol");

        Eigen::MatrixXd verts(mesh->get_elements().cols(), 2);
        for (int i = 0; i < verts.rows(); ++i)
            verts.row(i) = mesh->get_vertices().row(mesh->get_elements()(e, i));

        for (int q = 0; q < quad_wts.size(); ++q) {
            Eigen::Vector2d x_phys = mapping->map_to_physical(verts, quad_pts.row(q));

            // Numerical solution at quad point
            Eigen::Vector4d U_num = Eigen::Vector4d::Zero();
            for (int i = 0; i < n_basis; ++i)
                for (int v = 0; v < 4; ++v)
                    U_num[v] += numerical_sol(e, i * 4 + v) * phi(q, i);

            Eigen::Vector4d U_exact = exact(x_phys[0], x_phys[1], time);
            Eigen::Vector4d W_num = dgfem::conserved_to_primitive(U_num, gamma);
            Eigen::Vector4d W_exact = dgfem::conserved_to_primitive(U_exact, gamma);

            double dw = quad_wts(q) * std::abs(J_det(q));
            error_sq +=
                (W_num[var_idx] - W_exact[var_idx]) * (W_num[var_idx] - W_exact[var_idx]) * dw;
            norm_sq += W_exact[var_idx] * W_exact[var_idx] * dw;
        }
    }

    return std::sqrt(error_sq / norm_sq);
}

int main() {
    try {
        std::cout << "=== DGFEM Taylor-Green Vortex Test ===" << std::endl;

        // Physics setup
        constexpr double gamma = 1.4;
        constexpr double L = 2.0 * M_PI;
        const double rho0 = 1.0, p0 = 1.0 / gamma, U0 = 0.01;
        const double k = 2.0 * M_PI / L;
        const double Mach = U0 / std::sqrt(gamma * p0 / rho0);

        std::cout << "  Mach number: " << Mach << " (low-Mach regime)" << std::endl;

        // Create mesh (triangles, order 3, spacing 0.1, 4 vars, domain [0,2π]x[0,2π])
        auto mesh = dgfem::MeshSetup::create_standard_mesh(true, 3, 0.1, 4, 0.0, L, 0.0, L);
        dgfem::MeshSetup::print_info(mesh);

        // Periodic boundaries
        mesh->set_periodic_boundaries("Left", "Right");
        mesh->set_periodic_boundaries("Bottom", "Top");
        std::cout << "  Boundary conditions: PERIODIC" << std::endl;

        // Analytical solution (inviscid: nu=0)
        TaylorGreenVortex exact(rho0, U0, k, k, 0.0, gamma, p0);
        auto ic = [&exact](const Eigen::Vector2d& x) { return exact(x[0], x[1], 0.0); };

        // Time stepping
        constexpr double dt = 0.001, T_final = 1.0;
        constexpr int save_every = 100;

        std::cout << "\n--- Solving ---" << std::endl;
        std::cout << "  T_final = " << T_final << ", dt = " << dt << std::endl;

        dgfem::Timer timer("Taylor-Green solve");
        dgfem::EulerDGSolver solver(mesh, gamma);
        auto solutions = solver.solve(ic, T_final, dt, save_every);

        // Error analysis
        std::cout << "\n--- Error Analysis (t=" << T_final << ") ---" << std::endl;
        const auto& final_sol = solutions.back();
        std::array<std::string, 4> var_names = {"rho", "u", "v", "p"};

        for (int i = 0; i < 4; ++i) {
            double err = compute_variable_error(mesh, final_sol, exact, T_final, gamma, i);
            std::cout << "  L2 error (" << std::setw(3) << var_names[i] << "): " << std::scientific
                      << std::setprecision(4) << err << std::endl;
        }

        // Export
        std::cout << "\n--- Exporting " << solutions.size() << " frames ---" << std::endl;
        for (size_t i = 0; i < solutions.size(); ++i) {
            dgfem::VTKWriter::write_euler_solution(
                mesh, solutions[i], "../../output/taylor_green_" + std::to_string(i), gamma, 2);
            if (i % 5 == 0 || i == solutions.size() - 1)
                std::cout << "  Frame " << i << std::endl;
        }

        dgfem::MeshCreator::finalize_gmsh();
        std::cout << "\n=== Test COMPLETED ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
