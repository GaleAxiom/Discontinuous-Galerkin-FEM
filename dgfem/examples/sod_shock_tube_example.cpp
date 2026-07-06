/**
 * @file sod_shock_tube_example.cpp
 * @brief Sod shock tube: inviscid Euler solver against the exact Riemann solution.
 *
 * Every other Euler/Navier-Stokes example in this directory (acoustic wave,
 * Couette/shear flow) uses a smooth manufactured solution. The Sod problem is
 * different: it starts from a genuine discontinuity (rho_L=1, p_L=1 vs.
 * rho_R=0.125, p_R=0.1, both at rest, gamma=1.4) and develops a left
 * rarefaction fan, a contact discontinuity, and a right shock. This solver
 * has no slope limiter or shock-capturing beyond the Rusanov numerical flux
 * (see dgfem/include/dgfem/solver/time_stepping.hpp and grep results across
 * the codebase -- there is no TVD/WENO/limiter machinery), so this example is
 * as much a robustness check as an accuracy one: does an unlimited high-order
 * DG scheme survive a real shock at all, and how large are the Gibbs
 * oscillations it produces near the discontinuities?
 *
 * The 2D solver is exercised in a quasi-1D setup: a long, thin rectangular
 * domain with slip walls on top/bottom (so v stays exactly 0 and the problem
 * reduces to the classical 1D Riemann problem) and far-field boundaries on
 * the left/right fixed at the undisturbed left/right states -- valid because
 * the domain is long enough that no wave reaches those boundaries by
 * T_final.
 *
 * The exact solution is computed in closed form via the standard iterative
 * Riemann solver (Newton's method on the pressure function; see e.g. Toro,
 * "Riemann Solvers and Numerical Methods for Fluid Dynamics", ch. 4).
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

/**
 * @brief Exact solution of the 1D Riemann problem (Toro's iterative solver).
 */
struct ExactRiemannSolution {
    double gamma;
    double rho_L, u_L, p_L, c_L;
    double rho_R, u_R, p_R, c_R;
    double x0;
    double p_star = 0.0;
    double u_star = 0.0;

    ExactRiemannSolution(double gamma_, double rho_L_, double u_L_, double p_L_, double rho_R_,
                         double u_R_, double p_R_, double x0_)
        : gamma(gamma_), rho_L(rho_L_), u_L(u_L_), p_L(p_L_), rho_R(rho_R_), u_R(u_R_), p_R(p_R_),
          x0(x0_) {
        c_L = std::sqrt(gamma * p_L / rho_L);
        c_R = std::sqrt(gamma * p_R / rho_R);
        solve_star_state();
    }

    [[nodiscard]] double f_branch(double p, double rho_K, double p_K, double c_K) const {
        if (p > p_K) {
            double a_k = 2.0 / ((gamma + 1.0) * rho_K);
            double b_k = (gamma - 1.0) / (gamma + 1.0) * p_K;
            return (p - p_K) * std::sqrt(a_k / (p + b_k));
        }
        return (2.0 * c_K / (gamma - 1.0)) *
               (std::pow(p / p_K, (gamma - 1.0) / (2.0 * gamma)) - 1.0);
    }

    [[nodiscard]] double f_branch_deriv(double p, double rho_K, double p_K, double c_K) const {
        if (p > p_K) {
            double a_k = 2.0 / ((gamma + 1.0) * rho_K);
            double b_k = (gamma - 1.0) / (gamma + 1.0) * p_K;
            return std::sqrt(a_k / (b_k + p)) * (1.0 - (p - p_K) / (2.0 * (b_k + p)));
        }
        return (1.0 / (rho_K * c_K)) * std::pow(p / p_K, -(gamma + 1.0) / (2.0 * gamma));
    }

    [[nodiscard]] double total_f(double p) const {
        return f_branch(p, rho_L, p_L, c_L) + f_branch(p, rho_R, p_R, c_R) + (u_R - u_L);
    }

    [[nodiscard]] double total_f_deriv(double p) const {
        return f_branch_deriv(p, rho_L, p_L, c_L) + f_branch_deriv(p, rho_R, p_R, c_R);
    }

    void solve_star_state() {
        double p = 0.5 * (p_L + p_R);
        for (int iter = 0; iter < 50; ++iter) {
            double f = total_f(p);
            double fp = total_f_deriv(p);
            double p_new = p - f / fp;
            if (p_new < 1e-8) {
                p_new = 1e-8;
            }
            if (std::abs(p_new - p) < 1e-12 * std::abs(p_new)) {
                p = p_new;
                break;
            }
            p = p_new;
        }
        p_star = p;
        u_star = 0.5 * (u_L + u_R) +
                 0.5 * (f_branch(p_star, rho_R, p_R, c_R) - f_branch(p_star, rho_L, p_L, c_L));
    }

    // Returns (rho, u, p) at physical position x and time t > 0.
    [[nodiscard]] std::array<double, 3> sample(double x, double t) const {
        double xi = (x - x0) / std::max(t, 1e-12);

        if (xi <= u_star) {
            // Left of the contact discontinuity.
            if (p_star <= p_L) {
                // Left rarefaction fan.
                double c_star_L = c_L * std::pow(p_star / p_L, (gamma - 1.0) / (2.0 * gamma));
                double s_head = u_L - c_L;
                double s_tail = u_star - c_star_L;
                if (xi <= s_head) {
                    return {rho_L, u_L, p_L};
                }
                if (xi <= s_tail) {
                    double c =
                        ((gamma - 1.0) / (gamma + 1.0)) * (u_L + 2.0 * c_L / (gamma - 1.0) - xi);
                    double u = (2.0 / (gamma + 1.0)) * (c_L + 0.5 * (gamma - 1.0) * u_L + xi);
                    double rho = rho_L * std::pow(c / c_L, 2.0 / (gamma - 1.0));
                    double p = p_L * std::pow(c / c_L, 2.0 * gamma / (gamma - 1.0));
                    return {rho, u, p};
                }
                double rho_star_L = rho_L * std::pow(p_star / p_L, 1.0 / gamma);
                return {rho_star_L, u_star, p_star};
            }
            // Left shock.
            double rho_star_L = rho_L * ((p_star / p_L) + (gamma - 1.0) / (gamma + 1.0)) /
                                ((gamma - 1.0) / (gamma + 1.0) * (p_star / p_L) + 1.0);
            double s_shock = u_L - c_L * std::sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) +
                                                   (gamma - 1.0) / (2.0 * gamma));
            if (xi <= s_shock) {
                return {rho_L, u_L, p_L};
            }
            return {rho_star_L, u_star, p_star};
        }

        // Right of the contact discontinuity.
        if (p_star >= p_R) {
            // Right shock.
            double rho_star_R = rho_R * ((p_star / p_R) + (gamma - 1.0) / (gamma + 1.0)) /
                                ((gamma - 1.0) / (gamma + 1.0) * (p_star / p_R) + 1.0);
            double s_shock = u_R + c_R * std::sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) +
                                                   (gamma - 1.0) / (2.0 * gamma));
            if (xi >= s_shock) {
                return {rho_R, u_R, p_R};
            }
            return {rho_star_R, u_star, p_star};
        }
        // Right rarefaction fan.
        double c_star_R = c_R * std::pow(p_star / p_R, (gamma - 1.0) / (2.0 * gamma));
        double s_head = u_R + c_R;
        double s_tail = u_star + c_star_R;
        if (xi >= s_head) {
            return {rho_R, u_R, p_R};
        }
        if (xi >= s_tail) {
            double c = (2.0 * c_R - (gamma - 1.0) * (u_R - xi)) / (gamma + 1.0);
            double u = (-2.0 * c_R + (gamma - 1.0) * u_R + 2.0 * xi) / (gamma + 1.0);
            double rho = rho_R * std::pow(c / c_R, 2.0 / (gamma - 1.0));
            double p = p_R * std::pow(c / c_R, 2.0 * gamma / (gamma - 1.0));
            return {rho, u, p};
        }
        double rho_star_R = rho_R * std::pow(p_star / p_R, 1.0 / gamma);
        return {rho_star_R, u_star, p_star};
    }
};

/**
 * @brief L2 relative error and max pointwise error for one primitive variable at time t.
 */
std::pair<double, double> compute_variable_error(const std::shared_ptr<dgfem::DGMesh>& mesh,
                                                 const dgfem::DView2& numerical_sol,
                                                 const ExactRiemannSolution& exact, double gamma,
                                                 double t, int var_idx) {
    auto space = mesh->get_dg_space();
    auto mapping = space->get_mapping();
    const auto& phi = space->get_volume_basis_values();
    const auto& quad_pts = space->get_volume_quad()->points;
    const auto& quad_wts = space->get_volume_quad()->weights;

    int n_basis = space->get_basis()->get_n_basis();
    double error_sq = 0.0;
    double norm_sq = 0.0;
    double max_error = 0.0;

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

            auto [rho_ex, u_ex, p_ex] = exact.sample(x_phys[0], t);
            std::array<double, 4> w_exact{rho_ex, u_ex, 0.0, p_ex};

            double weight = quad_wts(q) * std::abs(J_det(q, 0));
            double diff = W_num[var_idx] - w_exact[var_idx];
            error_sq += diff * diff * weight;
            norm_sq += w_exact[var_idx] * w_exact[var_idx] * weight;
            max_error = std::max(max_error, std::abs(diff));
        }
    }

    double l2_rel = (norm_sq < 1e-14) ? std::sqrt(error_sq) : std::sqrt(error_sq / norm_sq);
    return {l2_rel, max_error};
}

}  // namespace

namespace {

constexpr double kGamma = 1.4;
constexpr double kRhoL = 1.0, kUL = 0.0, kPL = 1.0;
constexpr double kRhoR = 0.125, kUR = 0.0, kPR = 0.1;
constexpr double kX0 = 0.5;
constexpr double kDt = 1e-4;
constexpr double kTFinal = 0.1;
constexpr int kSaveEvery = 100;

struct CaseResult {
    std::array<double, 4> l2_rel{};
    std::array<double, 4> max_err{};
};

// Builds a fresh mesh, solves the Sod problem with or without the minmod limiter, writes VTK
// frames under a case-specific prefix, and returns the per-variable errors against the exact
// Riemann solution. Each case gets its own mesh/solver rather than reusing one, so the two
// runs cannot leak state into each other.
CaseResult run_case(const ExactRiemannSolution& exact, bool use_limiter, const std::string& label) {
    std::cout << "\n=== Case: " << label << " ===" << std::endl;

    // Long, thin quasi-1D domain: slip walls in y keep v == 0 exactly, far-field BCs on the
    // left/right are fixed at the undisturbed states (valid since no wave reaches x=0 or x=1
    // by T_final -- the fastest wave, the right shock, travels at ~1.75). The minmod limiter
    // (see CompressibleDGSolverBase::set_limiter_enabled) requires order-1 quad elements.
    auto mesh = dgfem::MeshSetup::create_standard_mesh(
        /*use_triangles=*/false,
        /*order=*/1,
        /*dx=*/0.01,
        /*n_vars=*/4,
        /*xmin=*/0.0,
        /*xmax=*/1.0,
        /*ymin=*/0.0,
        /*ymax=*/0.02);
    dgfem::MeshSetup::print_info(mesh);

    dgfem::Vec4 left_conserved =
        dgfem::primitive_to_conserved(dgfem::Vec4{kRhoL, kUL, 0.0, kPL}, kGamma);
    dgfem::Vec4 right_conserved =
        dgfem::primitive_to_conserved(dgfem::Vec4{kRhoR, kUR, 0.0, kPR}, kGamma);

    auto left_bc = std::make_shared<dgfem::BoundaryConditionEuler>(dgfem::BCTypeEuler::FAR_FIELD,
                                                                   left_conserved);
    auto right_bc = std::make_shared<dgfem::BoundaryConditionEuler>(dgfem::BCTypeEuler::FAR_FIELD,
                                                                    right_conserved);
    // SLIP_WALL reflects the interior state's normal velocity and ignores the stored value
    // entirely (see EulerWeakFormulation::boundary_face_residual), so any placeholder works.
    auto slip_bc = std::make_shared<dgfem::BoundaryConditionEuler>(dgfem::BCTypeEuler::SLIP_WALL,
                                                                   dgfem::Vec4{1.0, 0.0, 0.0, 1.0});

    mesh->set_boundary_condition_euler("Left", left_bc);
    mesh->set_boundary_condition_euler("Right", right_bc);
    mesh->set_boundary_condition_euler("Bottom", slip_bc);
    mesh->set_boundary_condition_euler("Top", slip_bc);

    auto initial_condition = [](const dgfem::Vec2& x) -> dgfem::Vec4 {
        dgfem::Vec4 primitive =
            (x[0] < kX0) ? dgfem::Vec4{kRhoL, kUL, 0.0, kPL} : dgfem::Vec4{kRhoR, kUR, 0.0, kPR};
        return dgfem::primitive_to_conserved(primitive, kGamma);
    };

    std::cout << "  dt = " << kDt << ", T_final = " << kTFinal
              << ", limiter = " << (use_limiter ? "ON" : "OFF") << std::endl;

    dgfem::Timer solve_timer("Sod shock tube solve (" + label + ")");
    dgfem::EulerDGSolver solver(mesh, kGamma);
    solver.set_limiter_enabled(use_limiter);
    auto solutions = solver.solve(initial_condition, kTFinal, kDt, kSaveEvery);

    if (solutions.empty()) {
        throw std::runtime_error("Euler solver did not return any solution frames.");
    }

    const auto& final_sol = solutions.back();
    std::array<std::string, 4> var_names{"rho", "u", "v", "p"};

    CaseResult result;
    std::cout << "\n  --- Errors vs exact Riemann solution at T_final ---" << std::endl;
    std::cout << "  var |    L2 rel   |  max pointwise" << std::endl;
    for (int v = 0; v < 4; ++v) {
        auto [l2_rel, max_err] = compute_variable_error(mesh, final_sol, exact, kGamma, kTFinal, v);
        result.l2_rel[v] = l2_rel;
        result.max_err[v] = max_err;
        std::cout << "  " << std::setw(3) << var_names[v] << " | " << std::scientific
                  << std::setprecision(4) << std::setw(11) << l2_rel << " | " << std::setw(11)
                  << max_err << std::endl;
    }

    for (size_t i = 0; i < solutions.size(); ++i) {
        std::string filename = "../../output/sod_" + label + "_" + std::to_string(i);
        dgfem::VTKWriter::write_euler_solution(mesh, solutions[i], filename, kGamma,
                                               /*refinement=*/1);
    }

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== DGFEM Sod Shock Tube Example ===" << std::endl;

        ExactRiemannSolution exact(kGamma, kRhoL, kUL, kPL, kRhoR, kUR, kPR, kX0);
        std::cout << "\n--- Exact Riemann star state ---" << std::endl;
        std::cout << "  p*      = " << exact.p_star << " (published reference: 0.30313)"
                  << std::endl;
        std::cout << "  u*      = " << exact.u_star << " (published reference: 0.92745)"
                  << std::endl;

        CaseResult unlimited = run_case(exact, /*use_limiter=*/false, "unlimited");
        CaseResult limited = run_case(exact, /*use_limiter=*/true, "limited");

        std::array<std::string, 4> var_names{"rho", "u", "v", "p"};
        std::cout << "\n=== Limiter comparison (errors vs exact Riemann solution) ===" << std::endl;
        std::cout << "  var |  L2 unlimited |  L2 limited  | max unlimited | max limited"
                  << std::endl;
        for (int v = 0; v < 4; ++v) {
            std::cout << "  " << std::setw(3) << var_names[v] << " | " << std::scientific
                      << std::setprecision(4) << std::setw(13) << unlimited.l2_rel[v] << " | "
                      << std::setw(12) << limited.l2_rel[v] << " | " << std::setw(13)
                      << unlimited.max_err[v] << " | " << std::setw(11) << limited.max_err[v]
                      << std::endl;
        }
        std::cout << "\n  (The unlimited run's nonzero error is expected: this solver has no "
                  << "shock-capturing\n   limiter by default, so Gibbs oscillations near the "
                  << "shock and contact discontinuity\n   are the dominant error source. The "
                  << "limited run should show smaller max pointwise\n   error at the cost of "
                  << "some smearing -- the classic monotonicity/accuracy trade-off.)" << std::endl;

        dgfem::MeshCreator::finalize_gmsh();
        std::cout << "\n=== Sod shock tube example COMPLETED ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
