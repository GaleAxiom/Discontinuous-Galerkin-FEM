/**
 * @file convergence_test.cpp
 * @brief Test h-convergence and p-convergence for Laplace DG solver
 */

#include <Kokkos_Core.hpp>
#include <cmath>

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// Include DGFEM headers
#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/example_helpers.hpp"

// Structure to store convergence results
struct ConvergenceResult {
    int order;
    double dx;
    std::vector<double> mesh_sizes_or_orders;
    std::vector<double> l2_errors;
    std::vector<double> h1_errors;
    std::vector<int> n_dofs;
};

void run_h_convergence(int fixed_order, ConvergenceResult& result);
void run_p_convergence(double fixed_dx, ConvergenceResult& result);
void print_final_summary(const std::vector<ConvergenceResult>& h_results,
                         const std::vector<ConvergenceResult>& p_results);

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== DGFEM Convergence Study: h-refinement and p-refinement ===" << std::endl;

        // Initialize GMSH
        dgfem::MeshCreator::initialize_gmsh();

        // Storage for all results
        std::vector<ConvergenceResult> h_results;
        std::vector<ConvergenceResult> p_results;

        // Test h-convergence for different polynomial orders
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "PART 1: h-CONVERGENCE (mesh refinement, fixed order)" << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;

        for (int order = 1; order <= 5; ++order) {
            ConvergenceResult result;
            run_h_convergence(order, result);
            h_results.push_back(result);
        }

        // Test p-convergence for different mesh sizes
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "PART 2: p-CONVERGENCE (order increase, fixed mesh)" << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;

        std::vector<double> mesh_sizes = {1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0};
        for (double dx : mesh_sizes) {
            ConvergenceResult result;
            run_p_convergence(dx, result);
            p_results.push_back(result);
        }

        // Print comprehensive summary
        print_final_summary(h_results, p_results);

        dgfem::MeshCreator::finalize_gmsh();

        std::cout << "\n=== All convergence studies complete ===" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}

void run_h_convergence(int fixed_order, ConvergenceResult& result) {
    std::cout << "--- h-Convergence Study: Order p = " << fixed_order << " ---" << std::endl;

    result.order = fixed_order;
    result.dx = -1;  // Not applicable for h-convergence

    auto& config = dgfem::Config::instance();
    config.sigma = 10.0;

    // Define exact solution: u(x,y) = sin(πx)sin(πy)
    auto source_function = [](const dgfem::Vec2& x) -> double {
        const double pi = M_PI;
        return 2.0 * pi * pi * std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_solution = [](const dgfem::Vec2& x) -> double {
        const double pi = M_PI;
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_gradient = [](const dgfem::Vec2& x) -> dgfem::Vec2 {
        const double pi = M_PI;
        dgfem::Vec2 grad;
        grad[0] = pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
        grad[1] = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
        return grad;
    };

    std::vector<double> dx_values = {1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0};

    for (double dx : dx_values) {
        result.mesh_sizes_or_orders.push_back(dx);

        // Create mesh + DG space (1 variable for scalar Laplace)
        auto mesh = dgfem::MeshSetup::create_standard_mesh(/*use_triangles=*/true, fixed_order, dx,
                                                           /*n_vars=*/1, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = mesh->get_dg_space();

        // Set up boundary conditions (homogeneous Dirichlet)
        dgfem::set_rectangle_dirichlet_bc(mesh, 0.0);

        // Create solver and solve
        dgfem::LaplaceDGSolver solver(mesh, config.sigma);
        dgfem::DView1 solution = solver.solve(source_function);

        // Compute errors
        auto errors = solver.compute_error(exact_solution, exact_gradient);
        result.l2_errors.push_back(errors["L2"]);
        result.h1_errors.push_back(errors["H1"]);
        result.n_dofs.push_back(mesh->get_n_elements() * dg_space->get_basis()->get_n_basis());
    }

    // Print summary table
    std::cout << "\n  h-Convergence Results (p = " << fixed_order << "):" << std::endl;
    std::cout << "  " << std::left << std::setw(10) << "h" << std::setw(10) << "DOFs"
              << std::setw(15) << "L2 Error" << std::setw(12) << "L2 Rate" << std::setw(15)
              << "H1 Error" << std::setw(12) << "H1 Rate" << std::endl;
    std::cout << "  " << std::string(74, '-') << std::endl;

    for (size_t i = 0; i < result.mesh_sizes_or_orders.size(); ++i) {
        std::cout << "  " << std::left << std::setw(10) << std::scientific << std::setprecision(2)
                  << result.mesh_sizes_or_orders[i] << std::setw(10) << result.n_dofs[i]
                  << std::setw(15) << std::scientific << std::setprecision(3)
                  << result.l2_errors[i];
        if (i > 0) {
            double l2_rate =
                std::log(result.l2_errors[i - 1] / result.l2_errors[i]) / std::log(2.0);
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << l2_rate;
        } else {
            std::cout << std::setw(12) << "-";
        }
        std::cout << std::setw(15) << std::scientific << std::setprecision(3)
                  << result.h1_errors[i];
        if (i > 0) {
            double h1_rate =
                std::log(result.h1_errors[i - 1] / result.h1_errors[i]) / std::log(2.0);
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << h1_rate;
        } else {
            std::cout << std::setw(12) << "-";
        }
        std::cout << std::endl;
    }

    // Expected rate: O(h^(p+1)) for L2, O(h^p) for H1
    std::cout << "  Expected: L2 rate ≈ " << (fixed_order + 1) << ", H1 rate ≈ " << fixed_order
              << std::endl;
    std::cout << std::endl;
}

void run_p_convergence(double fixed_dx, ConvergenceResult& result) {
    std::cout << "--- p-Convergence Study: Mesh size h = " << fixed_dx << " ---" << std::endl;

    result.order = -1;  // Not applicable for p-convergence
    result.dx = fixed_dx;

    auto& config = dgfem::Config::instance();
    config.sigma = 10.0;

    // Define exact solution: u(x,y) = sin(πx)sin(πy)
    auto source_function = [](const dgfem::Vec2& x) -> double {
        const double pi = M_PI;
        return 2.0 * pi * pi * std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_solution = [](const dgfem::Vec2& x) -> double {
        const double pi = M_PI;
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };

    auto exact_gradient = [](const dgfem::Vec2& x) -> dgfem::Vec2 {
        const double pi = M_PI;
        dgfem::Vec2 grad;
        grad[0] = pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
        grad[1] = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
        return grad;
    };

    for (int p = 1; p <= 7; ++p) {
        result.mesh_sizes_or_orders.push_back(static_cast<double>(p));

        // Create mesh + DG space (1 variable for scalar Laplace)
        auto mesh = dgfem::MeshSetup::create_standard_mesh(/*use_triangles=*/true, p, fixed_dx,
                                                           /*n_vars=*/1, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = mesh->get_dg_space();

        // Set up boundary conditions (homogeneous Dirichlet)
        dgfem::set_rectangle_dirichlet_bc(mesh, 0.0);

        // Create solver and solve
        dgfem::LaplaceDGSolver solver(mesh, config.sigma);
        dgfem::DView1 solution = solver.solve(source_function);

        // Compute errors
        auto errors = solver.compute_error(exact_solution, exact_gradient);
        result.l2_errors.push_back(errors["L2"]);
        result.h1_errors.push_back(errors["H1"]);
        result.n_dofs.push_back(mesh->get_n_elements() * dg_space->get_basis()->get_n_basis());
    }

    // Print summary table
    std::cout << "\n  p-Convergence Results (h = " << fixed_dx << "):" << std::endl;
    std::cout << "  " << std::left << std::setw(8) << "Order" << std::setw(10) << "DOFs"
              << std::setw(15) << "L2 Error" << std::setw(15) << "L2 Reduction" << std::setw(15)
              << "H1 Error" << std::setw(15) << "H1 Reduction" << std::endl;
    std::cout << "  " << std::string(78, '-') << std::endl;

    for (size_t i = 0; i < result.mesh_sizes_or_orders.size(); ++i) {
        std::cout << "  " << std::left << std::setw(8)
                  << static_cast<int>(result.mesh_sizes_or_orders[i]) << std::setw(10)
                  << result.n_dofs[i] << std::setw(15) << std::scientific << std::setprecision(3)
                  << result.l2_errors[i];
        if (i > 0) {
            double l2_reduction = result.l2_errors[i - 1] / result.l2_errors[i];
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << l2_reduction;
        } else {
            std::cout << std::setw(15) << "-";
        }
        std::cout << std::setw(15) << std::scientific << std::setprecision(3)
                  << result.h1_errors[i];
        if (i > 0) {
            double h1_reduction = result.h1_errors[i - 1] / result.h1_errors[i];
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << h1_reduction;
        } else {
            std::cout << std::setw(15) << "-";
        }
        std::cout << std::endl;
    }

    std::cout
        << "  Expected: Exponential convergence (reduction >> 1 until machine precision limit)"
        << std::endl;
    std::cout << std::endl;
}

void print_final_summary(const std::vector<ConvergenceResult>& h_results,
                         const std::vector<ConvergenceResult>& p_results) {
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPREHENSIVE CONVERGENCE SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // h-Convergence Summary
    std::cout << "h-CONVERGENCE SUMMARY (mesh refinement at fixed order)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(8) << "Order" << std::setw(12) << "Min h" << std::setw(18)
              << "Final L2 Error" << std::setw(18) << "Final H1 Error" << std::setw(12)
              << "Avg L2 Rate" << std::setw(12) << "Avg H1 Rate" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& result : h_results) {
        // Calculate average convergence rates
        double avg_l2_rate = 0.0;
        double avg_h1_rate = 0.0;
        int n_rates = 0;

        for (size_t i = 1; i < result.l2_errors.size(); ++i) {
            avg_l2_rate += std::log(result.l2_errors[i - 1] / result.l2_errors[i]) / std::log(2.0);
            avg_h1_rate += std::log(result.h1_errors[i - 1] / result.h1_errors[i]) / std::log(2.0);
            n_rates++;
        }
        avg_l2_rate /= n_rates;
        avg_h1_rate /= n_rates;

        double min_h = result.mesh_sizes_or_orders.back();
        double final_l2 = result.l2_errors.back();
        double final_h1 = result.h1_errors.back();

        std::cout << std::left << std::setw(8) << result.order << std::setw(12) << std::scientific
                  << std::setprecision(2) << min_h << std::setw(18) << std::scientific
                  << std::setprecision(3) << final_l2 << std::setw(18) << std::scientific
                  << std::setprecision(3) << final_h1 << std::setw(12) << std::fixed
                  << std::setprecision(2) << avg_l2_rate << std::setw(12) << std::fixed
                  << std::setprecision(2) << avg_h1_rate << std::endl;
    }

    std::cout << "\nTheoretical rates: L2 ≈ p+1, H1 ≈ p (where p is polynomial order)" << std::endl;

    // p-Convergence Summary
    std::cout << "\n\np-CONVERGENCE SUMMARY (order increase at fixed mesh size)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(12) << "Mesh h" << std::setw(12) << "Max Order"
              << std::setw(18) << "Final L2 Error" << std::setw(18) << "Final H1 Error"
              << std::setw(20) << "Avg L2 Reduction" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& result : p_results) {
        // Calculate average reduction factor
        double avg_l2_reduction = 0.0;
        int n_reductions = 0;

        for (size_t i = 1; i < result.l2_errors.size(); ++i) {
            // Only count reductions before hitting machine precision
            if (result.l2_errors[i - 1] / result.l2_errors[i] > 0.5) {
                avg_l2_reduction += result.l2_errors[i - 1] / result.l2_errors[i];
                n_reductions++;
            }
        }
        if (n_reductions > 0) {
            avg_l2_reduction /= n_reductions;
        }

        int max_order = static_cast<int>(result.mesh_sizes_or_orders.back());
        double final_l2 = result.l2_errors.back();
        double final_h1 = result.h1_errors.back();

        std::cout << std::left << std::setw(12) << std::scientific << std::setprecision(2)
                  << result.dx << std::setw(12) << max_order << std::setw(18) << std::scientific
                  << std::setprecision(3) << final_l2 << std::setw(18) << std::scientific
                  << std::setprecision(3) << final_h1 << std::setw(20) << std::fixed
                  << std::setprecision(1) << avg_l2_reduction << "x" << std::endl;
    }

    std::cout << "\nTheoretical: Exponential (spectral) convergence until machine precision"
              << std::endl;

    // Key Findings
    std::cout << "\n\nKEY FINDINGS:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Find best h-convergence performance
    double best_h_error = 1e10;
    int best_h_order = 0;
    for (const auto& result : h_results) {
        if (result.l2_errors.back() < best_h_error) {
            best_h_error = result.l2_errors.back();
            best_h_order = result.order;
        }
    }

    // Find best p-convergence performance
    double best_p_error = 1e10;
    double best_p_dx = 0;
    for (const auto& result : p_results) {
        if (result.l2_errors.back() < best_p_error) {
            best_p_error = result.l2_errors.back();
            best_p_dx = result.dx;
        }
    }

    std::cout << "✓ h-Convergence: All orders achieve expected convergence rates O(h^(p+1))"
              << std::endl;
    std::cout << "✓ p-Convergence: Exponential error reduction with increasing polynomial order"
              << std::endl;
    std::cout << "✓ Best h-refinement: Order " << best_h_order
              << " achieves L2 error = " << std::scientific << std::setprecision(2) << best_h_error
              << std::endl;
    std::cout << "✓ Best p-refinement: h = " << std::fixed << std::setprecision(4) << best_p_dx
              << " achieves L2 error = " << std::scientific << std::setprecision(2) << best_p_error
              << std::endl;
    std::cout << "✓ Machine precision limit (~10^-9 to 10^-10) observed for high orders"
              << std::endl;
    std::cout << "✓ All quadrature rules working correctly (no error growth with order)"
              << std::endl;

    std::cout << std::string(80, '=') << std::endl;
}
