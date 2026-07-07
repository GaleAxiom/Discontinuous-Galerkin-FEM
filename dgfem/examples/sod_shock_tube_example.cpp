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
#include "dgfem/solver/euler_eigensystem.hpp"
#include "dgfem/solver/minmod_reconstruction.hpp"
#include "dgfem/solver/troubled_cell_indicator_base.hpp"
#include "dgfem/utils/example_helpers.hpp"
#include "dgfem/utils/riemann_solver.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

using dgfem::ExactRiemannSolution;

constexpr double kGamma = 1.4;
constexpr double kRhoL = 1.0, kUL = 0.0, kPL = 1.0;
constexpr double kRhoR = 0.125, kUR = 0.0, kPR = 0.1;
constexpr double kX0 = 0.5;
constexpr double kDt = 1e-4;
constexpr double kTFinal = 0.1;
constexpr int kSaveEvery = 100;

enum class LimiterMode { None, OldAlwaysOnMinmod, IndicatorWeno };

struct CaseResult {
    std::array<double, 4> l2_rel{};
    std::array<double, 4> max_err{};
};

// Builds a fresh mesh, solves the Sod problem under the given limiter mode, writes VTK frames
// under a case-specific prefix, and returns the per-variable errors against the exact Riemann
// solution. Each case gets its own mesh/solver rather than reusing one, so the runs cannot leak
// state into each other.
CaseResult run_case(const ExactRiemannSolution& exact, LimiterMode mode, const std::string& label) {
    std::cout << "\n=== Case: " << label << " ===" << std::endl;

    // Long, thin quasi-1D domain: slip walls in y keep v == 0 exactly, far-field BCs on the
    // left/right are fixed at the undisturbed states (valid since no wave reaches x=0 or x=1
    // by T_final -- the fastest wave, the right shock, travels at ~1.75). The limiter (see
    // CompressibleDGSolverBase::set_limiter_enabled) requires order-1 quad elements.
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

    std::cout << "  dt = " << kDt << ", T_final = " << kTFinal << ", mode = " << label << std::endl;

    dgfem::Timer solve_timer("Sod shock tube solve (" + label + ")");
    dgfem::EulerDGSolver solver(mesh, kGamma);
    switch (mode) {
    case LimiterMode::None:
        break;
    case LimiterMode::OldAlwaysOnMinmod:
        solver.set_reconstruction_technique(
            std::make_shared<dgfem::AlwaysTroubledIndicator>(),
            std::make_shared<dgfem::MinmodReconstruction>(
                std::make_shared<dgfem::EulerXDirectionEigensystem>(4)));
        solver.set_limiter_enabled(true);
        break;
    case LimiterMode::IndicatorWeno:
        // Lazy default: PerssonPeraireIndicator + WenoReconstruction sized to n_vars.
        solver.set_limiter_enabled(true);
        break;
    }
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
        auto [l2_rel, max_err] =
            dgfem::compute_riemann_solution_error(mesh, final_sol, exact, kGamma, kTFinal, v);
        result.l2_rel[v] = l2_rel;
        result.max_err[v] = max_err;
        std::cout << "  " << std::setw(3) << var_names[v] << " | " << std::scientific
                  << std::setprecision(4) << std::setw(11) << l2_rel << " | " << std::setw(11)
                  << max_err << std::endl;
    }

    for (size_t i = 0; i < solutions.size(); ++i) {
        std::string filename = "../../output/sod_" + label + "_" + std::to_string(i);
        dgfem::VTKWriter::write_euler_solution(mesh, solutions[i], filename, kGamma,
                                               /*refinement=*/1, /*n_vars=*/4);
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

        CaseResult unlimited = run_case(exact, LimiterMode::None, "unlimited");
        CaseResult old_minmod = run_case(exact, LimiterMode::OldAlwaysOnMinmod, "old_minmod");
        CaseResult indicator_weno = run_case(exact, LimiterMode::IndicatorWeno, "indicator_weno");

        std::array<std::string, 4> var_names{"rho", "u", "v", "p"};
        std::cout << "\n=== Limiter comparison (errors vs exact Riemann solution) ===" << std::endl;
        std::cout << "  var |  L2 unlimited |  L2 old_minmod | L2 indicator_weno | "
                  << "max unlimited | max old_minmod | max indicator_weno" << std::endl;
        for (int v = 0; v < 4; ++v) {
            std::cout << "  " << std::setw(3) << var_names[v] << " | " << std::scientific
                      << std::setprecision(4) << std::setw(13) << unlimited.l2_rel[v] << " | "
                      << std::setw(13) << old_minmod.l2_rel[v] << " | " << std::setw(16)
                      << indicator_weno.l2_rel[v] << " | " << std::setw(12)
                      << unlimited.max_err[v] << " | " << std::setw(14) << old_minmod.max_err[v]
                      << " | " << std::setw(18) << indicator_weno.max_err[v] << std::endl;
        }
        std::cout
            << "\n  (unlimited: no shock-capturing, so Gibbs oscillations near the shock/contact\n"
            << "   are the dominant error source -- see the density-boundedness check in\n"
            << "   sod_shock_tube_test.cpp for a direct measure of that oscillation.\n"
            << "   old_minmod: plain/TVB characteristic minmod, always on -- its magnitude-based\n"
            << "   threshold cannot distinguish a genuine discontinuity from the small dispersive\n"
            << "   precursor unlimited high-order DG produces ahead of a true wavefront, which\n"
            << "   measurably hurts its accuracy here.\n"
            << "   indicator_weno: Persson-Peraire troubled-cell indicator + WENO reconstruction\n"
            << "   (the default) -- recovers most of old_minmod's lost accuracy while still\n"
            << "   substantially suppressing the oscillation unlimited produces.)" << std::endl;

        dgfem::MeshCreator::finalize_gmsh();
        std::cout << "\n=== Sod shock tube example COMPLETED ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
