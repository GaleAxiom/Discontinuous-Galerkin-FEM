/**
 * @file smooth_advection_limiter_test.cpp
 * @brief Isolates whether WenoReconstruction's smoothness-weighted blend actually beats plain
 * minmod, independent of the Persson-Peraire troubled-cell indicator's cell selectivity.
 *
 * sod_shock_tube_test.cpp's (now-deleted) WENO-vs-minmod probe found that, on the Sod shock
 * tube, pairing the SAME indicator with minmod instead of WENO gave essentially identical (if
 * not slightly better) results. That comparison couldn't actually distinguish "WENO's blend is
 * better" from "the indicator alone is doing all the work", because on that problem the
 * indicator only ever flags the shock/contact cells -- both reconstruction techniques are only
 * ever invoked on genuinely discontinuous data there, where a P1 basis gives them little room
 * to differ.
 *
 * This test removes that confound with AlwaysTroubledIndicator, forcing every cell to be
 * reconstructed every RK stage regardless of whether it is actually discontinuous. On a smooth
 * solution, that isolates the reconstruction technique's own behavior. The specific scenario is
 * a smooth density "entropy wave" advecting at constant velocity: with u = u0 and p = p0 held
 * exactly constant everywhere, ANY rho(x,t) satisfying the linear advection equation
 * d(rho)/dt + u0 d(rho)/dx = 0 is an EXACT solution of the full nonlinear Euler equations
 * (substitute into the momentum and energy equations -- both reduce to the same scalar
 * advection equation as continuity, since rho*u and E are then affine in rho alone). That gives
 * a closed-form reference solution -- a raised-cosine bump advecting rigidly at speed u0 --
 * without needing a Riemann solver or a high-resolution reference run.
 *
 * The classical flaw of plain (non-TVB) minmod is that it clips genuine smooth extrema: at a
 * local max of the cell-average sequence, the forward and backward differences have opposite
 * sign, so minmod3 returns exactly 0 regardless of how smooth the underlying data actually is --
 * eroding a smooth peak toward piecewise-constant under repeated application. WENO's
 * smoothness-weighted blend has no such hard cutoff. Forced onto every cell for many steps, this
 * difference should be directly measurable here even though it wasn't on the Sod problem.
 */

#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/solver/euler_eigensystem.hpp>
#include <dgfem/solver/minmod_reconstruction.hpp>
#include <dgfem/solver/troubled_cell_indicator_base.hpp>
#include <dgfem/solver/weno_reconstruction.hpp>
#include <dgfem/utils/example_helpers.hpp>

#include <cmath>

#include <memory>
#include <utility>

#include <gmsh.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

namespace {

constexpr double kGamma = 1.4;
constexpr double kRho0 = 1.0, kU0 = 1.0, kP0 = 1.0;
constexpr double kBumpAmplitude = 0.5;
constexpr double kBumpHalfWidth = 0.3;
constexpr double kX0 = 1.0;
constexpr double kXMin = 0.0, kXMax = 2.5;
constexpr double kDx = 0.02;
constexpr double kDt = 1e-3;
constexpr double kTFinal = 0.5;
constexpr int kSaveEvery = 1000;  // only need the final frame

// Raised-cosine (Hann) bump: smooth (C^1), compactly supported on |s| < 1, zero value AND zero
// derivative at the edges so it splices continuously into the constant background.
double bump(double s) {
    if (std::abs(s) >= 1.0) {
        return 0.0;
    }
    return 0.5 * (1.0 + std::cos(M_PI * s));
}

double exact_rho(double x, double t) {
    return kRho0 + kBumpAmplitude * bump((x - kX0 - kU0 * t) / kBumpHalfWidth);
}

enum class ReconMode { None, AlwaysMinmodPure, AlwaysWeno };

struct CaseErrors {
    double l2_rho, max_rho;
};

template <typename Fn>
void run_advection_solve(ReconMode mode, Fn&& consume) {
    try {
        gmsh::clear();
        gmsh::finalize();
    } catch (...) {
    }
    gmsh::initialize();

    // Same quasi-1D trick as sod_shock_tube_test.cpp: thin strip, slip walls in y keep v == 0
    // exactly. Far-field BCs are fixed at the (spatially uniform) background state, valid
    // because the bump stays well clear of both boundaries for the whole run (starts at x=1.0,
    // half-width 0.3, advects to x=1.5 by T_final=0.5, domain is [0, 2.5]).
    auto mesh = MeshSetup::create_standard_mesh(
        /*use_triangles=*/false,
        /*order=*/1,
        /*dx=*/kDx,
        /*n_vars=*/4,
        /*xmin=*/kXMin,
        /*xmax=*/kXMax,
        /*ymin=*/0.0,
        /*ymax=*/0.02);

    Vec4 background_conserved = primitive_to_conserved(Vec4{kRho0, kU0, 0.0, kP0}, kGamma);
    auto far_field_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, background_conserved);
    auto slip_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL, Vec4{1.0, 0.0, 0.0, 1.0});
    mesh->set_boundary_condition_euler("Left", far_field_bc);
    mesh->set_boundary_condition_euler("Right", far_field_bc);
    mesh->set_boundary_condition_euler("Bottom", slip_bc);
    mesh->set_boundary_condition_euler("Top", slip_bc);

    auto initial_condition = [](const Vec2& x) -> Vec4 {
        Vec4 primitive{exact_rho(x[0], 0.0), kU0, 0.0, kP0};
        return primitive_to_conserved(primitive, kGamma);
    };

    EulerDGSolver solver(mesh, kGamma);
    switch (mode) {
    case ReconMode::None:
        break;
    case ReconMode::AlwaysMinmodPure:
        // tvb_m = 0: plain, unmodified minmod -- no TVB magnitude threshold to (partially) shield
        // smooth extrema, so any clipping behavior is the limiter's own, not a tuning artifact.
        solver.set_reconstruction_technique(
            std::make_shared<AlwaysTroubledIndicator>(),
            std::make_shared<MinmodReconstruction>(std::make_shared<EulerXDirectionEigensystem>(4),
                                                    std::make_shared<Order1QuadBasisModeMap>(),
                                                    /*tvb_m=*/0.0));
        solver.set_limiter_enabled(true);
        break;
    case ReconMode::AlwaysWeno:
        solver.set_reconstruction_technique(
            std::make_shared<AlwaysTroubledIndicator>(),
            std::make_shared<WenoReconstruction>(std::make_shared<EulerXDirectionEigensystem>(4)));
        solver.set_limiter_enabled(true);
        break;
    }

    auto solutions = solver.solve(initial_condition, kTFinal, kDt, kSaveEvery);
    consume(mesh, solutions.back());

    try {
        gmsh::clear();
        gmsh::finalize();
    } catch (...) {
    }
}

CaseErrors run_advection_case(ReconMode mode) {
    CaseErrors result{};
    run_advection_solve(mode, [&](const std::shared_ptr<DGMesh>& mesh, const DView2& final_sol) {
        auto space = mesh->get_dg_space();
        const auto& phi = space->get_volume_basis_values();
        auto mapping = space->get_mapping();
        const auto& quad_pts = space->get_volume_quad()->points;
        const auto& quad_wts = space->get_volume_quad()->weights;
        int n_basis = space->get_basis()->get_n_basis();

        double error_sq = 0.0, norm_sq = 0.0, max_error = 0.0;
        for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
            const auto& elem_data = mesh->get_element_data(elem);
            const DView2& J_det = elem_data.at("J_det_vol");
            DView2 vertices = mesh->get_element_vertices(elem);

            for (int q = 0; q < static_cast<int>(quad_wts.size()); ++q) {
                Vec2 x_phys = mapping->map_to_physical(vertices, row2(quad_pts, q));
                double rho_num = 0.0;
                for (int i = 0; i < n_basis; ++i) {
                    rho_num += final_sol(elem, i * 4 + 0) * phi(q, i);
                }
                double rho_ex = exact_rho(x_phys[0], kTFinal);
                double weight = quad_wts(q) * std::abs(J_det(q, 0));
                double diff = rho_num - rho_ex;
                error_sq += diff * diff * weight;
                norm_sq += rho_ex * rho_ex * weight;
                max_error = std::max(max_error, std::abs(diff));
            }
        }
        result.l2_rho = std::sqrt(error_sq / norm_sq);
        result.max_rho = max_error;
    });
    return result;
}

}  // namespace

TEST(SmoothAdvectionLimiterTest, WenoPreservesSmoothPeakBetterThanPlainMinmodUnderForcedLimiting) {
    CaseErrors unlimited = run_advection_case(ReconMode::None);
    CaseErrors always_minmod = run_advection_case(ReconMode::AlwaysMinmodPure);
    CaseErrors always_weno = run_advection_case(ReconMode::AlwaysWeno);

    // Unlimited reproduces the exact smooth solution closely (no discontinuity anywhere).
    EXPECT_LT(unlimited.max_rho, 0.01);

    // Plain minmod, forced onto every cell every stage, measurably erodes the smooth peak --
    // this is the textbook flaw WENO exists to fix.
    EXPECT_GT(always_minmod.max_rho, 5.0 * unlimited.max_rho);

    // WENO, forced onto exactly the same cells, stays much closer to the unlimited/exact
    // solution than minmod does -- this is the real, isolated proof that WENO's blend (not
    // just indicator selectivity) is doing something minmod's hard cutoff cannot.
    EXPECT_LT(always_weno.l2_rho, 0.5 * always_minmod.l2_rho);
    EXPECT_LT(always_weno.max_rho, 0.5 * always_minmod.max_rho);
}
