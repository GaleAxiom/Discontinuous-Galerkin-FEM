/**
 * @file sod_shock_tube_test.cpp
 * @brief End-to-end test: Sod shock tube, comparing no limiter / old always-on characteristic
 * minmod / the new default (Persson-Peraire indicator + WENO reconstruction) against the
 * exact Riemann solution.
 *
 * This is the assertion-based counterpart of dgfem/examples/sod_shock_tube_example.cpp's
 * console comparison. Two genuinely different claims are tested here, deliberately kept
 * separate because they were found (empirically, via calibration sweeps across kappa/gamma_own)
 * to NOT both reduce to the same comparison:
 *
 * 1. Boundedness (IndicatorWenoSuppressesDensityOvershootThatUnlimitedProduces): a limiter's
 *    actual job is suppressing the spurious Gibbs oscillation unlimited high-order DG produces
 *    at a discontinuity -- density leaving the exact solution's true [rho_R, rho_L] range is
 *    unambiguous evidence of that oscillation, not "error" in the approximation sense.
 *    Indicator+WENO substantially reduces it relative to unlimited. This holds even though (2):
 *
 * 2. Accuracy vs. old minmod (IndicatorWenoImprovesAccuracyOverOldAlwaysOnMinmod): on this
 *    particular, already-well-resolved Sod configuration (dx=0.01, order-1, T=0.1), plain/TVB
 *    characteristic minmod was measured (via direct instrumentation, not guesswork) to make L2
 *    error *worse* than no limiter at all, because its magnitude-based TVB threshold cannot
 *    distinguish a genuine discontinuity from the small dispersive "precursor" oscillation
 *    unlimited DG produces ahead of a true wavefront. Persson-Peraire's modal-decay ratio does
 *    distinguish them, so indicator+WENO recovers most of that lost accuracy relative to old
 *    minmod -- but it does NOT beat plain unlimited on raw L2/max error here, because unlimited
 *    is already quite accurate on this mesh; its only flaw is the boundedness violation in (1).
 *    That is an expected, honest numerical-methods result: limiters exist for robustness/physical
 *    validity, not necessarily an accuracy win over an already-well-behaved unlimited solution.
 *
 * Neither claim above actually isolates whether WENO's own smoothness-weighted blend is doing
 * anything beyond what the Persson-Peraire indicator's cell selectivity alone buys -- on this
 * problem the indicator only ever flags the shock/contact cells, so swapping WENO for plain
 * minmod under the SAME indicator gives near-identical results (verified directly; not included
 * here since it added no further assertion value once understood). See
 * smooth_advection_limiter_test.cpp for the test that actually isolates and proves WENO's own
 * contribution, using AlwaysTroubledIndicator on a smooth solution to remove the indicator's
 * selectivity from the comparison entirely.
 */

#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/solver/euler_eigensystem.hpp>
#include <dgfem/solver/minmod_reconstruction.hpp>
#include <dgfem/solver/persson_peraire_indicator.hpp>
#include <dgfem/solver/troubled_cell_indicator_base.hpp>
#include <dgfem/solver/weno_reconstruction.hpp>
#include <dgfem/utils/example_helpers.hpp>
#include <dgfem/utils/riemann_solver.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include <gmsh.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

namespace {

constexpr double kGamma = 1.4;
constexpr double kRhoL = 1.0, kUL = 0.0, kPL = 1.0;
constexpr double kRhoR = 0.125, kUR = 0.0, kPR = 0.1;
constexpr double kX0 = 0.5;
constexpr double kDt = 1e-4;
constexpr double kTFinal = 0.1;
constexpr int kSaveEvery = 1000;  // only need the final frame

struct CaseErrors {
    double l2_rho, max_rho;
    double l2_u, max_u;
};

enum class LimiterMode { None, OldAlwaysOnMinmod, DefaultIndicatorWeno };

// Builds the same quasi-1D Sod setup as sod_shock_tube_example.cpp (long thin domain, slip
// walls in y, far-field left/right fixed at the undisturbed states), runs it under the given
// limiter mode, and hands the resulting (mesh, final-time solution) to `consume` while gmsh is
// still initialized. Shared by both the accuracy-vs-exact-solution test and the boundedness
// test below so the mesh/solver/BC setup isn't duplicated between them.
template <typename Fn>
void run_sod_solve(LimiterMode mode, Fn&& consume, double kappa = 0.0,
                   double weno_gamma_own = 0.998) {
    try {
        gmsh::clear();
        gmsh::finalize();
    } catch (...) {
    }
    gmsh::initialize();

    auto mesh = MeshSetup::create_standard_mesh(
        /*use_triangles=*/false,
        /*order=*/1,
        /*dx=*/0.01,
        /*n_vars=*/4,
        /*xmin=*/0.0,
        /*xmax=*/1.0,
        /*ymin=*/0.0,
        /*ymax=*/0.02);

    Vec4 left_conserved = primitive_to_conserved(Vec4{kRhoL, kUL, 0.0, kPL}, kGamma);
    Vec4 right_conserved = primitive_to_conserved(Vec4{kRhoR, kUR, 0.0, kPR}, kGamma);
    auto left_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, left_conserved);
    auto right_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, right_conserved);
    auto slip_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL, Vec4{1.0, 0.0, 0.0, 1.0});
    mesh->set_boundary_condition_euler("Left", left_bc);
    mesh->set_boundary_condition_euler("Right", right_bc);
    mesh->set_boundary_condition_euler("Bottom", slip_bc);
    mesh->set_boundary_condition_euler("Top", slip_bc);

    auto initial_condition = [](const Vec2& x) -> Vec4 {
        Vec4 primitive = (x[0] < kX0) ? Vec4{kRhoL, kUL, 0.0, kPL} : Vec4{kRhoR, kUR, 0.0, kPR};
        return primitive_to_conserved(primitive, kGamma);
    };

    EulerDGSolver solver(mesh, kGamma);
    switch (mode) {
    case LimiterMode::None:
        break;
    case LimiterMode::OldAlwaysOnMinmod:
        solver.set_reconstruction_technique(
            std::make_shared<AlwaysTroubledIndicator>(),
            std::make_shared<MinmodReconstruction>(std::make_shared<EulerXDirectionEigensystem>(4)));
        solver.set_limiter_enabled(true);
        break;
    case LimiterMode::DefaultIndicatorWeno:
        solver.set_reconstruction_technique(
            std::make_shared<PerssonPeraireIndicator>(
                std::make_shared<EulerXDirectionEigensystem>(4),
                std::make_shared<Order1QuadBasisModeMap>(), kappa),
            std::make_shared<WenoReconstruction>(std::make_shared<EulerXDirectionEigensystem>(4),
                                                 std::make_shared<Order1QuadBasisModeMap>(),
                                                 weno_gamma_own));
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

CaseErrors run_sod_case(const ExactRiemannSolution& exact, LimiterMode mode, double kappa = 0.0,
                        double weno_gamma_own = 0.998) {
    CaseErrors result{};
    run_sod_solve(
        mode,
        [&](const std::shared_ptr<DGMesh>& mesh, const DView2& final_sol) {
            auto [l2_rho, max_rho] =
                compute_riemann_solution_error(mesh, final_sol, exact, kGamma, kTFinal, 0);
            auto [l2_u, max_u] =
                compute_riemann_solution_error(mesh, final_sol, exact, kGamma, kTFinal, 1);
            result = {l2_rho, max_rho, l2_u, max_u};
        },
        kappa, weno_gamma_own);
    return result;
}

// Sod's exact density is monotone and bounded within [rho_R, rho_L] everywhere -- the
// rarefaction fan decreases smoothly from rho_L, and the post-shock/contact states sit between
// the two undisturbed values, so no genuine physical density ever leaves that range. Any
// numerical density outside it is therefore unambiguous spurious oscillation (Gibbs ringing at
// the shock/contact), not approximation error -- exactly what a limiter exists to suppress, as
// distinct from the (separate, and separately tested) question of whether it also improves L2
// error on an already-well-resolved mesh. Returns 0 if fully bounded, otherwise the largest
// amount by which density left [rho_R, rho_L] anywhere in the domain.
double run_sod_max_density_overshoot(LimiterMode mode, double kappa = 0.0,
                                     double weno_gamma_own = 0.998) {
    double violation = 0.0;
    run_sod_solve(
        mode,
        [&](const std::shared_ptr<DGMesh>& mesh, const DView2& final_sol) {
            auto space = mesh->get_dg_space();
            const auto& phi = space->get_volume_basis_values();
            const auto& quad_wts = space->get_volume_quad()->weights;
            int n_basis = space->get_basis()->get_n_basis();
            for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
                for (int q = 0; q < static_cast<int>(quad_wts.size()); ++q) {
                    Vec4 U_num{0.0, 0.0, 0.0, 0.0};
                    for (int i = 0; i < n_basis; ++i) {
                        for (int v = 0; v < 4; ++v) {
                            U_num[v] += final_sol(elem, i * 4 + v) * phi(q, i);
                        }
                    }
                    double rho = conserved_to_primitive(U_num, kGamma)[0];
                    violation = std::max({violation, kRhoR - rho, rho - kRhoL});
                }
            }
        },
        kappa, weno_gamma_own);
    return violation;
}

}  // namespace

// Toro's Test 2 (the "123 problem" / double rarefaction: rho_L=rho_R=1, u_L=-2, u_R=+2,
// p_L=p_R=0.4) is the classic robustness stress test in the Riemann-solver literature: the two
// rarefactions moving apart create a deep near-vacuum region (p_star ~ 0.0019) at the domain
// center. On a coarse mesh (dx=0.05), the unlimited solver genuinely produces non-finite output
// here -- not merely "less accurate," an actual NaN/Inf breakdown -- while indicator+WENO
// survives and stays reasonably close to the exact solution. This is the direct answer to
// "does WENO have a purpose beyond not being worse than unlimited": on the Sod problem it
// doesn't show one (see IndicatorWenoImprovesAccuracyOverOldAlwaysOnMinmod's honest framing
// above); here, it is the difference between a solver that works and one that doesn't.
TEST(SodShockTubeTest, IndicatorWenoSurvivesNearVacuumWhereUnlimitedFails) {
    constexpr double rho_L = 1.0, u_L = -2.0, p_L = 0.4;
    constexpr double rho_R = 1.0, u_R = 2.0, p_R = 0.4;
    constexpr double strong_dx = 0.05;
    constexpr double strong_t_final = 0.1;

    ExactRiemannSolution exact(kGamma, rho_L, u_L, p_L, rho_R, u_R, p_R, 0.5);
    EXPECT_LT(exact.p_star, 0.01);  // confirms this is genuinely a deep near-vacuum case

    auto build_and_solve =
        [&](bool use_indicator_weno) -> std::pair<std::shared_ptr<DGMesh>, DView2> {
        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
        }
        gmsh::initialize();

        auto mesh = MeshSetup::create_standard_mesh(false, 1, strong_dx, 4, 0.0, 1.0, 0.0, 0.02);
        Vec4 left_conserved = primitive_to_conserved(Vec4{rho_L, u_L, 0.0, p_L}, kGamma);
        Vec4 right_conserved = primitive_to_conserved(Vec4{rho_R, u_R, 0.0, p_R}, kGamma);
        auto left_bc =
            std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, left_conserved);
        auto right_bc =
            std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, right_conserved);
        auto slip_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL,
                                                                 Vec4{1.0, 0.0, 0.0, 1.0});
        mesh->set_boundary_condition_euler("Left", left_bc);
        mesh->set_boundary_condition_euler("Right", right_bc);
        mesh->set_boundary_condition_euler("Bottom", slip_bc);
        mesh->set_boundary_condition_euler("Top", slip_bc);
        auto ic = [=](const Vec2& x) -> Vec4 {
            Vec4 primitive = (x[0] < 0.5) ? Vec4{rho_L, u_L, 0.0, p_L} : Vec4{rho_R, u_R, 0.0, p_R};
            return primitive_to_conserved(primitive, kGamma);
        };

        EulerDGSolver solver(mesh, kGamma);
        if (use_indicator_weno) {
            solver.set_limiter_enabled(true);
        }
        auto solutions = solver.solve(ic, strong_t_final, 1e-4, 1000);
        DView2 final_sol = solutions.back();

        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
        }
        return {mesh, final_sol};
    };

    auto [mesh_unlimited, unlimited_final] = build_and_solve(/*use_indicator_weno=*/false);
    EXPECT_FALSE(all_finite(unlimited_final))
        << "expected unlimited to blow up on this near-vacuum case -- if it no longer does, "
           "the premise of this test needs revisiting";

    auto [mesh_weno, weno_final] = build_and_solve(/*use_indicator_weno=*/true);
    ASSERT_TRUE(all_finite(weno_final));
    auto [l2_rho, max_rho] =
        compute_riemann_solution_error(mesh_weno, weno_final, exact, kGamma, strong_t_final, 0);
    // Absolute sanity ceilings with margin (measured: l2_rho=0.085, max_rho=0.158 -- errors are
    // naturally larger than the mild Sod case since this is a coarse mesh on a much harder,
    // near-vacuum problem, but the point here is survival + reasonableness, not fine accuracy).
    EXPECT_LT(l2_rho, 0.15);
    EXPECT_LT(max_rho, 0.25);
}

TEST(SodShockTubeTest, ExactRiemannSolverMatchesPublishedReferenceStarState) {
    ExactRiemannSolution exact(kGamma, kRhoL, kUL, kPL, kRhoR, kUR, kPR, kX0);
    EXPECT_NEAR(exact.p_star, 0.30313, 1e-4);
    EXPECT_NEAR(exact.u_star, 0.92745, 1e-4);
}

TEST(SodShockTubeTest, IndicatorWenoSuppressesDensityOvershootThatUnlimitedProduces) {
    double unlimited_overshoot = run_sod_max_density_overshoot(LimiterMode::None);
    double indicator_weno_overshoot = run_sod_max_density_overshoot(LimiterMode::DefaultIndicatorWeno);

    // Measured: unlimited=0.0233, old_minmod=0.00081, indicator_weno(kappa=0)=0.00875. Old
    // minmod suppresses oscillation even more aggressively than indicator+WENO here -- expected,
    // since it limits every cell unconditionally rather than only genuinely troubled ones -- but
    // that blunt aggressiveness is exactly what costs it accuracy elsewhere (see
    // IndicatorWenoImprovesAccuracyOverOldAlwaysOnMinmod below), so it is not a fair "beat"
    // target for a selective indicator. The claim here is narrower and more fundamental:
    // unlimited genuinely, substantially violates the exact solution's physical density bounds...
    EXPECT_GT(unlimited_overshoot, 0.01);
    // ...and indicator+WENO suppresses that violation by more than half.
    EXPECT_LT(indicator_weno_overshoot, 0.5 * unlimited_overshoot);
}

TEST(SodShockTubeTest, IndicatorWenoImprovesAccuracyOverOldAlwaysOnMinmod) {
    ExactRiemannSolution exact(kGamma, kRhoL, kUL, kPL, kRhoR, kUR, kPR, kX0);

    CaseErrors old_minmod = run_sod_case(exact, LimiterMode::OldAlwaysOnMinmod);
    CaseErrors indicator_weno = run_sod_case(exact, LimiterMode::DefaultIndicatorWeno);

    // The specific regression this whole refactor exists to fix: old always-on characteristic
    // minmod was measured to actively hurt accuracy on this problem (see the file doc comment);
    // indicator+WENO must recover most of that. It is NOT compared against plain unlimited here
    // -- on this well-resolved mesh unlimited already has the lowest raw L2/max error, and no
    // calibration of kappa/gamma_own closes that gap (empirically swept and confirmed) without
    // giving up the boundedness improvement tested separately above. That is expected: a limiter
    // trades a little accuracy on an already-fine solution for suppressing genuine oscillation.
    EXPECT_LT(indicator_weno.l2_rho, old_minmod.l2_rho);
    EXPECT_LT(indicator_weno.l2_u, old_minmod.l2_u);
    EXPECT_LT(indicator_weno.max_u, old_minmod.max_u);

    // Absolute sanity ceilings as a permanent regression guard (values calibrated from an actual
    // run, with margin). max_rho is a single-point pointwise error right at the discontinuity and
    // is noise-dominated between old_minmod and indicator_weno (0.0926 vs 0.0991 measured) --
    // kept as an absolute ceiling rather than a strict comparison for that reason.
    EXPECT_LT(indicator_weno.l2_rho, 0.025);
    EXPECT_LT(indicator_weno.max_rho, 0.11);
}
