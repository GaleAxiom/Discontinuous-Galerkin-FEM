/**
 * @file reconstruction_technique_test.cpp
 * @brief Tests for the reconstruction technique hierarchy (MinmodReconstruction,
 * WenoReconstruction), including WenoReconstruction::blend_slopes in isolation.
 */

#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/basis_mode_map.hpp>
#include <dgfem/solver/euler_eigensystem.hpp>
#include <dgfem/solver/minmod_reconstruction.hpp>
#include <dgfem/solver/neighbor_connectivity.hpp>
#include <dgfem/solver/weno_reconstruction.hpp>

#include <memory>

#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

namespace {

constexpr int kRhoIdx = 0, kRhoUIdx = 1, kRhoVIdx = 2, kEIdx = 3;
constexpr int kNVars = 4;
constexpr int kNBasis = 4;
constexpr double kGamma = 1.4;

double minmod3_reference(double a, double b, double c) {
    if (a > 0.0 && b > 0.0 && c > 0.0) {
        return std::min({a, b, c});
    }
    if (a < 0.0 && b < 0.0 && c < 0.0) {
        return std::max({a, b, c});
    }
    return 0.0;
}

NeighborConnectivity make_three_in_a_row_connectivity() {
    NeighborConnectivity conn(3);
    conn[0] = {{1, FaceDirection::XPlus}};
    conn[1] = {{0, FaceDirection::XMinus}, {2, FaceDirection::XPlus}};
    conn[2] = {{1, FaceDirection::XMinus}};
    return conn;
}

// Mock BasisModeMap reporting a DIFFERENT mode as "linear in X" than Order1QuadBasisModeMap
// (2 instead of 1), and a different cross term (1 instead of 3) -- proves reconstruction
// classes actually consult the injected map rather than hardcoding mode indices.
class MockBasisModeMap : public BasisModeMap {
public:
    [[nodiscard]] std::string get_type() const override { return "Mock"; }
    [[nodiscard]] int linear_mode_index(Direction) const override { return 2; }
    [[nodiscard]] std::vector<int> dependent_cross_term_indices(Direction) const override {
        return {1};
    }
};

// Mock FluxEigensystemProvider with an identity R/L (so the "characteristic" path reduces to
// per-field pass-through) and a configurable CoupledFieldConfig -- lets the test drive a
// 5-wide, k-omega-shaped state (passive_indices={2,4}) through today's classes without any
// k-omega-specific code exiting.
class MockEigensystemProvider : public FluxEigensystemProvider {
public:
    explicit MockEigensystemProvider(CoupledFieldConfig config) : config_(std::move(config)) {}
    [[nodiscard]] std::string get_type() const override { return "Mock"; }
    [[nodiscard]] LocalEigensystem build(const std::vector<double>&, const std::vector<double>&,
                                         double) const override {
        LocalEigensystem es;
        es.n_coupled = 3;
        es.R = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        es.L = es.R;
        return es;
    }
    [[nodiscard]] const CoupledFieldConfig& coupled_field_config() const override {
        return config_;
    }

private:
    CoupledFieldConfig config_;
};

}  // namespace

// --- WenoReconstruction::blend_slopes (pure function) ---------------------------------------

TEST(WenoBlendSlopesTest, AllEqualCandidatesReturnsThatValue) {
    double blended = WenoReconstruction::blend_slopes(0.5, 0.5, 0.5, 1.0, 0.998, 1e-6, 2.0);
    EXPECT_NEAR(blended, 0.5, 1e-12);
}

TEST(WenoBlendSlopesTest, SmoothOwnSlopeStaysCloseToOwn) {
    double blended = WenoReconstruction::blend_slopes(0.02, 3.0, -2.5, 1.0, 0.998, 1e-6, 2.0);
    EXPECT_NEAR(blended, 0.019999999995533365, 1e-12);
}

// The key minmod-vs-WENO contrast: mixed-sign candidates make minmod3 return exactly 0, but
// WENO's smoothness-weighted blend stays nonzero, biased toward whichever candidate is
// smoothest (here, the small-magnitude left neighbor) rather than being all-or-nothing.
TEST(WenoBlendSlopesTest, MixedSignCandidatesDivergeFromMinmod) {
    double minmod_result = minmod3_reference(1.0, 0.05, -0.3);
    ASSERT_EQ(minmod_result, 0.0);

    double blended = WenoReconstruction::blend_slopes(1.0, 0.05, -0.3, 1.0, 0.998, 1e-6, 2.0);
    EXPECT_NEAR(blended, 0.05562065565090164, 1e-12);
    EXPECT_NE(blended, 0.0);
}

// --- MinmodReconstruction::apply: bit-identical to the pre-refactor apply_minmod_limiter_x --

TEST(MinmodReconstructionTest, ApplyMatchesPrefactorCharacteristicFormula) {
    std::vector<DView2> u(3);
    for (int e = 0; e < 3; ++e) {
        u[e] = DView2("u", kNBasis, kNVars);
    }
    // Cell averages (rho, rho*u, E) for elements 0,1,2; element 1's average sets the frozen
    // eigensystem (rho=1, u=0, p=1/1.4 -> c=1 exactly).
    double avg0[3] = {0.8, 0.1, 1.6};
    double avg1[3] = {1.0, 0.0, 1.0 / 0.56};
    double avg2[3] = {1.3, -0.05, 2.0};
    int coupled[3] = {kRhoIdx, kRhoUIdx, kEIdx};
    for (int k = 0; k < 3; ++k) {
        u[0](0, coupled[k]) = avg0[k];
        u[1](0, coupled[k]) = avg1[k];
        u[2](0, coupled[k]) = avg2[k];
    }
    // Element 1's own slope (mode-1 coefficients: rho=0.1, rho*u=0.2, E=0.05 -> physical
    // slopes 0.2, 0.4, 0.1 since dx=1).
    u[1](1, kRhoIdx) = 0.1;
    u[1](1, kRhoUIdx) = 0.2;
    u[1](1, kEIdx) = 0.05;
    // rho*v: own average kept at exactly 0 (a nonzero v here would perturb the pressure
    // conserved_to_primitive computes for element 1 -- via the kinetic-energy term -- shifting
    // the eigensystem away from the hand-derived c=1 state the coupled-field expectations
    // below assume). Own slope 0.15/2=0.075 (physical slope 0.15); neighbor averages 0.02/-0.1.
    u[0](0, kRhoVIdx) = 0.02;
    u[1](0, kRhoVIdx) = 0.0;
    u[1](1, kRhoVIdx) = 0.075;
    u[2](0, kRhoVIdx) = -0.1;

    std::vector<std::uint8_t> troubled = {0, 1, 0};
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    // tvb_m small enough that the TVB gate doesn't block minmod3 from actually engaging (the
    // production default of 50.0 would leave every field within tolerance here, since these
    // slopes are all under 0.25 in magnitude -- this test wants to exercise minmod3 itself).
    MinmodReconstruction minmod(std::make_shared<EulerXDirectionEigensystem>(kNVars),
                               std::make_shared<Order1QuadBasisModeMap>(), /*tvb_m=*/0.01);

    auto result = minmod.apply(u, troubled, conn, dx, kNVars, kGamma);

    // Hand-derived (via the same formula, computed independently in Python): char-space
    // minmod3 zeroes fields 0 and 2 (mixed-sign candidates) and keeps field 1's minimum
    // (0.12571428571428578), transformed back through R.
    EXPECT_NEAR(result[1](1, kRhoIdx), 0.06285714285714289, 1e-12);
    EXPECT_NEAR(result[1](1, kRhoUIdx), 0.0, 1e-12);
    EXPECT_NEAR(result[1](1, kEIdx), 0.0, 1e-12);
    // rho*v: own slope 0.15, fwd=(−0.1−0.05)/1=−0.15, bwd=(0.05−0.02)/1=0.03 -> mixed signs -> 0.
    EXPECT_NEAR(result[1](1, kRhoVIdx), 0.0, 1e-12);

    // Cross term (mode 3) zeroed for every variable whose slope was touched.
    for (int v = 0; v < kNVars; ++v) {
        EXPECT_NEAR(result[1](3, v), 0.0, 1e-12);
    }

    // Untroubled elements pass through byte-identical.
    for (int v = 0; v < kNVars; ++v) {
        EXPECT_NEAR(result[0](0, v), u[0](0, v), 1e-15);
        EXPECT_NEAR(result[2](0, v), u[2](0, v), 1e-15);
    }
}

// --- WenoReconstruction::apply: full pipeline at a hand-computable sonic state --------------

TEST(WenoReconstructionTest, ApplyAtSonicStateMatchesHandDerivedBlend) {
    std::vector<DView2> u(3);
    for (int e = 0; e < 3; ++e) {
        u[e] = DView2("u", kNBasis, kNVars);
    }
    // All three elements share the same cell average: rho=1, u=0, p=1/1.4 -> c=1 exactly.
    double avg[3] = {1.0, 0.0, 1.0 / 0.56};
    int coupled[3] = {kRhoIdx, kRhoUIdx, kEIdx};
    for (int e = 0; e < 3; ++e) {
        for (int k = 0; k < 3; ++k) {
            u[e](0, coupled[k]) = avg[k];
        }
    }
    // Own (element 1), left (element 0), right (element 2) OWN slopes (mode-1 coefficients;
    // physical slope = mode1*2/dx with dx=1).
    double own_modes[3] = {0.1, 0.2, 0.05};      // physical slopes 0.2, 0.4, 0.1
    double left_modes[3] = {0.05, 0.1, 0.02};    // physical slopes 0.1, 0.2, 0.04
    double right_modes[3] = {-0.3, -0.1, -0.15}; // physical slopes -0.6, -0.2, -0.3
    for (int k = 0; k < 3; ++k) {
        u[1](1, coupled[k]) = own_modes[k];
        u[0](1, coupled[k]) = left_modes[k];
        u[2](1, coupled[k]) = right_modes[k];
    }

    std::vector<std::uint8_t> troubled = {0, 1, 0};
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    WenoReconstruction weno(std::make_shared<EulerXDirectionEigensystem>(kNVars),
                           std::make_shared<Order1QuadBasisModeMap>());

    auto result = weno.apply(u, troubled, conn, dx, kNVars, kGamma);

    // Hand-derived (via the same formula, computed independently in Python).
    EXPECT_NEAR(result[1](1, kRhoIdx), 0.1300198742469535, 1e-10);
    EXPECT_NEAR(result[1](1, kRhoUIdx), 0.1662557909755815, 1e-10);
    EXPECT_NEAR(result[1](1, kEIdx), 0.1262958902021112, 1e-10);

    // Cross term (mode 3) is untouched by WenoReconstruction by design (unlike
    // MinmodReconstruction's discrete moment-limiter cascade -- see weno_reconstruction.cpp's
    // doc comment for why zeroing it unconditionally under a continuous blend is pure
    // information loss); it stays at its Kokkos-zero-initialized input value here.
    for (int v = 0; v < kNVars; ++v) {
        EXPECT_NEAR(result[1](3, v), 0.0, 1e-12);
    }
}

// --- Indirection checks: mock BasisModeMap / mock FluxEigensystemProvider ------------------

// Uses MinmodReconstruction (not WenoReconstruction) specifically because this test's point is
// to prove the cross-term-zeroing cascade consults the injected BasisModeMap rather than a
// hardcoded mode index -- WenoReconstruction deliberately never zeroes a cross term at all (see
// weno_reconstruction.cpp), so it can't demonstrate this particular indirection.
TEST(MinmodReconstructionTest, MockBasisModeMapChangesWhichModeIsReadAndZeroed) {
    std::vector<DView2> u(3);
    for (int e = 0; e < 3; ++e) {
        u[e] = DView2("u", kNBasis, kNVars);
        u[e](0, kRhoIdx) = 1.0;
        u[e](0, kEIdx) = 1.0 / 0.56;
        // rho, rho*u, E stay perfectly flat (all modes zero beyond mode 0) so the coupled
        // characteristic path contributes nothing, isolating rho*v as the only field that
        // can change -- and per the mock, its slope lives in mode 2, not mode 1.
    }
    // Decoy value in mode 1 (the *real* Order1Quad x-linear mode) that a correctly-wired
    // reconstruction must NOT read as a slope under the mock mapping.
    u[1](1, kRhoVIdx) = 5.0;
    // Real slope data (per the mock) lives in mode 2.
    u[0](2, kRhoVIdx) = 0.1;
    u[1](2, kRhoVIdx) = 0.3;
    u[2](2, kRhoVIdx) = -0.05;

    std::vector<std::uint8_t> troubled = {0, 1, 0};
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    // Small tvb_m so the (per the mock) physical slope of 0.6 clears the TVB threshold and
    // minmod3 actually engages; cell averages are all 0 here, so the forward/backward
    // cell-average differences are both 0 -- mixed-sign against the 0.6 physical slope --
    // making minmod3 collapse the limited slope to exactly 0.
    MinmodReconstruction minmod(std::make_shared<EulerXDirectionEigensystem>(kNVars),
                               std::make_shared<MockBasisModeMap>(), /*tvb_m=*/0.01);
    auto result = minmod.apply(u, troubled, conn, dx, kNVars, kGamma);

    // Mode 2 (the mock's "linear mode") was read as the slope (0.3, i.e. 0.6 physical) and
    // collapsed to 0 by minmod3 against the (both-zero) cell-average differences.
    EXPECT_NEAR(result[1](2, kRhoVIdx), 0.0, 1e-12);
    // Mode 1 (the decoy, and the mock's declared cross term) was zeroed, not left at 5.0 and
    // not read as a slope.
    EXPECT_NEAR(result[1](1, kRhoVIdx), 0.0, 1e-12);
}

TEST(WenoReconstructionTest, KOmegaShapedConfigLimitsExtraPassiveIndicesComponentwise) {
    constexpr int kNVarsWide = 5;  // rho, rho*u, rho*v, E, + one extra passive scalar
    CoupledFieldConfig config;
    config.coupled_indices = {0, 1, 3};
    config.passive_indices = {2, 4};

    std::vector<DView2> u(3);
    for (int e = 0; e < 3; ++e) {
        u[e] = DView2("u", kNBasis, kNVarsWide);
        u[e](0, 0) = 1.0;
        u[e](0, 3) = 1.0 / 0.56;
        // Indices 1 (rho*u) and 2,4 (passive) stay flat except where set below.
    }
    // Passive index 2: own/left/right own slopes (mode 1, since default mode map is used).
    u[0](1, 2) = 0.1;
    u[1](1, 2) = 0.3;
    u[2](1, 2) = -0.05;
    // Passive index 4: different values, to confirm it's handled independently of index 2.
    u[0](1, 4) = -0.2;
    u[1](1, 4) = 0.4;
    u[2](1, 4) = 0.1;

    std::vector<std::uint8_t> troubled = {0, 1, 0};
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    WenoReconstruction weno(std::make_shared<MockEigensystemProvider>(config),
                           std::make_shared<Order1QuadBasisModeMap>());
    // Should not throw or crash despite the 5-wide, k-omega-shaped state.
    auto result = weno.apply(u, troubled, conn, dx, kNVarsWide, kGamma);

    // Both passive indices actually changed (were blended away from their own-slope inputs),
    // independently of each other.
    EXPECT_NE(result[1](1, 2), u[1](1, 2));
    EXPECT_NE(result[1](1, 4), u[1](1, 4));
}
