/**
 * @file troubled_cell_indicator_test.cpp
 * @brief Tests for the troubled-cell indicator hierarchy (AlwaysTroubledIndicator,
 * PerssonPeraireIndicator).
 */

#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/basis_mode_map.hpp>
#include <dgfem/solver/euler_eigensystem.hpp>
#include <dgfem/solver/neighbor_connectivity.hpp>
#include <dgfem/solver/persson_peraire_indicator.hpp>
#include <dgfem/solver/troubled_cell_indicator_base.hpp>

#include <memory>

#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

namespace {

constexpr int kRhoIdx = 0, kRhoUIdx = 1, kRhoVIdx = 2, kEIdx = 3;
constexpr int kNVars = 4;
constexpr int kNBasis = 4;  // order-1 quad: {const, x-linear, y-linear, xy}
constexpr double kGamma = 1.4;

// Three elements in a row (0 - 1 - 2); element 1 is the only one with both x-neighbors.
NeighborConnectivity make_three_in_a_row_connectivity(bool include_y_entries = false) {
    NeighborConnectivity conn(3);
    conn[0] = {{1, FaceDirection::XPlus}};
    conn[1] = {{0, FaceDirection::XMinus}, {2, FaceDirection::XPlus}};
    conn[2] = {{1, FaceDirection::XMinus}};
    if (include_y_entries) {
        // Extra Y-direction entries that a purely-x-scoped indicator should simply ignore.
        conn[1].push_back({42, FaceDirection::YMinus});
        conn[1].push_back({43, FaceDirection::YPlus});
    }
    return conn;
}

// Flat, unperturbed rho/rho*u/E (zero slope everywhere -> the coupled characteristic triple
// never flags), with rho*v's own mode-0/mode-1 set explicitly so it's the sole determinant of
// whether an element gets flagged.
std::vector<DView2> make_state(double rhov_c0, double rhov_c1) {
    std::vector<DView2> u(3);
    for (int e = 0; e < 3; ++e) {
        u[e] = DView2("u", kNBasis, kNVars);
        u[e](0, kRhoIdx) = 1.0;
        u[e](0, kRhoUIdx) = 0.0;
        u[e](0, kEIdx) = 2.5;
        u[e](0, kRhoVIdx) = rhov_c0;
        u[e](1, kRhoVIdx) = rhov_c1;
        // All other modes (including mode-1 of rho/rho*u/E) stay Kokkos-zero-initialized.
    }
    return u;
}

std::shared_ptr<PerssonPeraireIndicator> make_indicator(double kappa = 1.0) {
    return std::make_shared<PerssonPeraireIndicator>(
        std::make_shared<EulerXDirectionEigensystem>(kNVars),
        std::make_shared<Order1QuadBasisModeMap>(), kappa);
}

}  // namespace

TEST(PerssonPeraireIndicatorTest, SmoothFieldNotFlagged) {
    // c0=1.0, c1=0.05 -> Se_log ~= -3.079, well below the kappa=1.0 threshold (-2.204).
    auto u = make_state(1.0, 0.05);
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    auto indicator = make_indicator();
    auto flags = indicator->detect_troubled_cells(u, conn, dx, kNVars, kGamma);

    EXPECT_EQ(flags[1], 0);
}

TEST(PerssonPeraireIndicatorTest, DiscontinuousFieldFlagged) {
    // c0=1.0, c1=2.0 -> Se_log ~= -0.243, above the kappa=1.0 threshold (-2.204).
    auto u = make_state(1.0, 2.0);
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    auto indicator = make_indicator();
    auto flags = indicator->detect_troubled_cells(u, conn, dx, kNVars, kGamma);

    EXPECT_NE(flags[1], 0);
}

TEST(PerssonPeraireIndicatorTest, BoundaryElementNeverFlaggedEvenIfWildlyDiscontinuous) {
    auto u = make_state(1.0, 100.0);  // extreme value, would certainly flag if not skipped
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    auto indicator = make_indicator();
    auto flags = indicator->detect_troubled_cells(u, conn, dx, kNVars, kGamma);

    EXPECT_EQ(flags[0], 0);  // element 0 has no left neighbor
    EXPECT_EQ(flags[2], 0);  // element 2 has no right neighbor
}

TEST(PerssonPeraireIndicatorTest, ExtraYDirectionEntriesDoNotChangeResult) {
    auto u = make_state(1.0, 2.0);
    std::vector<double> dx(3, 1.0);
    auto indicator = make_indicator();

    auto flags_without_y =
        indicator->detect_troubled_cells(u, make_three_in_a_row_connectivity(false), dx, kNVars, kGamma);
    auto flags_with_y =
        indicator->detect_troubled_cells(u, make_three_in_a_row_connectivity(true), dx, kNVars, kGamma);

    EXPECT_EQ(flags_without_y[1], flags_with_y[1]);
    EXPECT_NE(flags_with_y[1], 0);
}

TEST(PerssonPeraireIndicatorTest, GetTypeReportsPerssonPeraire) {
    auto indicator = make_indicator();
    EXPECT_EQ(indicator->get_type(), "PerssonPeraire");
}

TEST(AlwaysTroubledIndicatorTest, FlagsEveryInteriorElement) {
    auto u = make_state(1.0, 0.0);  // data doesn't matter for this indicator
    auto conn = make_three_in_a_row_connectivity();
    std::vector<double> dx(3, 1.0);

    AlwaysTroubledIndicator indicator;
    auto flags = indicator.detect_troubled_cells(u, conn, dx, kNVars, kGamma);

    EXPECT_EQ(flags[0], 0);  // boundary
    EXPECT_NE(flags[1], 0);  // interior: both neighbors present
    EXPECT_EQ(flags[2], 0);  // boundary
}

TEST(AlwaysTroubledIndicatorTest, GetTypeReportsAlwaysTroubled) {
    AlwaysTroubledIndicator indicator;
    EXPECT_EQ(indicator.get_type(), "AlwaysTroubled");
}
