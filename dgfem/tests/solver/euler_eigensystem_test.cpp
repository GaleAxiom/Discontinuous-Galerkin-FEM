/**
 * @file euler_eigensystem_test.cpp
 * @brief Tests for the 1D Euler flux-Jacobian eigensystem provider used by the troubled-cell
 * indicator / reconstruction technique hierarchy.
 */

#include <dgfem/solver/euler_eigensystem.hpp>

#include <cmath>

#include <gtest/gtest.h>
#include <stdexcept>

using namespace dgfem;
using namespace testing;

namespace {

double max_abs_diff_from_identity(const std::vector<std::vector<double>>& L,
                                  const std::vector<std::vector<double>>& R) {
    double max_diff = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += L[i][k] * R[k][j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            max_diff = std::max(max_diff, std::abs(sum - expected));
        }
    }
    return max_diff;
}

// 1D Euler flux Jacobian (on the reduced (rho, rho*u, E) triple), built directly from
// (rho, u, p, gamma) via the standard closed form -- independent of EulerXDirectionEigensystem
// itself, so this genuinely checks R's columns are eigenvectors rather than just checking
// internal self-consistency.
std::vector<std::vector<double>> build_flux_jacobian(double rho, double u, double p,
                                                      double gamma) {
    double c = std::sqrt(gamma * p / rho);
    double H = c * c / (gamma - 1.0) + 0.5 * u * u;
    (void)H;
    std::vector<std::vector<double>> A(3, std::vector<double>(3, 0.0));
    A[0][1] = 1.0;
    A[1][0] = 0.5 * (gamma - 3.0) * u * u;
    A[1][1] = (3.0 - gamma) * u;
    A[1][2] = gamma - 1.0;
    double Hh = c * c / (gamma - 1.0) + 0.5 * u * u;
    A[2][0] = 0.5 * (gamma - 1.0) * u * u * u - u * Hh;
    A[2][1] = Hh - (gamma - 1.0) * u * u;
    A[2][2] = gamma * u;
    return A;
}

void expect_eigenvector(const std::vector<std::vector<double>>& A,
                        const std::vector<std::vector<double>>& R, int col, double eigenvalue) {
    for (int i = 0; i < 3; ++i) {
        double Av_i = A[i][0] * R[0][col] + A[i][1] * R[1][col] + A[i][2] * R[2][col];
        EXPECT_NEAR(Av_i, eigenvalue * R[i][col], 1e-9)
            << "row " << i << ", col " << col;
    }
}

struct EulerState {
    double rho, u, p;
};

}  // namespace

TEST(EulerEigensystemTest, LInverseOfRAcrossRepresentativeStates) {
    constexpr double gamma = 1.4;
    std::vector<EulerState> states = {
        {1.0, 0.3, 1.0},     // subsonic
        {1.0, 0.99, 1.0},    // near-sonic (u close to c=sqrt(1.4)~1.183)
        {1.0, 2.5, 1.0},     // supersonic
        {0.01, 0.1, 0.05},   // small rho
        {0.7, -0.6, 0.8},    // negative velocity
    };

    EulerXDirectionEigensystem provider(4);
    for (const auto& s : states) {
        std::vector<double> prim{s.rho, s.u, 0.0, s.p};
        LocalEigensystem es = provider.build({1.0, 0.0}, prim, gamma);
        ASSERT_EQ(es.n_coupled, 3);
        EXPECT_LT(max_abs_diff_from_identity(es.L, es.R), 1e-9)
            << "state rho=" << s.rho << " u=" << s.u << " p=" << s.p;
    }
}

TEST(EulerEigensystemTest, RColumnsAreGenuineEigenvectors) {
    constexpr double gamma = 1.4;
    std::vector<EulerState> states = {
        {1.0, 0.3, 1.0}, {1.0, 2.5, 1.0}, {0.01, 0.1, 0.05}, {0.7, -0.6, 0.8}};

    EulerXDirectionEigensystem provider(4);
    for (const auto& s : states) {
        std::vector<double> prim{s.rho, s.u, 0.0, s.p};
        LocalEigensystem es = provider.build({1.0, 0.0}, prim, gamma);
        auto A = build_flux_jacobian(s.rho, s.u, s.p, gamma);
        double c = std::sqrt(gamma * s.p / s.rho);

        expect_eigenvector(A, es.R, 0, s.u - c);
        expect_eigenvector(A, es.R, 1, s.u);
        expect_eigenvector(A, es.R, 2, s.u + c);
    }
}

TEST(EulerEigensystemTest, CoupledFieldConfigForFourVars) {
    EulerXDirectionEigensystem provider(4);
    const CoupledFieldConfig& config = provider.coupled_field_config();
    EXPECT_EQ(config.coupled_indices, (std::vector<int>{0, 1, 3}));
    EXPECT_EQ(config.passive_indices, (std::vector<int>{2}));
}

// Proves the k-omega-shaped generalization (more passive scalars riding on the same 3-field
// mean-flow system) already works today, with zero k-omega-specific code -- constructing the
// same class with a wider n_vars just reports more passive indices.
TEST(EulerEigensystemTest, CoupledFieldConfigForSixVarsIsKOmegaShaped) {
    EulerXDirectionEigensystem provider(6);
    const CoupledFieldConfig& config = provider.coupled_field_config();
    EXPECT_EQ(config.coupled_indices, (std::vector<int>{0, 1, 3}));
    EXPECT_EQ(config.passive_indices, (std::vector<int>{2, 4, 5}));
}

TEST(EulerEigensystemTest, BuildThrowsOnNonXDirection) {
    EulerXDirectionEigensystem provider(4);
    std::vector<double> prim{1.0, 0.3, 0.0, 1.0};
    EXPECT_THROW(provider.build({0.0, 1.0}, prim, 1.4), std::invalid_argument);
    EXPECT_THROW(provider.build({1.0, 0.0, 0.0}, prim, 1.4), std::invalid_argument);
}

TEST(EulerEigensystemTest, GetTypeReportsEulerXDirection) {
    EulerXDirectionEigensystem provider(4);
    EXPECT_EQ(provider.get_type(), "EulerXDirection");
}
