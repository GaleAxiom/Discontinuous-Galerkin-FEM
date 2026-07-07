/**
 * @file compressible_dg_solver_limiter_test.cpp
 * @brief Integration tests for the limiter wiring on CompressibleDGSolverBase (via
 * EulerDGSolver): precondition checks, the lazy Persson-Peraire+WENO default, and the
 * AlwaysTroubled+Minmod pairing kept available for regression comparison.
 */

#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/basis_mode_map.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/solver/euler_eigensystem.hpp>
#include <dgfem/solver/minmod_reconstruction.hpp>
#include <dgfem/solver/troubled_cell_indicator_base.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <memory>
#include <stdexcept>

#include <gmsh.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

namespace {

class CompressibleDGSolverLimiterTest : public Test {
protected:
    void SetUp() override {
        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
        }
        gmsh::initialize();
    }

    void TearDown() override {
        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
        }
    }

    // Small quasi-1D discontinuous IC (a mini Sod-like jump) on a thin strip, matching the
    // scope set_limiter_enabled() actually supports (order-1 quad).
    static std::shared_ptr<DGMesh> make_quad_order1_mesh() {
        auto mesh = MeshCreator::create_rectangular_mesh(0.1, /*use_triangles=*/false, 0.0, 1.0,
                                                          0.0, 0.1);
        auto space = std::make_shared<DGSpace>("quad", 1);
        mesh->initialize_dg_space(space, 4);
        return mesh;
    }

    static Vec4 jump_ic(const Vec2& x) {
        Vec4 primitive =
            (x[0] < 0.5) ? Vec4{1.0, 0.0, 0.0, 1.0} : Vec4{0.125, 0.0, 0.0, 0.1};
        return primitive_to_conserved(primitive, 1.4);
    }
};

}  // namespace

TEST_F(CompressibleDGSolverLimiterTest, ThrowsOnTriangleMesh) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.2, /*use_triangles=*/true, 0.0, 1.0, 0.0,
                                                     0.2);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space, 4);

    EulerDGSolver solver(mesh, 1.4);
    EXPECT_THROW(solver.set_limiter_enabled(true), std::invalid_argument);
}

TEST_F(CompressibleDGSolverLimiterTest, ThrowsOnOrderTwoQuadMesh) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.2, /*use_triangles=*/false, 0.0, 1.0, 0.0,
                                                     0.2);
    auto space = std::make_shared<DGSpace>("quad", 2);
    mesh->initialize_dg_space(space, 4);

    EulerDGSolver solver(mesh, 1.4);
    EXPECT_THROW(solver.set_limiter_enabled(true), std::invalid_argument);
}

TEST_F(CompressibleDGSolverLimiterTest, LazyDefaultEnablesWithoutThrowingAndChangesResult) {
    constexpr double dt = 1e-4;
    constexpr double T_final = 5e-3;  // enough steps for a genuine shock to form in-cell

    auto mesh_unlimited = make_quad_order1_mesh();
    EulerDGSolver unlimited(mesh_unlimited, 1.4);
    auto frames_unlimited = unlimited.solve(jump_ic, T_final, dt, 1000);
    ASSERT_FALSE(frames_unlimited.empty());

    auto mesh_limited = make_quad_order1_mesh();
    EulerDGSolver limited(mesh_limited, 1.4);
    ASSERT_NO_THROW(limited.set_limiter_enabled(true));  // no prior set_reconstruction_technique
    auto frames_limited = limited.solve(jump_ic, T_final, dt, 1000);
    ASSERT_FALSE(frames_limited.empty());

    EXPECT_TRUE(all_finite(frames_limited.back()));

    // The lazy default (Persson-Peraire + WENO) should actually engage on this discontinuous
    // IC and produce a different result than no limiter at all.
    const auto& u_unlimited = frames_unlimited.back();
    const auto& u_limited = frames_limited.back();
    ASSERT_EQ(u_unlimited.extent(0), u_limited.extent(0));
    ASSERT_EQ(u_unlimited.extent(1), u_limited.extent(1));

    double max_diff = 0.0;
    for (int e = 0; e < static_cast<int>(u_unlimited.extent(0)); ++e) {
        for (int c = 0; c < static_cast<int>(u_unlimited.extent(1)); ++c) {
            max_diff = std::max(max_diff, std::abs(u_unlimited(e, c) - u_limited(e, c)));
        }
    }
    EXPECT_GT(max_diff, 1e-8);
}

TEST_F(CompressibleDGSolverLimiterTest, AlwaysTroubledMinmodPairingRunsAndChangesResult) {
    constexpr double dt = 1e-4;
    constexpr double T_final = 5e-4;

    auto mesh_unlimited = make_quad_order1_mesh();
    EulerDGSolver unlimited(mesh_unlimited, 1.4);
    auto frames_unlimited = unlimited.solve(jump_ic, T_final, dt, 1000);
    ASSERT_FALSE(frames_unlimited.empty());

    auto mesh_minmod = make_quad_order1_mesh();
    EulerDGSolver minmod_solver(mesh_minmod, 1.4);
    // A small tvb_m (vs. the 50.0 production default) so minmod3 actually engages within the
    // handful of steps this quick test runs, rather than every field staying under the TVB
    // tolerance the whole time.
    minmod_solver.set_reconstruction_technique(
        std::make_shared<AlwaysTroubledIndicator>(),
        std::make_shared<MinmodReconstruction>(std::make_shared<EulerXDirectionEigensystem>(4),
                                               std::make_shared<Order1QuadBasisModeMap>(),
                                               /*tvb_m=*/0.01));
    ASSERT_NO_THROW(minmod_solver.set_limiter_enabled(true));
    auto frames_minmod = minmod_solver.solve(jump_ic, T_final, dt, 1000);
    ASSERT_FALSE(frames_minmod.empty());

    EXPECT_TRUE(all_finite(frames_minmod.back()));

    const auto& u_unlimited = frames_unlimited.back();
    const auto& u_minmod = frames_minmod.back();
    double max_diff = 0.0;
    for (int e = 0; e < static_cast<int>(u_unlimited.extent(0)); ++e) {
        for (int c = 0; c < static_cast<int>(u_unlimited.extent(1)); ++c) {
            max_diff = std::max(max_diff, std::abs(u_unlimited(e, c) - u_minmod(e, c)));
        }
    }
    EXPECT_GT(max_diff, 1e-8);
}
