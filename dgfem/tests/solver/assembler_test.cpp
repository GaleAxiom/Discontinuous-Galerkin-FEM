#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/solution.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/assembler.hpp>
#include <dgfem/solver/weak_form.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace dgfem;
using namespace testing;

namespace {
DView2 make_identity(int n) {
    DView2 m("identity", n, n);
    for (int i = 0; i < n; ++i)
        m(i, i) = 1.0;
    return m;
}
DView2 make_zero_matrix(int n, int m) {
    return DView2("zero", n, m);
}
DView1 make_ones(int n) {
    DView1 v("ones", n);
    for (int i = 0; i < n; ++i)
        v(i) = 1.0;
    return v;
}
DView1 make_constant(int n, double val) {
    DView1 v("const", n);
    for (int i = 0; i < n; ++i)
        v(i) = val;
    return v;
}
DView1 make_linspaced(int n, double lo, double hi) {
    DView1 v("linspaced", n);
    for (int i = 0; i < n; ++i) {
        v(i) = (n == 1) ? lo : lo + (hi - lo) * i / (n - 1);
    }
    return v;
}
}  // namespace

// Mock classes for testing
class MockLaplaceWeakFormulation : public LaplaceWeakFormulation {
public:
    MOCK_METHOD(DView2, compute_volume_integral,
                ((const std::map<std::string, DView2>&), std::shared_ptr<DGSpace>),
                (const, override));
    MOCK_METHOD(DView1, compute_source_integral,
                (int, (std::function<double(const Vec2&)>), std::shared_ptr<DGMesh>),
                (const, override));
    MOCK_METHOD((std::tuple<DView2, DView2, DView2, DView2>), compute_interior_face_integral,
                (int, int, int, int, std::shared_ptr<DGMesh>, const IView1&), (const, override));
    MOCK_METHOD(DView2, compute_boundary_face_integral,
                (int, int, std::shared_ptr<DGMesh>, std::shared_ptr<BoundaryCondition>),
                (const, override));
    MOCK_METHOD(DView1, compute_boundary_rhs_integral,
                (int, int, std::shared_ptr<DGMesh>, std::shared_ptr<BoundaryCondition>),
                (const, override));
};

class MockEulerWeakFormulation : public EulerWeakFormulation {
public:
    // Note: Euler uses residual-based assembly, not standard matrix assembly
    // Only mock the residual methods if needed for testing
};

TEST(DGAssemblerTest, Construction) {
    auto mesh = create_test_mesh();
    auto weak_form = std::make_shared<MockLaplaceWeakFormulation>();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    auto n_dofs = mesh->get_n_elements() * mesh->get_dg_space()->get_basis()->get_n_basis();

    EXPECT_NO_THROW({
        DGAssembler assembler(mesh, weak_form);
        EXPECT_EQ(n_dofs,
                  mesh->get_n_elements() * mesh->get_dg_space()->get_basis()->get_n_basis());
    });
}

TEST(DGAssemblerTest, AssembleLaplaceSystem) {
    auto mesh = create_test_mesh();
    auto weak_form = std::make_shared<MockLaplaceWeakFormulation>();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    DGAssembler assembler(mesh, weak_form);

    // Set up mock expectations
    DView2 local_vol = make_identity(3);
    EXPECT_CALL(*weak_form, compute_volume_integral(_, _))
        .Times(2)
        .WillRepeatedly(Return(local_vol));

    DView2 K_LL = make_identity(3);
    DView2 K_LR = make_zero_matrix(3, 3);
    DView2 K_RL = make_zero_matrix(3, 3);
    DView2 K_RR = make_identity(3);

    EXPECT_CALL(*weak_form, compute_interior_face_integral(_, _, _, _, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::make_tuple(K_LL, K_LR, K_RL, K_RR)));

    DView2 K_bc = make_identity(3);
    EXPECT_CALL(*weak_form, compute_boundary_face_integral(_, _, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(K_bc));

    DView1 F_bc = make_ones(3);
    EXPECT_CALL(*weak_form, compute_boundary_rhs_integral(_, _, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(F_bc));

    DView1 F_src = make_constant(3, 0.5);
    EXPECT_CALL(*weak_form, compute_source_integral(_, _, _))
        .Times(2)
        .WillRepeatedly(Return(F_src));

    assembler.assemble([](const Vec2&) { return 1.0; });

    auto matrix = assembler.get_system_matrix();
    auto rhs = assembler.get_rhs();

    auto n_dofs = mesh->get_n_elements() * mesh->get_dg_space()->get_basis()->get_n_basis();
    // Check matrix and vector properties
    EXPECT_EQ(static_cast<int>(matrix->getGlobalNumRows()), n_dofs);
    EXPECT_EQ(static_cast<int>(matrix->getGlobalNumCols()), n_dofs);
    EXPECT_EQ(static_cast<int>(rhs->getGlobalLength()), n_dofs);
    EXPECT_GT(matrix->getGlobalNumEntries(), 0u);
    EXPECT_GT(norm(tpetra_to_view(*rhs)), 0);
}

TEST(DGAssemblerTest, DistributeSolution) {
    auto mesh = create_test_mesh();
    auto weak_form = std::make_shared<MockLaplaceWeakFormulation>();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    DGAssembler assembler(mesh, weak_form);

    auto n_dofs = mesh->get_n_elements() * mesh->get_dg_space()->get_basis()->get_n_basis();

    // Create a test solution vector
    DView1 solution_vec = make_linspaced(n_dofs, 0, 1);
    auto solution_tpetra = view_to_tpetra(solution_vec, make_serial_map(n_dofs));

    EXPECT_NO_THROW(assembler.distribute_solution(*solution_tpetra));

    // Verify solution is correctly distributed
    auto mesh_solution = mesh->get_solution();
    int n_basis = mesh->get_dg_space()->get_basis()->get_n_basis();
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        auto coeffs = mesh_solution->get_element_coeffs(e, 0);
        for (int i = 0; i < n_basis; ++i) {
            EXPECT_NEAR(coeffs(i), solution_vec(e * n_basis + i), 1e-12);
        }
    }
}
