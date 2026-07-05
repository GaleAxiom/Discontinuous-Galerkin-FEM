#include <Eigen/Sparse>
#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/solution.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/solver/assembler.hpp>
#include <dgfem/solver/weak_form.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace dgfem;
using namespace testing;

// Mock classes for testing
class MockLaplaceWeakFormulation : public LaplaceWeakFormulation {
public:
    MOCK_METHOD(Eigen::MatrixXd, compute_volume_integral,
                ((const std::map<std::string, Eigen::MatrixXd>&), std::shared_ptr<DGSpace>),
                (const, override));
    MOCK_METHOD(Eigen::VectorXd, compute_source_integral,
                (int, (std::function<double(const Eigen::Vector2d&)>), std::shared_ptr<DGMesh>),
                (const, override));
    MOCK_METHOD((std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>),
                compute_interior_face_integral,
                (int, int, int, int, std::shared_ptr<DGMesh>, const Eigen::VectorXi&),
                (const, override));
    MOCK_METHOD(Eigen::MatrixXd, compute_boundary_face_integral,
                (int, int, std::shared_ptr<DGMesh>, std::shared_ptr<BoundaryCondition>),
                (const, override));
    MOCK_METHOD(Eigen::VectorXd, compute_boundary_rhs_integral,
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
    Eigen::MatrixXd local_vol = Eigen::MatrixXd::Identity(3, 3);
    EXPECT_CALL(*weak_form, compute_volume_integral(_, _))
        .Times(2)
        .WillRepeatedly(Return(local_vol));

    Eigen::MatrixXd K_LL = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd K_LR = Eigen::MatrixXd::Zero(3, 3);
    Eigen::MatrixXd K_RL = Eigen::MatrixXd::Zero(3, 3);
    Eigen::MatrixXd K_RR = Eigen::MatrixXd::Identity(3, 3);

    EXPECT_CALL(*weak_form, compute_interior_face_integral(_, _, _, _, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::make_tuple(K_LL, K_LR, K_RL, K_RR)));

    Eigen::MatrixXd K_bc = Eigen::MatrixXd::Identity(3, 3);
    EXPECT_CALL(*weak_form, compute_boundary_face_integral(_, _, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(K_bc));

    Eigen::VectorXd F_bc = Eigen::VectorXd::Ones(3);
    EXPECT_CALL(*weak_form, compute_boundary_rhs_integral(_, _, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(F_bc));

    Eigen::VectorXd F_src = Eigen::VectorXd::Constant(3, 0.5);
    EXPECT_CALL(*weak_form, compute_source_integral(_, _, _))
        .Times(2)
        .WillRepeatedly(Return(F_src));

    assembler.assemble([](const Eigen::Vector2d&) { return 1.0; });

    auto matrix = assembler.get_system_matrix();
    auto rhs = assembler.get_rhs();

    auto n_dofs = mesh->get_n_elements() * mesh->get_dg_space()->get_basis()->get_n_basis();
    // Check matrix and vector properties
    EXPECT_EQ(static_cast<int>(matrix->getGlobalNumRows()), n_dofs);
    EXPECT_EQ(static_cast<int>(matrix->getGlobalNumCols()), n_dofs);
    EXPECT_EQ(static_cast<int>(rhs->getGlobalLength()), n_dofs);
    EXPECT_GT(matrix->getGlobalNumEntries(), 0u);
    EXPECT_GT(tpetra_to_eigen(*rhs).norm(), 0);
}

TEST(DGAssemblerTest, DistributeSolution) {
    auto mesh = create_test_mesh();
    auto weak_form = std::make_shared<MockLaplaceWeakFormulation>();
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);
    DGAssembler assembler(mesh, weak_form);

    auto n_dofs = mesh->get_n_elements() * mesh->get_dg_space()->get_basis()->get_n_basis();

    // Create a test solution vector
    Eigen::VectorXd solution_vec = Eigen::VectorXd::LinSpaced(n_dofs, 0, 1);
    auto solution_tpetra = eigen_to_tpetra(solution_vec, make_serial_map(n_dofs));

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

// TEST(DGAssemblerTest, AssembleEulerResidual) {
//     auto mesh = create_test_mesh();
//     auto weak_form = std::make_shared<MockEulerWeakFormulation>();
//     DGAssembler assembler(mesh, weak_form);

//     // Create initial solution
//     int n_vars = 4;  // For Euler equations
//     DGSolution u_sol(mesh->get_n_elements(), mesh->get_dg_space()->get_basis()->get_n_basis(),
//     n_vars);

//     // Set mock solution coefficients
//     for (int e = 0; e < mesh->get_n_elements(); ++e) {
//         for (int v = 0; v < n_vars; ++v) {
//             Eigen::VectorXd coeffs =
//             Eigen::VectorXd::Ones(mesh->get_dg_space()->get_basis()->get_n_basis());
//             u_sol.set_element_coeffs(e, v, coeffs);
//         }
//     }

//     // Set up mock expectations
//     int n_dofs_per_var = mesh->get_dg_space()->get_basis()->get_n_basis();
//     EXPECT_CALL(*weak_form, residual_volume_integral(_, _, _))
//         .Times(mesh->get_n_elements())
//         .WillRepeatedly(Return(Eigen::VectorXd::Ones(n_dofs_per_var * n_vars)));

//     EXPECT_CALL(*weak_form, residual_face_integral(_, _, _, _, _, _))
//         .Times(AtLeast(1))
//         .WillRepeatedly(Return(Eigen::VectorXd::Ones(n_dofs_per_var * n_vars)));

//     Eigen::VectorXd residual = assembler.assemble_euler_system(u_sol);

//     // Check residual properties
//     EXPECT_EQ(residual.size(), u_sol.get_global_coeffs().size());
//     EXPECT_GT(residual.norm(), 0);
// }